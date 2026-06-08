import os.path
from json import dumps

import torch
from comfy_kitchen import quantize_mxfp8, quantize_per_tensor_fp8
from comfy_kitchen.float_utils import (
    F4_E2M1_MAX,
    F8_E4M3_MAX,
    _f32_to_floatx_unpacked,
    _float8_round,
    pack_uint4,
    to_blocked,
)
from tqdm import tqdm

from backend.memory_management import get_torch_device, soft_empty_cache

from .. import STATE_DICT, load, save
from . import MODELS

EXCL = ("embed", "norm", "first_stage_model", "cond_stage_model", "vae", "text_encoder")


def _encode(info: dict[str, str]) -> torch.Tensor:
    return torch.tensor(list(dumps(info).encode("utf-8")), dtype=torch.uint8)


def scale_amax(w: torch.Tensor, max_value: float):
    return torch.amax(w.abs()).to(dtype=torch.float32) / max_value


def filter(key: str, weight: torch.Tensor) -> bool:
    if not key.endswith(".weight"):
        return False
    if any(excl in key for excl in EXCL):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.ndim != 2:
        return False

    return weight.size(0) % 32 == 0 and weight.size(1) % 32 == 0


def _quant_fp8(state_dict: STATE_DICT) -> STATE_DICT:
    quant_sd = {}
    quant_info = {"format": "float8_e4m3fn"}

    device = get_torch_device()

    _keys = list(state_dict.keys())
    for key in tqdm(_keys):
        weight = state_dict.pop(key)

        if not filter(key, weight):
            if weight.dtype is torch.float32:
                weight = weight.to(dtype=torch.float16)
            quant_sd[key] = weight
            continue

        weight = weight.to(device=device)
        weight_scale = scale_amax(weight, F8_E4M3_MAX)
        weight_quantized = quantize_per_tensor_fp8(weight, weight_scale)

        quant_sd[key] = weight_quantized.cpu()
        quant_sd[key.replace(".weight", ".weight_scale")] = weight_scale.cpu()
        quant_sd[key.replace(".weight", ".comfy_quant")] = _encode(quant_info)

    return quant_sd


def _nvfp4(x: torch.Tensor, scale_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = 16
    orig_shape = x.shape

    x = x.reshape(orig_shape[0], -1, block_size)
    max_abs = torch.amax(torch.abs(x), dim=-1)

    block_scale = max_abs.to(torch.float32) / F4_E2M1_MAX
    del max_abs

    scaled_block_scales = block_scale / scale_2
    del block_scale

    scaled_block_scales_fp8 = torch.clamp(scaled_block_scales, max=F8_E4M3_MAX)
    del scaled_block_scales

    scaled_block_scales_fp32 = _float8_round(scaled_block_scales_fp8)
    total_scale = scale_2 * scaled_block_scales_fp32
    del scaled_block_scales_fp32, scale_2

    zero_scale_mask = total_scale == 0
    total_scale_safe = torch.where(
        zero_scale_mask, torch.ones_like(total_scale), total_scale
    )
    del total_scale

    data_scaled = x.to(torch.float32)
    del x

    torch.cuda.empty_cache()
    data_scaled /= total_scale_safe.unsqueeze(-1)
    del total_scale_safe

    zero_mask_expanded = zero_scale_mask.unsqueeze(-1)
    del zero_scale_mask

    data_scaled.masked_fill_(zero_mask_expanded, 0.0)
    del zero_mask_expanded

    out_scales = scaled_block_scales_fp8
    del scaled_block_scales_fp8

    data_scaled.clamp_(-6.0, 6.0)
    data_scaled = data_scaled.view(orig_shape)

    rows = data_scaled.shape[0]
    chunk = 128
    first_lp = pack_uint4(_f32_to_floatx_unpacked(data_scaled[:1], 2, 1))
    data_lp = torch.empty(
        rows, *first_lp.shape[1:], dtype=first_lp.dtype, device=data_scaled.device
    )
    del first_lp

    for i in range(0, rows, chunk):
        lp_chunk = _f32_to_floatx_unpacked(data_scaled[i : i + chunk], 2, 1)
        data_lp[i : i + chunk] = pack_uint4(lp_chunk)
        del lp_chunk
    del data_scaled
    torch.cuda.empty_cache()

    blocked_scales = to_blocked(out_scales.to(torch.float8_e4m3fn), flatten=False)
    del out_scales

    return data_lp, blocked_scales


def _quant_nvfp4(state_dict: STATE_DICT) -> STATE_DICT:
    quant_sd = {}
    quant_info = {"format": "nvfp4"}

    device = get_torch_device()

    _keys = list(state_dict.keys())
    for key in tqdm(_keys):
        weight = state_dict.pop(key)

        if not filter(key, weight):
            if weight.dtype is torch.float32:
                weight = weight.to(dtype=torch.float16)
            quant_sd[key] = weight
            continue

        weight = weight.to(device=device)
        weight_scale_2 = scale_amax(weight, F8_E4M3_MAX * F4_E2M1_MAX)
        weight_quantized, weight_scale = _nvfp4(weight, weight_scale_2)

        quant_sd[key] = weight_quantized.cpu()
        quant_sd[key.replace(".weight", ".weight_scale")] = weight_scale.cpu()
        quant_sd[key.replace(".weight", ".weight_scale_2")] = weight_scale_2.cpu()
        quant_sd[key.replace(".weight", ".comfy_quant")] = _encode(quant_info)

    return quant_sd


def _quant_mxfp8(state_dict: STATE_DICT) -> STATE_DICT:
    quant_sd = {}
    quant_info = {"format": "mxfp8"}

    device = get_torch_device()

    _keys = list(state_dict.keys())
    for key in tqdm(_keys):
        weight = state_dict.pop(key)

        if not filter(key, weight):
            if weight.dtype is torch.float32:
                weight = weight.to(dtype=torch.float16)
            quant_sd[key] = weight
            continue

        weight = weight.to(device=device)
        weight_quantized, scale = quantize_mxfp8(weight)
        weight_scale = scale.view(torch.uint8)

        quant_sd[key] = weight_quantized.cpu()
        quant_sd[key.replace(".weight", ".weight_scale")] = weight_scale.cpu()
        quant_sd[key.replace(".weight", ".comfy_quant")] = _encode(quant_info)

    return quant_sd


@torch.inference_mode()
def quant_to_dtype(model: str, mode: str):
    path: str = MODELS[model]
    sd, meta = load(path)

    match mode:
        case "fp8_scaled":
            new_sd = _quant_fp8(sd)
        case "nvfp4":
            new_sd = _quant_nvfp4(sd)
        case "mxfp8":
            new_sd = _quant_mxfp8(sd)

    del sd
    soft_empty_cache()

    file = os.path.splitext(path)[0]

    save(f"{file}-{mode}.safetensors", new_sd, meta)
