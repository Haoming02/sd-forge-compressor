import os.path
from json import dumps
from typing import Final

import torch
from comfy_kitchen import quantize_per_tensor_fp8
from comfy_kitchen.float_utils import F8_E4M3_MAX
from comfy_kitchen.tensor import (
    TensorCoreConvRotW4A4Layout,
    TensorCoreMXFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
)
from tqdm import tqdm

from backend.memory_management import get_torch_device, soft_empty_cache

from .. import STATE_DICT, load, save
from . import MODELS

CONVROT_GROUPSIZE: Final[int] = 256
QUANT_GROUPSIZE: Final[int] = 64

EXCL: Final[tuple[str]] = (
    "embed",
    "bias",
    "norm",
    "scale",
    "llm",
    "adaln",
    "first_stage_model",
    "cond_stage_model",
    "vae",
    "text",
    "time",
)


def _encode(info: dict[str, str]) -> torch.Tensor:
    return torch.tensor(list(dumps(info).encode("utf-8")), dtype=torch.uint8)


def _scale_amax(w: torch.Tensor, max_value: float) -> torch.Tensor:
    return w.abs().max().float().div(max_value).clamp(min=1e-8)


def _filter(key: str, weight: torch.Tensor, *, group_size: int = 64) -> bool:
    if not key.endswith(".weight"):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.ndim != 2:
        return False
    if any(excl in key.lower() for excl in EXCL):
        return False

    in_features: int = weight.size(0)
    return in_features > group_size and in_features % group_size == 0


def quant_fp8(state_dict: STATE_DICT) -> STATE_DICT:
    quant_sd = {}
    quant_info = {"format": "float8_e4m3fn"}

    device = get_torch_device()

    _keys = list(state_dict.keys())
    for key in tqdm(_keys):
        weight = state_dict.pop(key)

        if not _filter(key, weight):
            if weight.dtype is torch.float32:
                weight = weight.to(dtype=torch.float16)
            quant_sd[key] = weight
            continue

        weight = weight.to(device=device)
        weight_scale = _scale_amax(weight, F8_E4M3_MAX)
        weight_quantized = quantize_per_tensor_fp8(weight, weight_scale)

        quant_sd[key] = weight_quantized.cpu()
        quant_sd[key.replace(".weight", ".weight_scale")] = weight_scale.cpu()
        quant_sd[key.replace(".weight", ".comfy_quant")] = _encode(quant_info)

    return quant_sd


def quant_nvfp4(state_dict: STATE_DICT) -> STATE_DICT:
    quant_sd = {}
    quant_info = {"format": "nvfp4"}

    device = get_torch_device()

    _keys = list(state_dict.keys())
    for key in tqdm(_keys):
        weight = state_dict.pop(key)

        if not _filter(key, weight):
            if weight.dtype is torch.float32:
                weight = weight.to(dtype=torch.float16)
            quant_sd[key] = weight
            continue

        weight = weight.to(device=device)

        qdata, params = TensorCoreNVFP4Layout.quantize(weight)

        quant_sd[key] = qdata.cpu()
        quant_sd[key.replace(".weight", ".weight_scale")] = params.block_scale.cpu()
        quant_sd[key.replace(".weight", ".weight_scale_2")] = params.scale.cpu()
        quant_sd[key.replace(".weight", ".comfy_quant")] = _encode(quant_info)

    return quant_sd


def quant_mxfp8(state_dict: STATE_DICT) -> STATE_DICT:
    quant_sd = {}
    quant_info = {"format": "mxfp8"}

    device = get_torch_device()

    _keys = list(state_dict.keys())
    for key in tqdm(_keys):
        weight = state_dict.pop(key)

        if not _filter(key, weight):
            if weight.dtype is torch.float32:
                weight = weight.to(dtype=torch.float16)
            quant_sd[key] = weight
            continue

        weight = weight.to(device=device)

        qdata, params = TensorCoreMXFP8Layout.quantize(weight)

        quant_sd[key] = qdata.cpu()
        quant_sd[key.replace(".weight", ".weight_scale")] = params.scale.cpu()
        quant_sd[key.replace(".weight", ".comfy_quant")] = _encode(quant_info)

    return quant_sd


def quant_int8(state_dict: STATE_DICT) -> STATE_DICT:
    quant_sd = {}
    quant_info = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": CONVROT_GROUPSIZE,
    }

    device = get_torch_device()

    _keys = list(state_dict.keys())
    for key in tqdm(_keys):
        weight = state_dict.pop(key)

        if not _filter(key, weight, group_size=CONVROT_GROUPSIZE):
            quant_sd[key] = weight.to(torch.bfloat16)
            continue

        weight = weight.to(device=device)

        qdata, params = TensorWiseINT8Layout.quantize(
            weight,
            per_channel=True,
            convrot=True,
            convrot_groupsize=CONVROT_GROUPSIZE,
        )

        quant_sd[key] = qdata.cpu()
        quant_sd[key.replace(".weight", ".weight_scale")] = params.scale.cpu()
        quant_sd[key.replace(".weight", ".comfy_quant")] = _encode(quant_info)

    return quant_sd


def quant_int4(state_dict: STATE_DICT) -> STATE_DICT:
    quant_sd = {}
    quant_info = {
        "format": "convrot_w4a4",
        "convrot_groupsize": CONVROT_GROUPSIZE,
        "quant_group_size": QUANT_GROUPSIZE,
    }

    device = get_torch_device()

    _keys = list(state_dict.keys())
    for key in tqdm(_keys):
        weight = state_dict.pop(key)

        if not _filter(key, weight, group_size=max(CONVROT_GROUPSIZE, QUANT_GROUPSIZE)):
            quant_sd[key] = weight.to(dtype=torch.bfloat16)
            continue

        weight = weight.to(device=device)

        qdata, params = TensorCoreConvRotW4A4Layout.quantize(
            weight,
            convrot_groupsize=CONVROT_GROUPSIZE,
            quant_group_size=QUANT_GROUPSIZE,
        )

        quant_sd[key] = qdata.cpu()
        quant_sd[key.replace(".weight", ".weight_scale")] = params.scale.cpu()
        quant_sd[key.replace(".weight", ".comfy_quant")] = _encode(quant_info)

    return quant_sd


@torch.inference_mode()
def quant_to_dtype(model: str, mode: str):
    path: str = MODELS[model]
    sd, meta = load(path)

    match mode:
        case "fp8_scaled":
            new_sd = quant_fp8(sd)
        case "nvfp4":
            new_sd = quant_nvfp4(sd)
        case "mxfp8":
            new_sd = quant_mxfp8(sd)
        case "int8_convrot":
            new_sd = quant_int8(sd)
        case "convrot_w4a4":
            new_sd = quant_int4(sd)

    del sd
    soft_empty_cache()

    file = os.path.splitext(path)[0]

    save(f"{file}-{mode}.safetensors", new_sd, meta)
