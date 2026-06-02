import os.path
from json import dumps

import gradio as gr
import torch
from comfy_kitchen import quantize_per_tensor_fp8
from comfy_kitchen.float_utils import F8_E4M3_MAX
from tqdm import tqdm

from backend.memory_management import get_torch_device
from modules_forge.packages.huggingface_guess import guess

from .. import STATE_DICT, load, save
from . import MODELS

TARGET: str = None
INCL: tuple[str] = None
EXCL: tuple[str] = None


def _encode(info: dict[str, str]) -> torch.Tensor:
    return torch.tensor(list(dumps(info).encode("utf-8")), dtype=torch.uint8)


def scale_amax(w: torch.Tensor, max_value: float):
    return torch.amax(w.abs()).to(dtype=torch.float32) / max_value


def filter(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    if TARGET not in key:
        return False
    if any(l in key for l in EXCL):
        return False
    return any(l in key for l in INCL)


def _quant_fp8(state_dict: STATE_DICT) -> STATE_DICT:
    quant_sd = {}
    quant_info = {"format": "float8_e4m3fn"}

    device = get_torch_device()

    _keys = list(state_dict.keys())
    for key in tqdm(_keys):
        weight = state_dict.pop(key)

        if not filter(key):
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


@torch.inference_mode()
def quant_to_dtype(model: str, mode: str):
    if mode != "fp8_scaled":
        raise gr.Error('Only "fp8_scaled" is supported currently...')

    path: str = MODELS[model]
    sd, meta = load(path)

    try:
        if (mdl := guess(sd)) is None:
            raise SystemError
    except Exception:
        raise gr.Error("Unrecognized Model...")

    if not mdl.huggingface_repo.endswith("Anima"):
        raise gr.Error('Only "Anima" is supported currently...')

    global TARGET, INCL, EXCL

    TARGET = "blocks."
    INCL = ("q_proj", "k_proj", "v_proj", "o_proj", "output_proj", ".mlp")
    EXCL = ("llm_adapter",)

    new_sd = _quant_fp8(sd)
    del sd

    file = os.path.splitext(path)[0]

    save(f"{file}-{mode}.safetensors", new_sd, meta)
