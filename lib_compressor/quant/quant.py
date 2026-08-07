import inspect
import os.path
from json import dumps
from typing import Final

import torch
from comfy_kitchen.tensor import (
    AsymW4A8Int8Layout,
    QuantizedLayout,
    TensorCoreConvRotW4A4Layout,
    TensorCoreFP8Layout,
    TensorCoreMXFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
)
from tqdm import tqdm

from backend.memory_management import get_torch_device, soft_empty_cache
from backend.patcher.lora import string_to_seed

from .. import STATE_DICT, load, save
from . import MODELS

COMMON_EXCL: Final[tuple[str]] = (
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

LAYER_EXCL: list[str] = []


def _encode(info: dict[str, str]) -> torch.Tensor:
    return torch.tensor(list(dumps(info).encode("utf-8")), dtype=torch.uint8)


def _parse_layers(layers: list[str]):
    import re

    pattern = re.compile(r"(layer|block)s?\.(\d+)\.")

    min_, max_ = 999, -1
    min_excl, max_excl = None, None

    for key in layers:
        if m := pattern.search(key):
            idx = int(m.group(2))

            if idx > max_:
                max_ = idx
                max_excl = m.group(0)
            if idx < min_:
                min_ = idx
                min_excl = m.group(0)

    if min_excl and max_excl:
        LAYER_EXCL.extend([min_excl, max_excl])


def _parse_parameters(layout: QuantizedLayout) -> dict[str, int]:
    params = {}

    sig = inspect.signature(layout)
    for name, param in sig.parameters.items():
        if name == "stochastic_rounding":
            continue
        if isinstance(param.default, int):
            params[name] = param.default

    return params


def _filter(key: str, weight: torch.Tensor, group_size: int) -> bool:
    if not key.endswith(".weight"):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.ndim != 2:
        return False
    if any(excl in key for excl in LAYER_EXCL):
        return False
    if any(excl in key.lower() for excl in COMMON_EXCL):
        return False

    return weight.size(0) % group_size == 0


def _quant(
    state_dict: STATE_DICT,
    layout: QuantizedLayout,
    params: dict,
    quant_info: torch.Tensor,
) -> STATE_DICT:
    quant_sd = {}

    device = get_torch_device()
    group_size = params.get("convrot_groupsize", 64)
    rounding = "stochastic_rounding" in inspect.signature(layout.quantize).parameters

    _keys = list(state_dict.keys())
    for key in tqdm(_keys):
        weight = state_dict.pop(key)

        if not _filter(key, weight, group_size):
            quant_sd[key] = weight.to(dtype=torch.bfloat16)
            continue

        weight = weight.to(device=device)

        if rounding:
            params["stochastic_rounding"] = string_to_seed(key)

        _qdata, _params = layout.quantize(weight, **params)
        mapping: dict[str, torch.Tensor] = layout.state_dict_tensors(_qdata, _params)

        for suffix, tensor in mapping.items():
            quant_sd[key + suffix] = tensor.cpu()

        quant_sd[key.replace(".weight", ".comfy_quant")] = quant_info.clone()

    return quant_sd


@torch.inference_mode()
def quant_to_dtype(model: str, mode: str, exclude: bool):
    path: str = MODELS[model]
    sd, meta = load(path)
    LAYER_EXCL.clear()

    if exclude:
        _parse_layers(list(sd.keys()))

    match mode:
        case "fp8_scaled":
            layout = TensorCoreFP8Layout
            info = {"format": "float8_e4m3fn"}
            params = {"scale": "recalculate", "dtype": torch.float8_e4m3fn}
        case "nvfp4":
            layout = TensorCoreNVFP4Layout
            info = {"format": "nvfp4"}
            params = {"scale": "recalculate"}
        case "mxfp8":
            layout = TensorCoreMXFP8Layout
            info = {"format": "mxfp8"}
            params = {}
        case "int8":
            layout = TensorWiseINT8Layout
            info = {"format": "int8_tensorwise"}
            params = {
                "is_weight": True,
                "per_channel": True,
                "convrot": False,
            }
        case "int8_convrot":
            layout = TensorWiseINT8Layout
            defaults = _parse_parameters(layout)
            info = {"format": "int8_tensorwise", "convrot": True, **defaults}
            params = {
                "is_weight": True,
                "per_channel": True,
                "convrot": True,
                **defaults,
            }
        case "w4a4_convrot":
            layout = TensorCoreConvRotW4A4Layout
            defaults = _parse_parameters(layout)
            info = {"format": "convrot_w4a4", **defaults}
            params = {**defaults, "linear_dtype": "int4"}
        case "w4a8_convrot":
            layout = AsymW4A8Int8Layout
            defaults = _parse_parameters(layout)
            info = {"format": "asym_w4a8_int8", **defaults}
            params = {**defaults, "scale_dtype": torch.float8_e4m3fn}

    new_sd = _quant(sd, layout, params, _encode(info))
    del sd

    soft_empty_cache()

    file = os.path.splitext(path)[0]

    save(f"{file}-{mode}.safetensors", new_sd, meta)
