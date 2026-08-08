import os.path
from typing import Any, Optional, TypeAlias

import gradio as gr
import torch
from safetensors.torch import save_file

from backend.state_dict import convert_quantization, detect_quantization
from backend.utils import load_torch_file

STATE_DICT: TypeAlias = dict[str, torch.Tensor]
METADATA: TypeAlias = Optional[dict[str, Any]]


def _dequantize(sd: STATE_DICT):
    import json

    c = next(iter(key for key in sd.keys() if key.endswith(".comfy_quant")), None)
    layer_conf = json.loads(sd[c].numpy().tobytes())

    if layer_conf["format"] not in ("float8_e4m3fn", "float8_e5m2"):
        raise gr.Error("Model is already quantized (only fp8_scaled is supported)...")
    else:
        gr.Warning("Dequantizing...", 1.0)

    from tqdm import tqdm

    from backend.quant_ops import TensorCoreFP8Layout

    _keys = list(sd.keys())
    for key in tqdm(_keys):
        if not key.endswith(".weight"):
            continue

        if (scale := (key + "_scale")) not in _keys:
            continue

        params = TensorCoreFP8Layout.Params(sd.pop(scale), torch.bfloat16, None)
        sd[key] = TensorCoreFP8Layout.dequantize(sd.pop(key), params)


def load(path: str | os.PathLike) -> tuple[STATE_DICT, METADATA]:
    if not os.path.isfile(path):
        raise gr.Error(f'Invalid Path: "{path}"')

    if not path.endswith((".ckpt", ".pt", ".pth", ".bin", ".safetensors", ".sft")):
        raise gr.Error(f'Non-Supported File Format: "{os.path.splitext(path)[1]}"')

    sd, meta = load_torch_file(path, return_metadata=True)
    sd, meta = convert_quantization(sd, meta)

    if detect_quantization(sd) is not None:
        _dequantize(sd)

    meta.pop("_quantization_metadata", None)
    return sd, meta


def save(path: str | os.PathLike, sd: STATE_DICT, meta: METADATA):
    if os.path.isfile(path):
        gr.Warning("File already exists ; Overriding...")

    save_file(sd, path, metadata=meta if meta else None)

    gr.Info("Done")
