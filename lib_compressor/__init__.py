import os.path
from typing import Any, Optional, TypeAlias

import gradio as gr
import torch
from safetensors.torch import save_file

from backend.utils import load_torch_file

STATE_DICT: TypeAlias = dict[str, torch.Tensor]
METADATA: TypeAlias = Optional[dict[str, Any]]


def _is_quant(sd: STATE_DICT, meta: METADATA) -> bool:
    if "_quantization_metadata" in (meta or {}):
        return True
    if any("comfy_quant" in key for key in sd):
        return True
    if any("weight_scale" in key for key in sd):
        return True
    if any("scaled_fp8" in key for key in sd):
        return True

    return False


def load(path: str | os.PathLike) -> tuple[STATE_DICT, METADATA]:
    if not os.path.isfile(path):
        raise gr.Error(f'Invalid Path: "{path}"')

    if not path.endswith((".ckpt", ".pt", ".pth", ".bin", ".safetensors", ".sft")):
        raise gr.Error(f'Non-Supported File Format: "{os.path.splitext(path)[1]}"')

    sd, meta = load_torch_file(path, return_metadata=True)

    if _is_quant(sd, meta):
        raise gr.Error("Model is already quantized...")

    return sd, meta


def save(path: str | os.PathLike, sd: STATE_DICT, meta: METADATA):
    if os.path.isfile(path):
        gr.Warning("File already exists ; Overriding...")

    save_file(sd, path, metadata=meta if meta else None)

    gr.Info("Done")
