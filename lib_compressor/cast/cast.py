import os.path

import torch
from tqdm import tqdm

from .. import STATE_DICT, load, save
from . import DTYPE


def _cast(sd: STATE_DICT, dtype: torch.dtype) -> STATE_DICT:
    cast_sd = {}

    _keys = list(sd.keys())
    for key in tqdm(_keys):
        cast_sd[key] = sd.pop(key).to(dtype)

    return cast_sd


@torch.inference_mode()
def cast_to_dtype(model: str, mode: str):
    path: str = model.strip('"').strip()
    sd, _ = load(path)

    new_sd = _cast(sd, DTYPE[mode])
    del sd

    file = os.path.splitext(path)[0]

    save(f"{file}-{mode}.safetensors", new_sd, None)
