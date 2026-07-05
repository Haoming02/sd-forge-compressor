# https://github.com/bedovyy/comfy-dit-quantizer/blob/main/utils/convrot.py

import math

import torch

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def build_hadamard(
    size: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:

    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    H4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )

    H = H4
    current_size = 4

    while current_size < size:
        H = torch.kron(H, H4)
        current_size *= 4

    H_normalized = H / (size**0.5)
    _HADAMARD_CACHE[cache_key] = H_normalized

    return H_normalized


def rotate_weight(
    weight: torch.Tensor, H: torch.Tensor, group_size: int
) -> torch.Tensor:
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size
    W_grouped = weight.view(out_f, n_groups, group_size)
    H_t = H.T.to(dtype=weight.dtype, device=weight.device)
    W_rot = torch.matmul(W_grouped, H_t)
    return W_rot.reshape(out_f, in_f)
