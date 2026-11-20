"""
Helpers of the extras API (`extras.extras`): tiling and argument checks of the partition samplers.
"""
from __future__ import annotations

import numpy as np

from ..lattice import integer_grid


def tile(x, repeat: int, flatten: bool):
    N, D = x.shape
    offsets = integer_grid(np.arange(repeat), D) / repeat
    tiled = x[None] / repeat + offsets[:, None, :]
    return tiled.reshape(-1, D) if flatten else tiled.reshape(*([repeat] * D), N, D)


def depth(N: int) -> int:
    """log2(N), N must be a power of 2."""
    d = int(np.log2(N))
    if 2 ** d != N:
        raise ValueError(f"N must be a power of 2 (got N={N}): every recursion step splits each cell in two")
    return d

