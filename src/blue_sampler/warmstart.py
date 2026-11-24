"""
Cheap, well spread point sets in [0, 1)^D: starting points of the solvers, atoms of the partitions.

    sobol              scrambled Sobol sequence (scipy)
    korobov_lattice    rank-1 lattice (structure_factor_v2.lattices)
    jittered_lattice   Korobov lattice, every point jittered in its cell (method "latjit",
                       structure_factor_v2.lattices)
    warmstart_points   one of the above by name, or a user array

The lattices live in structure_factor_v2: this module is blue_sampler's only internal import
of it (everything else goes through here).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from .lattices import jittered_lattice, curled_lattice, sobol

__all__ = ["WarmstartMethod", "warmstart_points", "sobol", "jittered_lattice", "curled_lattice"]

WarmstartMethod = Literal["latjit", "latcurl", "Sobol"]


def warmstart_points(N: int, D: int, method: WarmstartMethod | NDArray | None = None,
                     seed: int | None = None) -> NDArray:
    """
    N points in [0, 1)^D. method: None or "Sobol", "latjit" (Korobov lattice, random
    shift), or an (N, D) array, returned as a copy.
    """
    if isinstance(method, np.ndarray):
        if method.shape != (N, D):
            raise ValueError(f"warmstart array must have shape {(N, D)}, got {method.shape}")
        return method.copy()
    if method in (None, "Sobol"):
        return sobol(N, D, seed=seed)
    if method == "latjit":
        return jittered_lattice(N, D, seed=seed)
    if method == "latcurl":
        return curled_lattice(N, D, seed=seed)
    raise ValueError(f"unknown warmstart {method!r}: expected a {WarmstartMethod} or an (N, D) array")
