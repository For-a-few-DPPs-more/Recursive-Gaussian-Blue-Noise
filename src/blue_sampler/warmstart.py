"""
Initial point clouds in [0, 1)^D (warm starts) and cheap low-discrepancy samplers.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .lattice import korobov_lattice

WarmstartMethod = Literal["Goodlattice", "Sobol"]


def sobol(N: int = 2 ** 15, D: int = 2, seed: int | None = None) -> NDArray:
    """
    N points of a scrambled Sobol sequence in [0, 1)^D (wrapper of ``scipy.stats.qmc.Sobol``).

    For best uniformity N should be a power of 2 (SciPy warns otherwise).
    """
    from scipy.stats import qmc
    return qmc.Sobol(d=D, seed=seed).random(N)


def jittered_lattice(N: int, D: int) -> NDArray:
    """A Korobov lattice, randomly translated, each point jittered within its cell."""
    x = korobov_lattice(N, D) + np.random.rand(1, D)
    x = x + (np.random.rand(N, D) - 0.5) / N ** (1 / D)
    return x % 1.0


def warmstart_points(
    N: int,
    D: int,
    method: WarmstartMethod | NDArray | None = None,
    seed: int | None = None,
) -> NDArray:
    """
    Generate an initial point cloud in [0, 1)^D.

    Parameters
    ----------
    N, D : int
        Number of points and ambient dimension.
    method : {None, "Goodlattice", "Sobol"} or ndarray of shape (N, D)
        None and "Sobol": scrambled Sobol sequence. "Goodlattice": optimised Korobov
        lattice with a random shift.
        An array is returned as a copy.
    seed : int, optional
        Seed of the stochastic generators (Sobol, Goodlattice).
    """
    if isinstance(method, np.ndarray):
        if method.shape != (N, D):
            raise ValueError(f"warmstart array must have shape {(N, D)}, got {method.shape}")
        return method.copy()
    if method in (None, "Sobol"):
        return sobol(N, D, seed=seed)
    if method == "Goodlattice":
        return korobov_lattice(N, D, seed=seed)
    raise ValueError(f"unsupported warmstart={method!r}; expected None, 'Goodlattice', "
                     "'Sobol', or an ndarray of shape (N, D).")
