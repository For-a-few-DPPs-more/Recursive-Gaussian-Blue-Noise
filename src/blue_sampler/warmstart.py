"""
Initial point clouds in [0, 1)^D (warm starts) and cheap low-discrepancy samplers.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import pinwheel
from .lattice import korobov_lattice

WarmstartMethod = Literal["Goodlattice", "Sobol", "Pinwheel"]


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


def crop_to_unit_cube(x: NDArray, N: int, intensity: float | None = None) -> NDArray:
    """
    Extract N points from a large point cloud x (..., D), around its centre after a random
    rotation, and map them onto the unit torus [0, 1)^D.

    intensity: number of points of x per unit volume (estimated from the bounding cube if None).
    Assumes x densely covers a cube around its centre larger than the N points needed.
    """
    D = x.shape[-1]
    rotation, _ = np.linalg.qr(np.random.default_rng().normal(size=(D, D)))
    x = x.reshape(-1, D) @ rotation
    x = x - x.mean(axis=0, keepdims=True)
    order = np.argsort(np.max(np.abs(x), axis=1))       # sup-norm: nested cubes
    x = x[order]
    if intensity is None:
        intensity = len(x) / (2 * np.max(np.abs(x))) ** D
    side = (N / intensity) ** (1 / D)
    x = x[:N] / side + 0.5
    return x - np.floor(x)


def pinwheel_points(N: int) -> NDArray:
    """N 2D points: 3 moment-matched points per triangle of a pinwheel tiling, cropped."""
    from .moments import fit_points
    base_points = fit_points(pinwheel.BASE_TRIANGLE, "polygons", p=3, verbose=0)   # (3, 2)
    depth = int(np.log(N / 3) / np.log(5) + 1)
    M, t = pinwheel.affine_maps(pinwheel.BASE_TRIANGLE, pinwheel.unit_square_tiling(depth))
    x = np.einsum("nij,kj->nki", M, base_points) + t[:, None, :]
    return crop_to_unit_cube(x, N, intensity=len(M) * len(base_points))


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
    method : {None, "Goodlattice", "Sobol", "Pinwheel"} or ndarray of shape (N, D)
        None and "Sobol": scrambled Sobol sequence. "Goodlattice": optimised Korobov
        lattice with a random shift. "Pinwheel": pinwheel tiling (D=2 only, Sobol otherwise).
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
    if method == "Pinwheel":
        if D == 2:
            return pinwheel_points(N)
        warnings.warn(f"warmstart='Pinwheel' is only supported for D=2 (got D={D}); "
                      "falling back to Sobol.", UserWarning, stacklevel=2)
        return sobol(N, D, seed=seed)
    raise ValueError(f"unsupported warmstart={method!r}; expected None, 'Goodlattice', "
                     "'Sobol', 'Pinwheel', or an ndarray of shape (N, D).")
