"""
Selection of Fourier wave vectors k in Z^D (one representative per +-k pair).

lowest_modes            the M lowest modes, with M = round(chi * D * (N - 1)) exactly
isotropic_wave_vectors  exhaustive low modes + isotropically sampled high modes
"""
from __future__ import annotations

from math import gamma, pi
from typing import NamedTuple

import numpy as np

from .integer import half_ball, min_r2_with_pairs


class WaveVectors(NamedTuple):
    k: np.ndarray        # (M, D) int32, one representative per +-k pair
    r2: np.ndarray       # (M,)   int64, |k|^2
    w: np.ndarray        # (M,)   float64 loss weights in (0, 1], max = 1

    @property
    def M(self) -> int:
        return len(self.k)


def n_modes(N: int, D: int, chi: float) -> int:
    """Number of constrained modes: chi times the D (N - 1) degrees of freedom."""
    return int(round(chi * D * (N - 1)))


def mode_weights(r2) -> np.ndarray:
    """Loss weights 1 / (|k|^2 + 1e-3), normalised so that max = 1."""
    w = 1.0 / (np.asarray(r2, dtype=np.float64) + 1e-3)
    return w / w.max()


def lowest_modes(N: int, D: int, chi: float, rng=None) -> WaveVectors:
    """The M = round(chi * D * (N - 1)) lowest modes; ties in the last shell are broken at random."""
    rng = np.random.default_rng() if rng is None else rng
    M = n_modes(N, D, chi)
    if M < 1:
        raise ValueError(f"chi={chi} gives no mode for N={N}, D={D}")

    # volume-based guess of the radius, only used to size the shell counting
    ball_fraction = (pi ** (D / 2) / gamma(D / 2 + 1)) / 2 ** D
    k_frac = (2 * chi * D / ball_fraction) ** (1.0 / D)
    r2_guess = 1.5 * (N ** (1.0 / D) * k_frac / 2) ** 2 + 10

    k = half_ball(min_r2_with_pairs(D, M, r2_guess), D)
    r2 = np.sum(k.astype(np.int64) ** 2, axis=1)
    order = np.lexsort((rng.random(len(r2)), r2))[:M]
    k, r2 = k[order], r2[order]
    return WaveVectors(k=k, r2=r2, w=mode_weights(r2))


def isotropic_wave_vectors(k_low: int, k_max: float, D: int, n_high: int) -> np.ndarray:
    """All modes with |k| <= k_low, plus n_high modes drawn isotropically in k_low < |k| < k_max
    (rounded to Z^D, deduplicated, origin removed)."""
    low = half_ball(k_low ** 2, D)
    dirs = np.random.normal(size=(n_high, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = np.random.uniform(k_low, k_max, size=(n_high, 1))
    high = np.rint(radii * dirs).astype(int)
    k = np.concatenate([low, high], axis=0)
    return np.unique(k[np.any(k != 0, axis=1)], axis=0)
