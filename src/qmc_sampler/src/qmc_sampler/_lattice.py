"""
Adapted from blue_sampler/lattice/integer.py and lattice/modes.py (copied so that
qmc_sampler does not depend on blue_sampler).

Integer points of Z^D: grids, balls and half-balls.

Every ball is described by an *integer* squared radius ``r2max``:
``ball(r2max, D) = {k in Z^D : |k|^2 <= r2max}``.

A half-ball keeps one representative per pair {k, -k} (origin excluded).
Convention: the first non-zero coordinate of k is positive, and the vectors
are returned in lexicographic order.
Balls are built shell by shell (cylinders over the (D-1)-ball), so the
memory never exceeds the size of the output, even for millions of points.
"""
from __future__ import annotations

from math import gamma, pi

import numpy as np


def integer_grid(axis, D: int) -> np.ndarray:
    """Cartesian product axis^D as an (len(axis)^D, D) array, 'ij' ordering."""
    grids = np.meshgrid(*([np.asarray(axis)] * D), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def cube(R: int, D: int) -> np.ndarray:
    """All integer vectors of [-R, R]^D, (2R+1)^D x D, int32."""
    return integer_grid(np.arange(-R, R + 1, dtype=np.int32), D)


def isqrt(x):
    """Exact floor(sqrt(x)) for non-negative integers (scalar or array)."""
    x = np.asarray(x, dtype=np.int64)
    z = np.floor(np.sqrt(x.astype(np.float64))).astype(np.int64)
    z -= z * z > x
    z += (z + 1) * (z + 1) <= x
    return z


def _cylinders(counts, lo):
    """Concatenation of the ranges lo[i], ..., lo[i] + counts[i] - 1 (int32)."""
    starts = np.cumsum(counts) - counts
    z = np.arange(int(counts.sum()), dtype=np.int32)
    z -= np.repeat(starts, counts).astype(np.int32)
    z += np.repeat(np.asarray(lo, dtype=np.int32), counts)
    return z


def ball(r2max: int, D: int) -> np.ndarray:
    """All k in Z^D with |k|^2 <= r2max, int32, lexicographic order."""
    if D == 1:
        K = int(isqrt(r2max))
        return np.arange(-K, K + 1, dtype=np.int32)[:, None]
    head = ball(r2max, D - 1)
    m = isqrt(r2max - np.sum(head.astype(np.int64) ** 2, axis=1))
    counts = 2 * m + 1
    return np.concatenate([np.repeat(head, counts, axis=0), _cylinders(counts, -m)[:, None]], axis=1)


def _half_ball_unsorted(r2max: int, D: int) -> np.ndarray:
    """Half-ball with the *last* non-zero coordinate positive (cheap to build)."""
    if D == 1:
        return np.arange(1, int(isqrt(r2max)) + 1, dtype=np.int32)[:, None]
    head = ball(r2max, D - 1)                       # last coordinate z > 0
    m = isqrt(r2max - np.sum(head.astype(np.int64) ** 2, axis=1))
    upper = np.concatenate([np.repeat(head, m, axis=0), _cylinders(m, np.ones_like(m))[:, None]], axis=1)
    flat = _half_ball_unsorted(r2max, D - 1)        # last coordinate z = 0
    flat = np.concatenate([flat, np.zeros((len(flat), 1), dtype=np.int32)], axis=1)
    return np.concatenate([upper, flat], axis=0)


def half_ball(r2max: int, D: int) -> np.ndarray:
    """One vector per pair {k, -k} with 0 < |k|^2 <= r2max (first non-zero coordinate > 0),
    int32, lexicographic order."""
    k = _half_ball_unsorted(int(r2max), D)
    first = k[np.arange(len(k)), np.argmax(k != 0, axis=1)]
    k = k * np.sign(first)[:, None].astype(np.int32)
    return k[np.lexsort(k.T[::-1])]


def min_r2_with_pairs(D: int, n_pairs: int, r2_guess: float = 4) -> int:
    """Smallest integer r2 such that the half-ball of squared radius r2 holds >= n_pairs vectors.

    Uses the theta series r_D(n) = #{k : |k|^2 = n} (D-fold convolution of the 1D one):
    nothing is built.
    """
    n_max = max(int(r2_guess), 4)
    while True:
        r = np.zeros(n_max + 1, dtype=np.int64)
        r[0] = 1
        squares = np.arange(isqrt(n_max) + 1) ** 2
        for _ in range(D):
            new = np.zeros_like(r)
            for s, s2 in enumerate(squares):
                new[s2:] += (1 if s == 0 else 2) * r[: n_max + 1 - s2]
            r = new
        pairs = (np.cumsum(r) - 1) // 2               # origin excluded, one per +-k pair
        r2 = int(np.searchsorted(pairs, n_pairs, side="left"))
        if r2 <= n_max:
            return r2
        n_max *= 2


def n_modes(N: int, D: int, chi: float) -> int:
    """Number of constrained modes: chi times the D (N - 1) degrees of freedom."""
    return int(round(chi * D * (N - 1)))


def mode_weights(r2) -> np.ndarray:
    """Loss weights 1 / (|k|^2 + 1e-3), normalised so that max = 1."""
    w = 1.0 / (np.asarray(r2, dtype=np.float64) + 1e-3)
    return w / w.max()


def lowest_modes(N: int, D: int, chi: float, rng=None) :
    """(k, w): the M = round(chi * D * (N - 1)) lowest modes and their weights; ties in the last
    shell are broken at random."""
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
    return k, mode_weights(r2)


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
