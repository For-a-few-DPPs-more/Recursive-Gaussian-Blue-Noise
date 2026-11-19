"""
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
