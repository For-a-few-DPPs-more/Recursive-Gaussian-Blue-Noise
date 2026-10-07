"""
Exact-Chi wave-vector selection shared by the three NUFFT-like pipelines
(finufft D <= 3, KeOps fallback, JAX fallback).

The number of modes is FIXED EXACTLY: M = round(Chi * D * N) pairs ±k
(a single representative per pair, since fk(-k) = conj(fk(k)) gives the same
loss up to a factor 2). The radius (as an integer |k|^2) is the minimal excess
radius containing at least M pairs, computed by counting (theta series),
without building anything. Only that minimal ball is then constructed.
"""

from math import gamma, pi
from typing import NamedTuple

import numpy as np


class WaveVectors(NamedTuple):
    k: np.ndarray        # (M, D) int32, one representative per ±k pair
    r2: np.ndarray       # (M,)   int64, |k|^2
    w: np.ndarray        # (M,)   float64, weights in (0, 1], max = 1
    M: int               # number of independent modes
    chi_eff: float       # M / (D * n_eff), equals Chi up to 1/(2 D n_eff)


# ----------------------------------------------------------------------------
# Integer lattice: exact counting and half-space construction
# ----------------------------------------------------------------------------

def _isqrt(x):
    """Exact floor(sqrt(x)) for integers (scalar or array)."""
    x = np.asarray(x, dtype=np.int64)
    z = np.floor(np.sqrt(x.astype(np.float64))).astype(np.int64)
    z -= (z * z > x)
    z += ((z + 1) * (z + 1) <= x)
    return z


def _min_r2_for_pairs(D, M_target, n_guess):
    """Smallest integer n* such that #{k in Z^D, 0 < |k|^2 <= n*} / 2 >= M_target.

    r_D(n) = #{k : |k|^2 = n} is obtained by convolving the 1D theta series
    D times. No point is ever constructed.
    """
    n_max = max(int(n_guess), 4)
    while True:
        r = np.zeros(n_max + 1, dtype=np.int64)
        r[0] = 1
        for _ in range(D):
            new = np.zeros_like(r)
            s = 0
            while s * s <= n_max:
                new[s * s:] += (1 if s == 0 else 2) * r[: n_max + 1 - s * s]
                s += 1
            r = new
        pairs = (np.cumsum(r) - 1) // 2          # ±k pairs, origin excluded
        idx = int(np.searchsorted(pairs, M_target, side="left"))
        if idx <= n_max:
            return idx
        n_max *= 2                               # search window too small: double it


def _cylinder_z(counts, lo):
    """For each cylinder i, the integers lo[i], lo[i]+1, ..., lo[i]+counts[i]-1,
    concatenated (int32, no meshgrid)."""
    total = int(counts.sum())
    starts = (np.cumsum(counts) - counts).astype(np.int32)
    z = np.arange(total, dtype=np.int32)
    z -= np.repeat(starts, counts)
    z += np.repeat(np.asarray(lo, dtype=np.int32), counts)
    return z


def _full_ball_int(D, r2max):
    """All k in Z^D with |k|^2 <= r2max (exact integer bound), int32."""
    if D == 1:
        K = int(_isqrt(r2max))
        return np.arange(-K, K + 1, dtype=np.int32)[:, None]
    prev = _full_ball_int(D - 1, r2max)
    r2 = np.sum(prev.astype(np.int64) ** 2, axis=1)
    m = _isqrt(r2max - r2)
    counts = 2 * m + 1
    base = np.repeat(prev, counts, axis=0)
    z = _cylinder_z(counts, -m)
    return np.concatenate([base, z[:, None]], axis=1)


def _half_ball_int(D, r2max):
    """Half-space {last nonzero coordinate > 0}, origin excluded.
    One representative per ±k pair, without ever building the full D-dim ball."""
    if D == 1:
        K = int(_isqrt(r2max))
        return np.arange(1, K + 1, dtype=np.int32)[:, None]

    # last coordinate z > 0, the first (D-1) coordinates arbitrary
    full_prev = _full_ball_int(D - 1, r2max)
    r2 = np.sum(full_prev.astype(np.int64) ** 2, axis=1)
    m = _isqrt(r2max - r2)                          # z in 1..m
    base = np.repeat(full_prev, m, axis=0)
    z = _cylinder_z(m, np.ones_like(m))
    part1 = np.concatenate([base, z[:, None]], axis=1)
    del base, z, full_prev

    # last coordinate z = 0, the first (D-1) coordinates in the half-space
    half_prev = _half_ball_int(D - 1, r2max)
    part2 = np.concatenate(
        [half_prev, np.zeros((len(half_prev), 1), dtype=np.int32)], axis=1
    )
    return np.concatenate([part1, part2], axis=0)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def n_modes_target(N, D, Chi):
    """Exact number of independent (±k pair) modes: round(Chi * D * n_eff).
    """
    n_eff = (N - 1)
    return int(round(Chi * D * n_eff))


def exact_chi_modes(N, D, Chi, rng=None):
    """Return (K (M, D) int32, r2 (M,) int64) with M = round(Chi*D*n_eff) EXACTLY.

    One representative per ±k pair (origin excluded). The M smallest |k|^2 are
    kept; ties in the last (partially filled) shell are broken randomly.
    """
    rng = np.random.default_rng() if rng is None else rng
    M_target = n_modes_target(N, D, Chi)

    # Starting bound for the search (volume-based), ONLY used to size the
    # counting array; the final radius is the exact minimal excess radius.
    ball_ratio = (pi ** (D / 2) / gamma(D / 2 + 1)) / (2 ** D)
    kfrac = (2 * Chi * D / ball_ratio) ** (1.0 / D)
    n_guess = 1.5 * (N ** (1.0 / D) * kfrac / 2) ** 2 + 10

    r2star = _min_r2_for_pairs(D, M_target, n_guess)

    K = _half_ball_int(D, r2star)
    r2 = np.sum(K.astype(np.int64) ** 2, axis=1)
    assert len(K) >= M_target

    # sort by |k|^2, random tie-breaking within the last shell
    order = np.lexsort((rng.random(len(r2)), r2))[:M_target]
    return K[order], r2[order]


def mode_weights(r2):
    """Loss weights 1 / (|k|^2 + 1e-3), normalised so that max = 1 (float64)."""
    rpow = 1.0 / (np.asarray(r2, dtype=np.float64) + 1e-3)
    return rpow / rpow.max()


def get_wave_vectors(N, D, Chi, rng=None):
    """Select the wave vectors for (N, D, Chi) and check that Chi is exact."""
    K, r2 = exact_chi_modes(N, D, Chi, rng)
    M = int(K.shape[0])

    n_eff = (N - 1) 
    assert M == n_modes_target(N, D, Chi), (M, N, D, Chi)
    chi_eff = M / (D * n_eff)
    assert abs(chi_eff - Chi) <= 0.5 / (D * n_eff) + 1e-12, (chi_eff, Chi)

    return WaveVectors(k=K, r2=r2, w=mode_weights(r2), M=M, chi_eff=chi_eff)