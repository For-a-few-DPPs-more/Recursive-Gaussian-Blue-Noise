"""
qmc_geom.py
-----------
Geometric utilities for selecting Fourier modes on the unit torus.

Main interface
--------------
get_wave_vectors(N, D, chi, ...)

For given parameters N, D, and chi, this function returns:

    - The selected wave vectors k in the exclusion region.
    - The corresponding Fourier weights, used to compute the
      weighted loss

          sum_k weight(k) * |X_hat(k)|**2

    - The number of selected wave vectors, for a sanity check.
      The expected count is

          M = chi * D * (N - 1)

Exclusion geometries
--------------------
The exclusion region can have different shapes, specified by
the geometry parameter:

    - "l2": Euclidean (L2) geometry.
    - "l1": Manhattan (L1) geometry.
    - "wd": Weighted discrepancy (WD) geometry.

WD Fourier weights
------------------
For "wd", the Fourier weights are derived from the one-dimensional
kernel used by scipy.stats.qmc.discrepancy(method="WD"):

    K(t) = 3/2 - |t| + |t|**2,  for t in [0, 1],

with periodic extension.

Its Fourier coefficients are:

    - 4/3 at frequency m = 0.
    - 1 / (2 * pi**2 * m**2) for nonzero integer frequencies m.

Fourier half space symetry convention
-----------------------
Only one representative of each pair {k, -k} is retained:
the first nonzero coordinate of k must be positive.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, count, product
from math import comb, factorial, log, pi
from typing import Iterator, List, Optional, Tuple
import heapq

import numpy as np


CHI = 0.43
EPSILON = 1e-12
INV_TWO_PI_SQUARED = 1.0 / (2.0 * pi * pi)
ZERO_FREQUENCY_WEIGHT = 4.0 / 3.0


def _n_modes_target(N: int, D: int, chi: float = CHI) -> int:
    return max(0, int(round(chi * D * (N - 1))))


def _hatK(m: int) -> float:
    """One-dimensional Fourier coefficient of the WD kernel."""
    m = abs(int(m))
    if m == 0:
        return ZERO_FREQUENCY_WEIGHT
    return INV_TWO_PI_SQUARED / (m * m)


def wd_weight(K: np.ndarray) -> np.ndarray:
    """Return product WD Fourier weights for an array of integer vectors."""
    abs_K = np.abs(np.asarray(K, dtype=np.float64))
    factors = np.full(abs_K.shape, ZERO_FREQUENCY_WEIGHT, dtype=np.float64)
    nonzero = abs_K != 0
    factors[nonzero] = INV_TWO_PI_SQUARED / abs_K[nonzero] ** 2
    return factors.prod(axis=1)


def _orbit_size(k: Tuple[int, ...], D: int) -> int:
    """Count half-space modes in the signed-permutation orbit of k."""
    s = len(k)
    if s == 0 or s > D:
        return 0

    permutations_count = factorial(s)
    for multiplicity in Counter(k).values():
        permutations_count //= factorial(multiplicity)

    return comb(D, s) * permutations_count * (1 << (s - 1))


def _orbit_weight(k: Tuple[int, ...], D: int) -> float:
    """Return the common WD Fourier weight of an orbit."""
    s = len(k)
    weight = ZERO_FREQUENCY_WEIGHT ** (D - s)
    for frequency in k:
        weight *= _hatK(frequency)
    return weight


def _log_orbit_weight(k: Tuple[int, ...], D: int) -> float:
    """Log weight used for stable priority-queue ordering."""
    s = len(k)
    return (
        (D - s) * log(ZERO_FREQUENCY_WEIGHT)
        + s * log(INV_TWO_PI_SQUARED)
        - 2.0 * sum(log(frequency) for frequency in k)
    )


def _distinct_permutations(values: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    """Yield distinct permutations without materializing duplicates."""
    counts = Counter(values)
    choices = sorted(counts, reverse=True)
    length = len(values)
    current: List[int] = []

    def visit() -> Iterator[Tuple[int, ...]]:
        if len(current) == length:
            yield tuple(current)
            return
        for value in choices:
            if counts[value] == 0:
                continue
            counts[value] -= 1
            current.append(value)
            yield from visit()
            current.pop()
            counts[value] += 1

    yield from visit()


def _expand_half_orbit(
    k: Tuple[int, ...],
    D: int,
    limit: Optional[int] = None,
) -> Iterator[Tuple[int, ...]]:
    """Yield distinct orbit vectors whose first nonzero entry is positive."""
    s = len(k)
    if s == 0 or s > D or limit == 0:
        return

    emitted = 0
    for positions in combinations(range(D), s):
        for values in _distinct_permutations(k):
            for tail_signs in product((-1, 1), repeat=s - 1):
                signs = (1,) + tail_signs
                vector = [0] * D
                for position, value, sign in zip(positions, values, signs):
                    vector[position] = value * sign

                yield tuple(vector)
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def half_wd_ball(
    D: int,
    M: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the M highest-weight nonzero WD modes in the half-space.

    Orbit representatives are nonincreasing positive integer tuples. A
    best-first heap explores the infinite set of representatives in exact
    nonincreasing weight order. Unlike a fixed frequency cutoff, this gives
    an exact top-M selection without omitting potentially better modes.

    """
    if D < 1:
        raise ValueError("D must be at least 1")
    if M <= 0:
        return np.empty((0, D), dtype=np.int32), np.empty(0, dtype=np.float64)

    # Heap entries are (-log_weight, tie_breaker, canonical_tuple).
    # Canonical tuples are nonincreasing and contain only positive values.
    queue = []
    serial = count()
    seen = set()

    for support in range(1, D + 1):
        representative = (1,) * support
        seen.add(representative)
        heapq.heappush(
            queue,
            (-_log_orbit_weight(representative, D), next(serial), representative),
        )

    vectors: List[Tuple[int, ...]] = []
    weights: List[float] = []

    while len(vectors) < M:
        if not queue:
            raise RuntimeError("Orbit search unexpectedly exhausted.")

        _, _, representative = heapq.heappop(queue)
        orbit_size = _orbit_size(representative, D)
        remaining = M - len(vectors)
        take = min(orbit_size, remaining)
        orbit_weight = _orbit_weight(representative, D)

        vectors.extend(_expand_half_orbit(representative, D, limit=take))
        weights.extend([orbit_weight] * take)

        # If the requested cutoff falls inside this orbit, all its modes have
        # the same weight and every unvisited orbit has no larger weight.
        if take < orbit_size:
            break

        # Increasing any frequency creates a lower-weight representative.
        # These monotone neighbours enumerate every canonical tuple.
        for index, frequency in enumerate(representative):
            if index > 0 and frequency == representative[index - 1]:
                continue

            child = list(representative)
            child[index] += 1
            child_tuple = tuple(child)
            if child_tuple not in seen:
                seen.add(child_tuple)
                heapq.heappush(
                    queue,
                    (
                        -_log_orbit_weight(child_tuple, D),
                        next(serial),
                        child_tuple,
                    ),
                )

    K = np.asarray(vectors, dtype=np.int32).reshape(M, D)
    r = np.asarray(weights, dtype=np.float64)
    return K, r


# ------------------------------------------------------------------
# L1 helpers (unchanged)
# ------------------------------------------------------------------
def _count_l1_ball(D, radius):
    return sum((2 ** j) * comb(D, j) * comb(radius, j)
               for j in range(min(D, radius) + 1))

def _full_l1_ball(D, radius):
    if D == 1:
        return np.arange(-radius, radius + 1, dtype=np.int32)[:, None]
    prev = _full_l1_ball(D - 1, radius)
    prev_norm = np.abs(prev.astype(np.int64)).sum(1)
    remaining = radius - prev_norm
    counts = 2 * remaining + 1
    total = int(counts.sum())
    wdts = np.cumsum(counts, dtype=np.int64) - counts
    row_ids = np.repeat(np.arange(len(prev), dtype=np.int64), counts)
    z = np.arange(total, dtype=np.int64) - np.repeat(wdts, counts)
    z -= np.repeat(remaining, counts)
    return np.concatenate([prev[row_ids], z.astype(np.int32)[:, None]], 1)

def _half_l1_ball(D, radius):
    if radius <= 0:
        return np.empty((0, D), dtype=np.int32)
    if D == 1:
        return np.arange(1, radius + 1, dtype=np.int32)[:, None]
    prev = _full_l1_ball(D - 1, radius)
    prev_norm = np.abs(prev.astype(np.int64)).sum(1)
    max_z = radius - prev_norm
    counts = max_z
    total = int(counts.sum())
    if total:
        wdts = np.cumsum(counts, dtype=np.int64) - counts
        row_ids = np.repeat(np.arange(len(prev), dtype=np.int64), counts)
        z = np.arange(total, dtype=np.int64) - np.repeat(wdts, counts) + 1
        positive_part = np.concatenate(
            [prev[row_ids], z.astype(np.int32)[:, None]], 1)
    else:
        positive_part = np.empty((0, D), dtype=np.int32)
    previous_half = _half_l1_ball(D - 1, radius)
    zero_part = np.concatenate(
        [previous_half, np.zeros((len(previous_half), 1), dtype=np.int32)], 1)
    return np.concatenate([positive_part, zero_part], 0)

def delete_last_incomplete_orbit(K, W, epsilon=1e-10):
    order = np.argsort(-W)
    W_sorted = W[order]
    n = np.sum(np.abs(W_sorted - W_sorted[-1]) < epsilon)
    return K[order[:-n]], len(K) - n

def get_wave_vectors(
    N: int,
    D: int,
    mode: str = "wd",
    chi: float = CHI,
    rng: Optional[np.random.Generator] = None,
    epsilon: float = EPSILON,
):
    if mode == "l2":
        from blue_sampler.run.run_nufft_helpers import get_wave_vectors as gwv_l2
        WV = gwv_l2(N, D, chi, rng)
        K, M = delete_last_incomplete_orbit(WV.k, WV.w)
        return K, M

    rng = np.random.default_rng() if rng is None else rng
    M_target = _n_modes_target(N, D, chi)
    if M_target == 0:
        empty = np.empty((0, D), dtype=np.int32)
        return empty, np.empty(0, dtype=np.int64), np.empty(0), 0
    if mode == "wd":
        K, r = half_wd_ball(D, M_target)
        weights = r / (r.max() + epsilon)
    elif mode == "l1":
        radius = 1
        while (_count_l1_ball(D, radius) - 1) // 2 < M_target:
            radius *= 2
        low, high = max(1, radius // 2), radius
        while low < high:
            mid = (low + high) // 2
            if (_count_l1_ball(D, mid) - 1) // 2 >= M_target:
                high = mid
            else:
                low = mid + 1
        radius = low
        K = _half_l1_ball(D, radius)
        score = np.abs(K).sum(1)
        order = np.lexsort((rng.random(len(score)), score))[:M_target]
        K = K[order]
        raw = 1.0 / (score[order].astype(np.float64) ** 2 + epsilon)
        weights = raw / raw.max()
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'wd' or 'l1'.")
    K = np.ascontiguousarray(K.astype(np.int32))
    K, M = delete_last_incomplete_orbit(K, weights)
    return K, M