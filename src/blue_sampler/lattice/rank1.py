"""
Korobov rank-1 lattices ("good lattices") on the unit torus.

The generating vector is z = (1, a, a^2, ..., a^(D-1)) mod N and the points are
x_t = t z / N mod 1. The integer a maximises the spectral test: the length lambda_1
of the shortest non-zero vector of the dual lattice {h : h.z = 0 mod N}, estimated
by an LLL reduction in O(D^3) per candidate (instead of an exhaustive search).
For D <= 6 the LLL approximation factor is <= 2.37, so the first reduced vector is
virtually always the true shortest vector.

References: Sloan & Joe (1994), Lattice Methods for Multiple Integration;
Niederreiter (1992), Random Number Generation and Quasi-Monte Carlo Methods.
"""
from __future__ import annotations

from math import gcd

import numpy as np


def korobov_generator(a: int, N: int, D: int) -> np.ndarray:
    """z = (1, a, a^2, ..., a^(D-1)) mod N."""
    z = np.empty(D, dtype=np.int64)
    z[0] = 1
    for j in range(1, D):
        z[j] = (z[j - 1] * a) % N
    return z


def _dual_basis(z: np.ndarray, N: int) -> np.ndarray:
    """Rows: a basis of the dual lattice {h : h.z = 0 mod N} (determinant N)."""
    D = len(z)
    B = np.eye(D, dtype=np.int64)
    B[0, 0] = N
    B[1:, 0] = (-z[1:]) % N
    return B


def _gram_schmidt(B: np.ndarray):
    D = len(B)
    B_star = np.zeros_like(B)
    mu = np.zeros((D, D))
    for i in range(D):
        B_star[i] = B[i]
        for j in range(i):
            mu[i, j] = B[i] @ B_star[j] / (B_star[j] @ B_star[j])
            B_star[i] -= mu[i, j] * B_star[j]
    return B_star, mu


def lll_reduce(B: np.ndarray, delta: float = 0.99) -> np.ndarray:
    """LLL reduction of the lattice spanned by the rows of B."""
    B = B.astype(np.float64).copy()
    k = 1
    while k < len(B):
        _, mu = _gram_schmidt(B)
        for j in range(k - 1, -1, -1):          # size reduction
            q = round(mu[k, j])
            if q != 0:
                B[k] -= q * B[j]
        B_star, mu = _gram_schmidt(B)
        if B_star[k] @ B_star[k] >= (delta - mu[k, k - 1] ** 2) * (B_star[k - 1] @ B_star[k - 1]):
            k += 1                               # Lovasz condition holds
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            k = max(1, k - 1)
    return np.round(B).astype(np.int64)


def spectral_test(a: int, N: int, D: int) -> float:
    """Estimate of lambda_1 (shortest dual vector) for the generator a."""
    reduced = lll_reduce(_dual_basis(korobov_generator(a, N, D), N))
    return float(np.linalg.norm(reduced[0].astype(np.float64)))


def best_korobov_generator(N: int, D: int, max_candidates: int | None = 100, rng=None) -> np.ndarray:
    """Generating vector z maximising the spectral test among (at most max_candidates) coprime a."""
    rng = np.random.default_rng() if rng is None else rng
    candidates = [a for a in range(2, N) if gcd(a, N) == 1]
    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = sorted(rng.choice(candidates, size=max_candidates, replace=False))
    best_a = max(candidates, key=lambda a: spectral_test(a, N, D))
    return korobov_generator(int(best_a), N, D)


def korobov_lattice(N: int, D: int, shift: bool = True, seed: int | None = None,
                    max_candidates: int | None = 100) -> np.ndarray:
    """N points of an optimised Korobov lattice in [0, 1)^D, with an optional
    random toroidal shift (randomised QMC). Fully reproducible for a given seed."""
    rng = np.random.default_rng(seed)
    z = best_korobov_generator(N, D, max_candidates, rng)
    points = (np.arange(N, dtype=np.float64)[:, None] * (z.astype(np.float64) / N)) % 1.0
    if shift:
        points = (points + rng.random(D)) % 1.0
    return points
