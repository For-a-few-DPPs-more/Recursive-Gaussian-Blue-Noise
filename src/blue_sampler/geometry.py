"""Small geometric helpers: polygons, simplices, random rotations."""
from __future__ import annotations

import numpy as np


def cross2d(a, b):
    """z-component of the cross product of 2D vectors."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def polygon_area(vertices: np.ndarray) -> np.ndarray:
    """Signed area (shoelace) of polygons (..., M, 2), vertices in boundary order."""
    return 0.5 * np.sum(cross2d(vertices, np.roll(vertices, -1, axis=-2)), axis=-1)


def simplex(D: int) -> np.ndarray:
    """Vertices (D+1, D) of a regular simplex centred at the origin, unit circumradius."""
    if D == 1:
        return np.array([[-1.0], [1.0]])
    tip = np.zeros((1, D))
    tip[0, -1] = 1.0
    base = np.hstack((simplex(D - 1), np.zeros((D, 1))))
    return np.vstack((np.sqrt(1.0 - (1.0 / D) ** 2) * base - tip / D, tip))


def random_rotations(x: np.ndarray, n: int, d_out: int) -> np.ndarray:
    """n random orthonormal embeddings of the point set x (M, d_in) into R^d_out: (n, M, d_out)."""
    Q, _ = np.linalg.qr(np.random.randn(n, d_out, x.shape[-1]))
    return np.einsum("nij,kj->nki", Q, x)
