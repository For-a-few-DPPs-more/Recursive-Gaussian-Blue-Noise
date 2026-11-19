"""
Conway's pinwheel tiling: each 1-2-sqrt(5) right triangle is split into 5 copies,
rotated by arctan(1/2). The orientations are dense on the circle (isotropic tiling).
"""
from __future__ import annotations

import numpy as np

BASE_TRIANGLE = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])

# affine maps (M, t) sending the base triangle onto its 5 children
_CHILDREN = [
    (np.array([[-1, -2], [-2, 1]]) / 5, np.array([2 / 5, 4 / 5])),
    (np.array([[2, -1], [-1, -2]]) / 5, np.array([1 / 5, 2 / 5])),
    (np.array([[2, 1], [-1, 2]]) / 5, np.array([1 / 5, 2 / 5])),
    (np.array([[2, -1], [-1, -2]]) / 5, np.array([6 / 5, 2 / 5])),
    (np.array([[-2, -1], [1, -2]]) / 5, np.array([6 / 5, 2 / 5])),
]


def subdivide(triangles: np.ndarray) -> np.ndarray:
    """(T, 3, 2) -> (5T, 3, 2)."""
    return np.concatenate([np.einsum("ij,nkj->nki", M, triangles) + t for M, t in _CHILDREN], axis=0)


def unit_square_tiling(depth: int) -> np.ndarray:
    """Pinwheel tiling of [0, 1]^2 by 4 * 5^depth triangles (4 base triangles, subdivided)."""
    tiles = BASE_TRIANGLE[None]
    for _ in range(depth):
        tiles = subdivide(tiles)
    tiles = np.concatenate([tiles, -tiles + np.array([2.0, 1.0])], axis=0)            # [0, 2] x [0, 1]
    tiles = np.concatenate([tiles, tiles * np.array([-1.0, 1.0]) + np.array([2.0, 1.0])], axis=0)  # [0, 2]^2
    return tiles / 2.0


def affine_maps(source: np.ndarray, targets: np.ndarray):
    """Affine maps x -> M x + t sending the triangle `source` (3, 2) onto each of `targets` (T, 3, 2)."""
    Q0, Q1, Q2 = source
    P0, P1, P2 = targets[:, 0], targets[:, 1], targets[:, 2]
    A = np.column_stack([Q1 - Q0, Q2 - Q0])
    B = np.stack([P1 - P0, P2 - P0], axis=2)
    M = B @ np.linalg.inv(A)
    return M, P0 - (M @ Q0[:, None])[:, :, 0]
