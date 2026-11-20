"""
Conway's pinwheel tiling: each 1-2-sqrt(5) right triangle is split into 5 copies,
rotated by arctan(1/2). The orientations are dense on the circle (isotropic tiling).

pinwheel_map     affine maps of the tiling applied to points of the base triangle
pinwheel_points  N 2D points: 3 moment-matched points per triangle, cropped to [0, 1)^2
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..moments import fit_points

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


def pinwheel_map(points, depth: int):
    """Apply the affine maps of the pinwheel tiling to (2,), (M, 2) or (T, M, 2) points."""
    M, t = affine_maps(BASE_TRIANGLE, unit_square_tiling(depth))
    if points.ndim == 1:
        return np.einsum("nij,j->ni", M, points) + t
    if points.ndim == 2:
        return np.einsum("nij,kj->nki", M, points) + t[:, None, :]
    return np.einsum("nij,nkj->nki", M, points) + t[:, None, :]


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
    base_points = fit_points(BASE_TRIANGLE, "polygons", p=3, verbose=0)   # (3, 2)
    depth = int(np.log(N / 3) / np.log(5) + 1)
    M, t = affine_maps(BASE_TRIANGLE, unit_square_tiling(depth))
    x = np.einsum("nij,kj->nki", M, base_points) + t[:, None, :]
    return crop_to_unit_cube(x, N, intensity=len(M) * len(base_points))
