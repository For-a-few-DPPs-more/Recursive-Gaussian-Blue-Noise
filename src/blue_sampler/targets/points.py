"""Point sets as densities."""
from __future__ import annotations

import numpy as np

from .. import config

CFG = config.TARGETS


def check_points(points) -> np.ndarray:
    """A (K, D) point set, finite and not reduced to a single point (any bounding box)."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
        raise ValueError(f"target points must have shape (K, D) with K >= 2, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("target points contain NaN or inf")
    if np.ptp(points, axis=0).max() == 0:
        raise ValueError("target points are all identical")
    return points


def points_to_density(points: np.ndarray, shape: tuple[int, int], sigma: float | None = None) -> np.ndarray:
    """
    Pseudo image of a 2D point set: histogram on a `shape` grid indexed [x, y], periodic
    gaussian blur, normalised to sum 1.

    sigma: blur in pixels; default CFG.points_blur_spacing mean spacings of the points
    (at least CFG.points_blur_min pixels), smooth enough to hide the points themselves
    while keeping the variations of their density.
    """
    from .image import blur
    if points.shape[1] != 2:
        raise ValueError(f"a pseudo image needs 2D points, got D={points.shape[1]}")
    G = shape[0]
    if sigma is None:
        spacing = G / np.sqrt(len(points))
        sigma = max(CFG.points_blur_min, CFG.points_blur_spacing * spacing)
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=shape, range=[[0, 1], [0, 1]])
    rho = blur(hist, sigma).clip(min=0)
    return rho / rho.sum()
