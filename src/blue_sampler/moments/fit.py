"""Entry point of moment matching: cells (clusters or polygons) -> representative points."""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from .. import config
from .solver import solve_moments_lm
from .targets import cluster_moments, moment_orders, polygon_moments

CFG = config.MOMENTS
CellKind = Literal["clusters", "polygons"]


def required_n(p: int, D: int) -> int:
    """Smallest n with at least as many free coordinates (n D) as constraints
    (D for the centroid + one per central-moment order). Necessary, not sufficient."""
    return -(-(D + len(moment_orders(p, D))) // D)


def _cell_moments(cells, kind: CellKind, orders):
    if kind == "clusters":
        return cluster_moments(cells, np.full(cells.shape[:2], 1 / cells.shape[1]), orders)
    return polygon_moments(cells, orders)


def _initial_points(cells, kind: CellKind, n: int, rng):
    """n random atoms of each cluster, or n uniform points in each polygon's bounding box."""
    if kind == "clusters":
        idx = np.argsort(rng.random(cells.shape[:2]), axis=1)[:, :n]
        return np.take_along_axis(cells, idx[..., None], axis=1)
    lo, hi = cells.min(1), cells.max(1)
    return rng.uniform(lo[:, None], hi[:, None], (len(cells), n, 2))


def fit_points(
    cells: np.ndarray,
    kind: CellKind,
    p: int = 3,
    n_iter: int = 20,
    n_points: int | None = None,
    lambda0: float = CFG.lambda0,
    verbose: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Replace each cell by n points matching its centroid and central moments up to order p - 1.

    cells: (B, K, D) clusters of K atoms, or (B, M, 2) polygons (a single cell may omit B).
    n_points: points per cell, default `required_n(p, D)`.
    Returns (B, n, D) (or (n, D) for a single cell).
    """
    if kind not in ("clusters", "polygons"):
        raise ValueError(f"unknown cell kind {kind!r}, expected 'clusters' or 'polygons'")
    if not 2 <= p <= CFG.max_p:
        raise ValueError(f"p must satisfy 2 <= p <= {CFG.max_p}, got {p}")
    single = cells.ndim == 2
    cells = cells[None] if single else cells
    D = cells.shape[-1]
    if kind == "polygons" and D != 2:
        raise ValueError("polygons must be 2D")

    # work in normalised coordinates (unit RMS radius around the mean vertex / atom)
    center = cells.mean(axis=1, keepdims=True)
    radius = np.sqrt(((cells - center) ** 2).sum(axis=2).mean(axis=1))[:, None, None]
    cells = (cells - center) / radius

    orders = moment_orders(p, D)
    n = required_n(p, D) if n_points is None else n_points
    if kind == "clusters" and n >= cells.shape[1]:
        K = cells.shape[1]
        if verbose >= 1:
            warnings.warn(f"Few atoms per cluster (K={K}) for m={n} points to fit: m reduced to "
                          f"{max(1, K - 1)}. Increase the cluster size (ideally K >= 2m) to avoid overfitting.")
        n = max(1, K - 1)

    centroid, moments = _cell_moments(cells, kind, orders)
    if p == 2 and n == 1:                     # centroid only: no system to solve
        points = centroid[:, None, :]
    else:
        init = _initial_points(cells, kind, n, np.random.default_rng(seed))
        points, _ = solve_moments_lm(centroid, moments, orders, init, n_iter=n_iter,
                                     lambda0=lambda0, verbose=verbose)

    points = points * radius + center
    return points[0] if single else points
