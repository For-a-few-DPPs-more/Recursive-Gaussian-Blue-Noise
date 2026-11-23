"""
Moment matching: replace a cell (a cluster of atoms, or a polygon) by n equal-weight points
sharing its centroid and its central moments of order 2 to p - 1.

Notation of Lotz & Klatt, "Persistence of asymptotic variance under transport"
(arXiv:2605.22803), Eq. (1.8):

    1/lambda_d(C) int_C x^q dx = 1/n sum_j X_j^q,    |q| in {0, ..., p-1}

q = 0 holds by normalisation, q = 1 fixes the centroid, 2 <= |q| <= p-1 are the central
moment constraints, solved for all cells at once by Levenberg-Marquardt.
"""
from __future__ import annotations

import warnings
from itertools import product
from math import comb
from typing import Literal

import numpy as np

from . import config
from .progress import Progress

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
    lambda0: float | None = None,
    verbose: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Replace each cell by n points matching its centroid and central moments up to order p - 1.

    cells: (B, K, D) clusters of K atoms, or (B, M, 2) polygons (a single cell may omit B).
    p: moments of order < p are matched (p = 2: centroid only).
    n_iter: Levenberg-Marquardt iterations.
    n_points: points per cell, default `required_n(p, D)`.
    lambda0: initial Levenberg-Marquardt damping, default config.MOMENTS.lambda0.
    seed: seed of the initial points.
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
        lambda0 = CFG.lambda0 if lambda0 is None else lambda0
        points, _ = solve_moments_lm(centroid, moments, orders, init, n_iter=n_iter,
                                     lambda0=lambda0, verbose=verbose)

    points = points * radius + center
    return points[0] if single else points


# ── Moments of the cells ─────────────────────────────────────────────────────

def moment_orders(p: int, D: int) -> list[tuple[int, ...]]:
    """Multi-indices alpha in N^D with 2 <= |alpha| <= p - 1, sorted by (|alpha|, alpha)."""
    orders = [a for a in product(range(p), repeat=D) if 2 <= sum(a) <= p - 1]
    return sorted(orders, key=lambda a: (sum(a), a))


def monomial(delta: np.ndarray, alpha) -> np.ndarray:
    """prod_d delta[..., d] ** alpha[d]."""
    out = np.ones(delta.shape[:-1])
    for d, a in enumerate(alpha):
        out *= delta[..., d] ** a
    return out


def cluster_moments(points: np.ndarray, weights: np.ndarray, orders):
    """Centroid (B, D) and central moments (B, len(orders)) of weighted point clouds.
    points (B, n, D); weights (B, n) or (n,), summing to 1."""
    centroid = np.sum(points * weights[..., None], axis=1)
    delta = points - centroid[:, None]
    moments = np.stack([np.sum(weights * monomial(delta, a), axis=1) for a in orders], axis=1) \
        if orders else np.empty((len(points), 0))
    return centroid, moments


# Raw moments int_P x^i y^j dA via Green's theorem, edge by edge:
#   int_P x^i y^j = 1/(i+j+2) sum_edges cross_e int_0^1 x_e(t)^i y_e(t)^j dt,
# with cross_e = x yn - xn y and x_e(t) = x + t (xn - x) (binomial closed form below).

def _edge_integral(i, j, x, xn, y, yn):
    """int_0^1 (x + t dx)^i (y + t dy)^j dt."""
    dx, dy = xn - x, yn - y
    out = np.zeros_like(x)
    for a in range(i + 1):
        for b in range(j + 1):
            out += (comb(i, a) * comb(j, b) * x ** a * dx ** (i - a) * y ** b * dy ** (j - b)
                    / ((i - a) + (j - b) + 1))
    return out


def polygon_raw_moments(vertices: np.ndarray, order: int) -> dict:
    """{(i, j): int_P x^i y^j dA} for i + j <= order; vertices (B, M, 2) in boundary order."""
    x, y = vertices[..., 0], vertices[..., 1]
    xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * yn - xn * y
    raw = {(0, 0): 0.5 * cross.sum(1)}
    for total in range(1, order + 1):
        for i in range(total + 1):
            j = total - i
            raw[(i, j)] = np.sum(cross * _edge_integral(i, j, x, xn, y, yn), axis=1) / (total + 2)
    return raw


def polygon_moments(vertices: np.ndarray, orders):
    """Centroid (B, 2) and central moments (B, len(orders)) of uniform polygons."""
    raw = polygon_raw_moments(vertices, max([sum(a) for a in orders] + [1]))
    area = raw[(0, 0)]
    cx, cy = raw[(1, 0)] / area, raw[(0, 1)] / area
    moments = np.empty((len(vertices), len(orders)))
    for k, (i, j) in enumerate(orders):
        moments[:, k] = sum(comb(i, a) * comb(j, b) * (-cx) ** (i - a) * (-cy) ** (j - b) * raw[(a, b)]
                            for a in range(i + 1) for b in range(j + 1)) / area
    return np.stack([cx, cy], axis=1), moments


# ── Levenberg-Marquardt ───────────────────────────────────────────────────────

def _residuals_and_jacobian(points, target_centroid, target_moments, orders):
    """Residuals (B, R) and Jacobian (B, R, n*D) w.r.t. the flattened point coordinates."""
    B, n, D = points.shape
    w = np.full(n, 1 / n)
    centroid, moments = cluster_moments(points, w, orders)
    delta = points - centroid[:, None]

    J_centroid = np.zeros((B, D, n, D))
    for d in range(D):
        J_centroid[:, d, :, d] = w
    J_moments = np.zeros((B, len(orders), n * D))
    for k, alpha in enumerate(orders):
        for d, a in enumerate(alpha):
            if a == 0:
                continue
            lowered = tuple(b - (j == d) for j, b in enumerate(alpha))
            deriv = a * monomial(delta, lowered)       # d/d delta_d of delta^alpha
            # the centroid moves with every point: subtract the mean derivative
            J_moments[:, k, d::D] = w * (deriv - np.sum(w * deriv, axis=1, keepdims=True))

    r = np.concatenate([centroid - target_centroid, moments - target_moments], axis=1)
    J = np.concatenate([J_centroid.reshape(B, D, n * D), J_moments], axis=1)
    return r, J


def solve_moments_lm(target_centroid, target_moments, orders, init_points,
                     n_iter: int = 50, lambda0: float = 1e-2, verbose: int = 1):
    """
    Batched Levenberg-Marquardt: move the n points of each of the B cells until their centroid
    and central moments match the targets. Stops early once every cost is below
    config.MOMENTS.tol. Returns (points (B, n, D), cost (B,)).
    """
    B, n, D = init_points.shape
    points = init_points.copy()
    eye = np.eye(n * D)[None]
    lam = np.full(B, lambda0)

    r, J = _residuals_and_jacobian(points, target_centroid, target_moments, orders)
    cost = 0.5 * np.sum(r * r, axis=1)

    bar = Progress(f"✦ {D}D moment matching ({B:,} x {n} points)", total=n_iter, verbose=verbose)
    for _ in range(n_iter):
        JT = J.transpose(0, 2, 1)
        JTJ = JT @ J
        lam = np.maximum(lam, np.trace(JTJ, axis1=1, axis2=2) * CFG.min_lambda_rel / JTJ.shape[1])
        step = np.linalg.solve(JTJ + lam[:, None, None] * eye, -(JT @ r[..., None]))[..., 0]

        trial = (points.reshape(B, -1) + step).reshape(B, n, D)
        r_trial, J_trial = _residuals_and_jacobian(trial, target_centroid, target_moments, orders)
        cost_trial = 0.5 * np.sum(r_trial * r_trial, axis=1)

        ok = cost_trial < cost
        points = np.where(ok[:, None, None], trial, points)
        r = np.where(ok[:, None], r_trial, r)
        J = np.where(ok[:, None, None], J_trial, J)
        cost = np.where(ok, cost_trial, cost)
        lam = np.where(ok, lam * CFG.lambda_decrease, lam * CFG.lambda_increase)
        bar.tick(f"cost {cost.max():.1e}")
        if cost.max() < CFG.tol:
            break
    bar.done()
    return points, cost
