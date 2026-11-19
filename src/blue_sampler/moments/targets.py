"""Centroids and central moments of the cells to match: point clusters and polygons."""
from __future__ import annotations

from itertools import product
from math import comb

import numpy as np


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


# ── Polygons ──────────────────────────────────────────────────────────────────
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
