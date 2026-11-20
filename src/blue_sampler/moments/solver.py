"""Levenberg-Marquardt fit of n equal-weight points to a centroid and central moments."""
from __future__ import annotations

import numpy as np

from .. import config
from ..progress import Progress
from .targets import cluster_moments, monomial

CFG = config.MOMENTS


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
                     n_iter: int = 50, lambda0: float = CFG.lambda0, tol: float = CFG.tol,
                     verbose: int = 1):
    """Batched LM solve. Returns (points (B, n, D), cost (B,))."""
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
        if cost.max() < tol:
            break
    bar.done()
    return points, cost
