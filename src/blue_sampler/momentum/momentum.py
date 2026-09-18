"""Fit a small averaging set X_1, ..., X_n to a target distribution's moments.

Notation follows Lotz & Klatt, "Persistence of asymptotic variance under
transport" (arXiv:2605.22803), Section 5.2 / Eq. (1.8):

    1/lambda_d(C) * int_C x^q dx = 1/n * sum_{j=1}^n X_j^q,   q in {0, ..., p-1}

`p` is the article's p: the moments q = 0, ..., p-1 of the target cell C
(a weighted point cloud for geometry="clusters", or a polygon boundary for
geometry="polygons") are matched by an equal-weight quadrature / averaging
set of n points. q = 0 (total mass) is automatic once weights/areas are
normalized; q = 1 fixes the centroid; q = 2, ..., p-1 are the genuine
central-moment constraints solved for via Levenberg-Marquardt.

There is no `n` argument: n is computed automatically (see `required_n`)
as the smallest point count giving at least as many free coordinates as
scalar constraints. When p == 2 only the centroid is requested, so n = 1
and the target centroid is returned directly with no LM solve at all.
"""

import numpy as np
import warnings
import time

from .momentum_clusters import (
    moment_orders,
    weighted_central_moments,
    solve_moments_lm,
)

from .momentum_polygons import central_moments as polygon_central_moments


MAX_P = 100

def _from_geometry(geometry, gtype, p, verbose, n_iter, **kwargs):
    no_batch = geometry.ndim == 2
    if no_batch:
        geometry = geometry[None]

    if gtype not in ("clusters", "polygons"):
        raise ValueError(
            f"Unknown geometry type: {gtype!r}, "
            f"expected one of ('clusters', 'polygons')"
        )

    center = geometry.mean(axis = 1)[:, None, :]
    radius = np.sqrt(((geometry-center)**2).sum(axis = 2).mean(axis = 1))[:, None, None]
    geometry = (geometry - center)/radius
    points, _ = momentum_fit(
        distribution=geometry,
        distribution_type=gtype,
        p=p,
        verbose = verbose,
        n_iter = n_iter,
        **kwargs
    )

    points = points*radius + center
    if no_batch:
        points = points[0]
    return points

def required_n(p, D):
    """The smallest n for which the equal-weight averaging set has at
    least as many free coordinates (n points x D coords) as scalar
    constraints (D for the centroid, plus one per central-moment order
    q in {2, ..., p-1}): n = ceil((D + len(moment_orders(p, D))) / D).

    This is a necessary condition (not sufficient) for the moment-fit
    problem to be well posed -- it guarantees the system is not
    under-determined, not that an exact solution exists for every target
    distribution.
    """

    n_constraints = D + len(moment_orders(p, D))

    return -(-n_constraints // D)  # ceil division


def _centroid_only(distribution, geometry, weights):
    """Just the target centroid (q = 1), no central moments computed."""

    if geometry == "clusters":

        B, n, _ = distribution.shape

        if weights is None:
            weights = np.full((B, n), 1 / n)

        centroid, _ = weighted_central_moments(
            distribution,
            weights,
            orders=[],
        )

        return centroid

    if geometry == "polygons":

        centroid, _ = polygon_central_moments(
            distribution,
            orders=[],
        )

        return centroid

    raise ValueError(geometry)


def _target(distribution, geometry, orders, weights):

    if geometry == "clusters":

        B, n, _ = distribution.shape

        if weights is None:
            weights = np.full((B, n), 1 / n)

        return weighted_central_moments(
            distribution,
            weights,
            orders,
        )

    if geometry == "polygons":
        return polygon_central_moments(
            distribution,
            orders,
        )

    raise ValueError(geometry)


def _init(distribution, geometry, n, rng):

    B = distribution.shape[0]

    if geometry == "clusters":

        idx = np.argsort(
            rng.random(distribution.shape[:2]),
            axis=1,
        )[:, :n]

        return np.take_along_axis(
            distribution,
            idx[..., None],
            axis=1,
        )

    if geometry == "polygons":

        lo = distribution.min(1)
        hi = distribution.max(1)

        return rng.uniform(
            lo[:, None],
            hi[:, None],
            (B, n, 2),
        )

    raise ValueError(geometry)


def momentum_fit(
    distribution,
    distribution_type="clusters",
    p=3,
    n_per_cells = "auto",
    verbose=1,
    weights=None,
    n_restarts=1,
    restart_tol=1e-30,
    n_iter=100,
    lambda0=1e-2,
    tol=1e-30,
    random_state=None,
):
    """Fit n points X_1, ..., X_n whose centroid and central moments
    (q = 1, ..., p-1) match those of `distribution`.

    n is determined automatically (see `required_n`) or controlled via
    the n_per_cells parametter.
    """

    geometry = distribution_type
    D = distribution.shape[-1]

    if geometry == "polygons":
        assert D == 2

    if not (2 <= p <= MAX_P):
        raise ValueError(f"p must satisfy 2 <= p <= {MAX_P}, got {p}")

    orders = moment_orders(p, D)

    n = required_n(p, D) if n_per_cells == "auto" else n_per_cells

    if geometry == "clusters":
        nc = distribution.shape[-2]
        if n >= nc:
            if verbose >= 1:
                warnings.warn(
                    f"There are few target atoms per cluster (K={nc}) relative to the number\n "
                    f"of points to fit (m={n}), so m was reduced to {nc - 1}.\n "
                    f"Increase cluster size (ideally K >= 2m) to prevent overfitting.\n"
                )
                time.sleep(1)
            n = max(1, nc - 1)
        

    if p == 2 and n == 1:
        # Only q = 1 (the centroid) is requested: n = 1, and the unique
        # point matching the centroid is the centroid itself. No
        # central-moment constraints, no LM solve needed.
        centroid = _centroid_only(distribution, geometry, weights)
        return centroid[:, None, :], None

    target_centroid, target_moments = _target(
        distribution,
        geometry,
        orders,
        weights,
    )

    rng = np.random.default_rng(random_state)

    best_points = None
    best_cost = None
    history = None

    for _ in range(max(1, n_restarts)):

        init = _init(
            distribution,
            geometry,
            n,
            rng,
        )

        points, history, cost = solve_moments_lm(
            target_centroid,
            target_moments,
            orders,
            init,
            verbose = verbose,
            n_iter=n_iter,
            lambda0=lambda0,
            tol=tol,
        )

        if best_points is None:

            best_points = points
            best_cost = cost

        else:

            better = cost < best_cost

            best_points = np.where(
                better[:, None, None],
                points,
                best_points,
            )

            best_cost = np.where(
                better,
                cost,
                best_cost,
            )

        if np.all(best_cost < restart_tol):
            break

    return best_points, history