"""
Balanced recursive tessellation of the unit square into convex quadrilaterals (2D only).

Each quadrilateral (4 ordered vertices) is cut by a straight line of random direction.
- Equal-area mode (no atoms): the cut position solves area(piece) = area / 2, by a
  quadratic interpolation of the area between the two middle vertex levels.
- Median mode (atoms given): the cut goes through the median of the atoms of the cell,
  so that each piece receives half of them.
A direction is valid when the cut crosses two opposite edges (both pieces are
quadrilaterals); `pool` directions are tried at once, more if none is valid.

Polytopes become complicated in higher dimension: see clusters.py for any D.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..geometry import cross2d

CFG = config.TESSELS
UNIT_SQUARE = np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])


def _area4(P0, P1, P2, P3):
    return 0.5 * (cross2d(P0, P1) + cross2d(P1, P2) + cross2d(P2, P3) + cross2d(P3, P0))


def _on_edge(V1, V2, h1, h2, t, eps):
    """Point of the edge V1 V2 at level t, the levels of V1, V2 being h1, h2.
    Broadcasts over a trailing 'pool' axis when t has one more axis than V1."""
    alpha = (t - h1) / np.where(np.abs(h2 - h1) < eps, eps, h2 - h1)
    if t.ndim == V1.ndim:                                   # (B, pool)
        return V1[:, None, :] + alpha[..., None] * (V2 - V1)[:, None, :]
    return V1 + alpha[:, None] * (V2 - V1)


def _equal_area_level(l1, lm, l2, f1, fm, f2, eps):
    """Root in [l1, l2] of the quadratic through (l1, f1), (lm, fm), (l2, f2)."""
    dx = 0.5 * (l2 - l1)
    safe_dx = np.where(dx < eps, eps, dx)
    a = (f2 + f1 - 2.0 * fm) / (2.0 * safe_dx ** 2)
    b = (f2 - f1) / (2.0 * safe_dx)
    sqrt_disc = np.sqrt(np.maximum(b ** 2 - 4 * a * fm, 0.0))
    two_a = np.where(np.abs(a) < eps, eps, 2 * a)
    linear = np.abs(a) < eps
    x = np.where(linear, -fm / np.where(np.abs(b) < eps, eps, b), (-b + sqrt_disc) / two_a)
    x = np.where((np.abs(x) > dx) & ~linear, (-b - sqrt_disc) / two_a, x)
    return np.clip(lm + x, l1, l2)


def _try_split(quads, atoms, pool, rng, eps):
    """One vectorised split attempt with `pool` random directions per quad.
    Returns (piece1, piece2, atoms1, atoms2, ok) with ok[b] = a valid direction was found."""
    A, B, C, D = (quads[:, i] for i in range(4))
    theta = rng.uniform(0.0, np.pi, size=(len(quads), pool))
    nx, ny = -np.sin(theta), np.cos(theta)                  # normal of the cut
    level = lambda P: P[:, None, 0] * nx + P[:, None, 1] * ny
    hA, hB, hC, hD = level(A), level(B), level(C), level(D)
    h = np.stack((hA, hB, hC, hD), axis=-1)
    h_sorted = np.sort(h, axis=-1)
    l1, l2 = h_sorted[..., 1], h_sorted[..., 2]             # admissible levels: between the middle vertices
    low = h <= l1[..., None]
    cuts_DA_BC = (low[..., 0] & low[..., 1]) | (low[..., 2] & low[..., 3])

    if atoms is None:
        half_area = (_area4(A, B, C, D) * 0.5)[:, None]

        def area_below(t):
            return np.where(
                cuts_DA_BC,
                _area4(A[:, None], B[:, None], _on_edge(B, C, hB, hC, t, eps), _on_edge(D, A, hD, hA, t, eps)),
                _area4(_on_edge(A, B, hA, hB, t, eps), B[:, None], C[:, None], _on_edge(C, D, hC, hD, t, eps)),
            ) - half_area

        lm = 0.5 * (l1 + l2)
        f1, fm, f2 = area_below(l1), area_below(lm), area_below(l2)
        valid = f1 * f2 <= 0
    else:
        h_atoms = atoms[:, None, :, 0] * nx[..., None] + atoms[:, None, :, 1] * ny[..., None]
        order = np.argsort(h_atoms, axis=-1)
        h_atoms = np.take_along_axis(h_atoms, order, axis=-1)
        K2 = atoms.shape[1] // 2
        t_median = 0.5 * (h_atoms[..., K2 - 1] + h_atoms[..., K2])
        valid = (l1 <= t_median) & (t_median <= l2)

    ok = np.any(valid, axis=1)
    sel = np.argmax(valid, axis=1)[:, None]                 # first valid direction
    pick = lambda arr: np.take_along_axis(arr, sel, axis=1)[:, 0]
    hA, hB, hC, hD, cuts_DA_BC = pick(hA), pick(hB), pick(hC), pick(hD), pick(cuts_DA_BC)
    if atoms is None:
        t = _equal_area_level(pick(l1), pick(lm), pick(l2), pick(f1), pick(fm), pick(f2), eps)
    else:
        t = pick(t_median)

    P_AB, P_BC = _on_edge(A, B, hA, hB, t, eps), _on_edge(B, C, hB, hC, t, eps)
    P_CD, P_DA = _on_edge(C, D, hC, hD, t, eps), _on_edge(D, A, hD, hA, t, eps)
    mask = cuts_DA_BC[:, None, None]
    piece1 = np.where(mask, np.stack((A, B, P_BC, P_DA), axis=1), np.stack((P_AB, B, C, P_CD), axis=1))
    piece2 = np.where(mask, np.stack((C, D, P_DA, P_BC), axis=1), np.stack((P_CD, D, A, P_AB), axis=1))
    if atoms is None:
        return piece1, piece2, None, None, ok

    sorted_atoms = np.take_along_axis(atoms, np.take_along_axis(order, sel[..., None], axis=1)[:, 0, :, None], axis=1)
    below, above = sorted_atoms[:, :K2], sorted_atoms[:, K2:]
    c1 = piece1.mean(axis=1)
    piece1_below = (c1[:, 0] * pick(nx) + c1[:, 1] * pick(ny) < t)[:, None, None]
    return piece1, piece2, np.where(piece1_below, below, above), np.where(piece1_below, above, below), ok


def fair_split(quads, atoms=None, rng=None):
    """(B, 4, 2) -> (2B, 4, 2) (and atoms (B, K, 2) -> (2B, K/2, 2)), retrying with larger pools."""
    rng = np.random.default_rng() if rng is None else rng
    B = len(quads)
    pieces = np.empty((2, B, 4, 2), dtype=quads.dtype)
    halves = None if atoms is None else np.empty((2, B, atoms.shape[1] // 2, 2), dtype=atoms.dtype)
    todo, pool = np.arange(B), CFG.pool
    for _ in range(CFG.max_retries + 1):
        if len(todo) == 0:
            break
        p1, p2, a1, a2, ok = _try_split(quads[todo], None if atoms is None else atoms[todo], pool, rng, CFG.eps)
        done = todo[ok]
        pieces[0, done], pieces[1, done] = p1[ok], p2[ok]
        if atoms is not None:
            halves[0, done], halves[1, done] = a1[ok], a2[ok]
        todo, pool = todo[~ok], pool * CFG.pool_growth
    if len(todo):
        raise RuntimeError(f"no valid quad/quad split found for {len(todo)} polygon(s)")
    quads = pieces.reshape(2 * B, 4, 2)
    return quads if atoms is None else (quads, halves.reshape(2 * B, -1, 2))


def tessellate(depth: int, atoms=None):
    """2^depth quads tiling the unit square (or, with atoms (1, K, 2), their bounding box,
    each quad holding K / 2^depth atoms). Returns quads, or (quads, atoms)."""
    if atoms is None:
        quads = UNIT_SQUARE
        for _ in range(depth):
            quads = fair_split(quads)
        return quads
    atoms = np.array(atoms, dtype=float)                    # copy: the caller's array is left untouched
    lo = atoms.min()
    scale = (atoms - lo).max()
    atoms = np.clip((atoms - lo) / scale, CFG.eps, 1.0 - CFG.eps)
    quads = UNIT_SQUARE
    for _ in range(depth):
        quads, atoms = fair_split(quads, atoms)
    return scale * quads + lo, scale * atoms + lo


def merge_siblings(quads, eps: float = 1e-7):
    """Inverse of one `fair_split` level: rebuild each parent from its two children."""
    half = len(quads) // 2
    p1, p2 = quads[:half], quads[half:]
    cut_DA_BC = np.any(np.linalg.norm(p1[:, 2, None, :] - p2, axis=-1) < eps, axis=-1)[:, None, None]
    return np.where(cut_DA_BC,
                    np.stack((p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1]), axis=1),
                    np.stack((p2[:, 2], p1[:, 1], p1[:, 2], p2[:, 1]), axis=1))
