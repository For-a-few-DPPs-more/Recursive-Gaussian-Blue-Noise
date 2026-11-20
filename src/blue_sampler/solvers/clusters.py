"""
Balanced recursive clustering in any dimension, and the cstit sampler.

A batch of atoms (B, K, D) is split into (2B, K/2, D): every cluster is cut in two
equal halves at the median of its projection on a random direction of S^(D-1).
The first B clusters are the lower halves, the last B the upper halves.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..lattice import korobov_lattice
from ..moments import fit_points, required_n
from ..targets import as_target, for_cstit

CFG = config.CSTIT


def fair_split(clusters: np.ndarray, rng) -> np.ndarray:
    """(B, K, D) -> (2B, K/2, D), median cut along one random direction per cluster."""
    B, K, D = clusters.shape
    if K % 2:
        raise ValueError(f"K={K} must be even to split into two equal halves")
    directions = rng.normal(size=(B, D)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    order = np.argsort(np.einsum("bkd,bd->bk", clusters, directions), axis=-1)
    clusters = np.take_along_axis(clusters, order[..., None], axis=1)
    return np.concatenate((clusters[:, : K // 2], clusters[:, K // 2:]), axis=0)


def merge_siblings(clusters: np.ndarray) -> np.ndarray:
    """Inverse of `fair_split`: (2B, K, D) -> (B, 2K, D)."""
    half = len(clusters) // 2
    return np.concatenate((clusters[:half], clusters[half:]), axis=1)


def clusterise(depth: int, D: int, atoms: np.ndarray | None = None,
               atoms_per_cluster: int = 16, rng=None) -> np.ndarray:
    """2^depth balanced clusters (2^depth, K / 2^depth, D) of `atoms` (K, D) or (1, K, D)
    (default: a Korobov lattice of 2^depth * atoms_per_cluster points)."""
    rng = np.random.default_rng() if rng is None else rng
    N = 1 << depth
    if atoms is None:
        atoms = korobov_lattice(N * atoms_per_cluster, D)
    atoms = np.asarray(atoms).reshape(1, -1, D).astype(np.float32)
    if atoms.shape[1] % N:
        raise ValueError(f"the number of atoms K={atoms.shape[1]} must be divisible by N={N}")
    for _ in range(depth):
        atoms = fair_split(atoms, rng)
    return atoms


def cstit(N: int, D: int, target=None, n_iter: int = 24, lr: float = 1.0, verbose: int = 1) -> np.ndarray:
    """N points: fair clustering into cells, then moment matching (order 3) in each cell.
    N must be a power of 2. target: image path, point set (K, D, a multiple of the number of
    cells) or `targets.Target`, read as atoms by `targets.for_cstit`."""
    p = CFG.moment_order
    points_per_cell = 2 ** int(np.ceil(np.log2(required_n(p, D)) - 1e-6))
    depth = int(np.log2(N) + 1e-6) - int(np.log2(points_per_cell))
    target = None if target is None else as_target(target, D)
    atoms = for_cstit(target, N, D, n_cells=2 ** depth)
    cells = clusterise(depth, D, atoms, atoms_per_cluster=CFG.atoms_per_cluster)
    points = fit_points(cells, "clusters", p=p, n_iter=n_iter, n_points=points_per_cell,
                        lambda0=lr * CFG.lr_to_lambda, verbose=verbose)
    points = points.reshape(-1, D)
    return points if target is None else target.from_unit(points)
