"""
Fair partitions of the unit cube (tessellations, clusters), their conversion to points, and
`tile`.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..cstit import clusterise
from ..moments import fit_points
from ..spectral import fast_stippling
from ..targets import as_target, for_cstit
from ..torus import integer_grid
from ..lattices import korobov_lattice
from .tessels import tessellate
from .viz import plot_polygons


def tile(x: NDArray, repeat: int, flatten: bool = True) -> NDArray:
    """
    Pave the unit torus with repeat^D shrunk copies of x (N, D).
    Returns (N * repeat^D, D), or (repeat, ..., repeat, N, D) if flatten=False.

    >>> x4 = bluex.tile(blue.sample_points(N=1000), repeat=2)   # 4000 points
    """
    N, D = x.shape
    offsets = integer_grid(np.arange(repeat), D) / repeat
    tiled = x[None] / repeat + offsets[:, None, :]
    return tiled.reshape(-1, D) if flatten else tiled.reshape(*([repeat] * D), N, D)


def _depth(N: int) -> int:
    """log2(N) for N a power of 2: the number of halvings."""
    depth = int(np.log2(N))
    if 2 ** depth != N:
        raise ValueError(f"N must be a power of 2 (got N={N}): every cell is split in two recursively")
    return depth


# ── Partitions ────────────────────────────────────────────────────────────────

def sample_tessels(N: int = 2 ** 15, D: int = 2, target=None, atoms_per_tessel: int = 128,
                   return_atoms: bool = False) -> NDArray | tuple[NDArray, NDArray]:
    """
    Split the unit square into N random quadrilaterals (N, 4, 2), N a power of 2 (2D only).

    Without target every cut halves the area. With a target (image path or point set, see
    `blue_sampler.targets`) every cut halves the atoms of the target: an image is sampled i.i.d.
    (N * atoms_per_tessel atoms), a point set (K, 2) is used as is (K a multiple of N, K/N >= 100
    recommended), and the quads come out in its frame.
    return_atoms: also return the atoms grouped by quad, (N, K/N, 2).

    >>> pts = bluex.tessels_to_points(bluex.sample_tessels(N=1024)).reshape(-1, 2)
    """
    if D != 2:
        raise ValueError(f"tessels are 2D only (got D={D}), use sample_clusters in higher dimension")
    depth = _depth(N)
    if target is None:
        return tessellate(depth)
    target = as_target(target, 2)
    atoms = for_cstit(target, N, n_cells=N, n_atoms=N * atoms_per_tessel)
    quads, atoms = tessellate(depth, atoms[None])
    quads, atoms = target.from_unit(quads), target.from_unit(atoms)
    return (quads, atoms) if return_atoms else quads


def sample_clusters(N: int = 2 ** 15, D: int = 2, target="korobov", atoms_per_cluster: int = 8) -> NDArray:
    """
    Split a point set into N balanced clusters (N, K/N, D), N a power of 2.

    target: "korokov" (a korobov lattice), None (uniform atoms: N * atoms_per_cluster points of that
    generator, None meaning Sobol), or a target (see `blue_sampler.targets`): an image (D = 2,
    N * atoms_per_cluster i.i.d. atoms) or a point set (K, D), K a multiple of N. The clusters
    come out in the frame of the target.

    >>> pts = bluex.clusters_to_points(bluex.sample_clusters(N=1024)).reshape(-1, 2)
    """
    depth = _depth(N)
    if target is None or (isinstance(target, str) and target == "korobov"):
        return clusterise(depth, D, korobov_lattice(N * atoms_per_cluster, D))
    target = as_target(target, D)
    atoms = for_cstit(target, N, n_cells=N, n_atoms=N * atoms_per_cluster)
    return target.from_unit(clusterise(depth, D, atoms))


def clusters_to_points(clusters: NDArray, p: int = 3, verbose: int = 1, n_iter: int = 20) -> NDArray:
    """Replace each cluster of (N, k, D) by m points matching its moments up to order p - 1
    (p=3: 3 points per cluster in 2D; p=5: 7). Returns (N, m, D)."""
    return fit_points(clusters, "clusters", p=p, n_iter=n_iter, verbose=verbose)


def tessels_to_points(tessels: NDArray, p: int = 3, verbose: int = 1, n_iter: int = 20) -> NDArray:
    """Replace each polygon of (N, M, 2) by m points matching its moments up to order p - 1
    (p=3: 3 points per polygon; p=5: 7). Returns (N, m, 2)."""
    return fit_points(tessels, "polygons", p=p, n_iter=n_iter, verbose=verbose)


def image_to_quads(image, N: int = 2 ** 15, K: int = 4) -> NDArray:
    """
    Image stippling with N quads holding equal shares of the image, plotted and returned
    (N, 4, 2). N is rounded down to a power of 2.
    image: path or 2D point set (see `blue_sampler.targets`). K: atoms per quad, drawn as blue
    noise by fast spectral stippling (qmc_sampler; higher is better, and slower).
    """
    N = 2 ** int(np.log2(N))
    quads = sample_tessels(N=N, target=fast_stippling(image, K * N, verbose=0))
    plot_polygons(quads, color="blue", linewidth=0)
    return quads
