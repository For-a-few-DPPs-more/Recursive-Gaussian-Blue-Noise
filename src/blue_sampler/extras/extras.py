"""
Extras of the Public API.

Points      tile, warmstart_points, sobol
Partitions  sample_tessels, sample_clusters, tessels_to_points, clusters_to_points, image_to_quads
Pinwheel    pinwheel_base, pinwheel_transform (see also pinwheel.pinwheel_points)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..samplers import fast_stippling
from ..moments import fit_points
from ..solvers.clusters import clusterise
from ..solvers.tessels import tessellate
from ..targets import as_target, for_cstit
from . import pinwheel, samplers
from .viz import plot_polygons
from ..warmstart import WarmstartMethod, sobol, warmstart_points  # noqa: F401 (re-exported)


def tile(x: NDArray, repeat: int, flatten: bool = True) -> NDArray:
    """
    Pave the unit torus with repeat^D shrunk copies of x (N, D).

    Returns (N * repeat^D, D), or (repeat, ..., repeat, N, D) if flatten=False.

    >>> x4 = blue.tile(blue.sample_points(N=1000), repeat=2)   # 4000 points
    """
    return samplers.tile(x, repeat, flatten)


# ── Partitions ────────────────────────────────────────────────────────────────

def sample_tessels(N: int = 2 ** 15, D: int = 2, target=None, atoms_per_tessel: int = 128,
                   return_atoms: bool = False) -> NDArray | tuple[NDArray, NDArray]:
    """
    Split the unit square into N random quadrilaterals (N, 4, 2), N a power of 2 (2D only).

    Without target every cut halves the area. With a target (see `blue_sampler.targets`) every
    cut halves its atoms: an image is sampled i.i.d. (N * atoms_per_tessel atoms), a point set
    (K, 2) is used as is (K a multiple of N, K/N >= 100 recommended); the quads come out in the
    frame of the target. return_atoms: also return the atoms grouped by quad (N, K/N, 2).

    >>> pts = blue.tessels_to_points(blue.sample_tessels(N=1024)).reshape(-1, 2)
    """
    if D != 2:
        raise ValueError(f"tessels are 2D only (got D={D}); use sample_clusters in higher dimension")
    depth = samplers.depth(N)
    if target is None:
        return tessellate(depth)
    target = as_target(target, 2)
    atoms = for_cstit(target, N, 2, n_cells=N, n_atoms=N * atoms_per_tessel)
    quads, atoms = tessellate(depth, atoms[None])
    quads, atoms = target.from_unit(quads), target.from_unit(atoms)
    return (quads, atoms) if return_atoms else quads


def sample_clusters(N: int = 2 ** 15, D: int = 2, target="Goodlattice", atoms_per_cluster: int = 8) -> NDArray:
    """
    Split a point set into N balanced clusters (N, K/N, D), N a power of 2.

    target: "Goodlattice" / "Sobol" / None (uniform: N * atoms_per_cluster atoms of that generator),
    or a target (see `blue_sampler.targets`): an image (D = 2, sampled i.i.d., N * atoms_per_cluster
    atoms) or a point set (K, D), K a multiple of N. The clusters come out in the frame of the target.

    >>> pts = blue.clusters_to_points(blue.sample_clusters(N=1024)).reshape(-1, 2)
    """
    depth = samplers.depth(N)
    if target is None or (isinstance(target, str) and target in ("Goodlattice", "Sobol")):
        return clusterise(depth, D, warmstart_points(N * atoms_per_cluster, D, target))
    target = as_target(target, D)
    atoms = for_cstit(target, N, D, n_cells=N, n_atoms=N * atoms_per_cluster)
    return target.from_unit(clusterise(depth, D, atoms))


def clusters_to_points(clusters: NDArray, p: int = 3, verbose: int = 1, n_iter: int = 20) -> NDArray:
    """Replace each cluster (N, k, D) by m points matching its moments up to order p - 1
    (p=3: 3 points per cluster in 2D; p=5: 7). Returns (N, m, D)."""
    return fit_points(clusters, "clusters", p=p, n_iter=n_iter, verbose=verbose)


def tessels_to_points(tessels: NDArray, p: int = 3, verbose: int = 1, n_iter: int = 20) -> NDArray:
    """Replace each polygon (N, M, 2) by m points matching its moments up to order p - 1
    (p=3: 3 points per tessel; p=5: 7). Returns (N, m, 2)."""
    return fit_points(tessels, "polygons", p=p, n_iter=n_iter, verbose=verbose)


def image_to_quads(image, N: int = 2 ** 15, K: int = 4) -> NDArray:
    """
    Image stippling with N quads holding equal shares of the image density, plotted and
    returned (N, 4, 2). image: path or 2D point set (see `blue_sampler.targets`); its atoms are
    blue noise (fast nufft stippling). K: atoms per quad (higher is better, slower).
    N is rounded down to a power of 2.
    """
    N = 2 ** int(np.log2(N))
    quads = sample_tessels(N=N, target=fast_stippling(image, K * N, verbose=0))
    plot_polygons(quads, color="blue", linewidth=0)
    return quads


# ── Pinwheel tiling ───────────────────────────────────────────────────────────

def pinwheel_base() -> NDArray:
    """The base Conway triangle: the 1-2-sqrt(5) right triangle of [0, 2] x [0, 1]."""
    return pinwheel.BASE_TRIANGLE.copy()


def pinwheel_transform(points: NDArray | None = None, depth: int = 4) -> NDArray:
    """
    Map points of the base triangle onto every triangle of a pinwheel tiling of [0, 1]^2
    (T = 4 * 5^depth triangles, aperiodic, isotropic).

    points: (2,), (M, 2) the same for every triangle, or (T, M, 2); default the triangle itself.
    Returns (T, M, 2), or (T, 2) for a single point.

    >>> flat = blue.pinwheel_transform(blue.tessels_to_points(blue.pinwheel_base()), depth=4).reshape(-1, 2)
    """
    points = pinwheel.BASE_TRIANGLE if points is None else np.asarray(points)
    return pinwheel.pinwheel_map(points, depth)
