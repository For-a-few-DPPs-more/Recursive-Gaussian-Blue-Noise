"""
Public API.

Points      sample_points, image_to_points, tile, warmstart_points, sobol
Partitions  sample_tessels, sample_clusters, tessels_to_points, clusters_to_points, image_to_quads
Pinwheel    pinwheel_base, pinwheel_transform

Adaptive Gaussian blue noise (image stippling) lives in `blue_sampler.adaptive`.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import pinwheel, utils
from .moments import fit_points
from .solvers.clusters import clusterise
from .solvers.tessels import tessellate
from .viz import plot, plot_polygons
from .warmstart import WarmstartMethod, sobol, warmstart_points  # noqa: F401 (re-exported)

Method = Literal["rgbn", "gaussian", "nufft", "nufft+", "latjit", "cstit"]


# ── Points ────────────────────────────────────────────────────────────────────

def sample_points(
    N: int = 2 ** 15,
    D: int = 2,
    lr: float = 1.0,
    method: Method = "rgbn",
    warmstart: NDArray | WarmstartMethod | None = None,
    n_iter_scale: int = 6,
    target: NDArray | str | None = None,
    verbose: int = 1,
) -> NDArray:
    """
    N blue-noise points in [0, 1)^D, shape (N, D).

    method
        "rgbn"      recursive Gaussian blue noise, fast (default)
        "gaussian"  exact Gaussian blue noise, all pairs, slow for large N (always used for N <= 1000)
        "nufft"     spectral optimisation, chi = 0.35;  "nufft+": chi = 0.43, slower, better
        "cstit"     fair clustering + moment matching, N a power of 2
        "latjit"    randomly jittered lattice
    D           fastest for 2-3, supported up to 5, experimental above
    lr          multiplier of the learning rates
    warmstart   "Goodlattice", "Sobol", "Pinwheel" or an (N, D) array, see `warmstart_points`
    n_iter_scale  multiplier of the number of iterations (0: return the warm start)
    target      (K, D) points in [0, 1)^D or an image path: non-uniform density (nufft, nufft+, cstit)
    verbose     0: silent, 1: progress

    >>> x = blue.sample_points(N=10_000)
    """
    return utils.sample(N, D, lr, method, warmstart, n_iter_scale, target, verbose)


def image_to_points(image: str, N: int = 2 ** 14, fast: bool = False) -> NDArray:
    """
    Image stippling: N points, dense where the image is dark, plotted and returned (N, 2).

    Default: adaptive Gaussian blue noise (`blue_sampler.adaptive`); fast: spectral solver (nufft).
    """
    if fast:
        points = utils.fast_stippling(image, N)
    else:
        from .adaptive import adaptive_rgbn
        points = adaptive_rgbn(image, N)
    plot(points, figsize=(8, 8))
    return points


def tile(x: NDArray, repeat: int, flatten: bool = True) -> NDArray:
    """
    Pave the unit torus with repeat^D shrunk copies of x (N, D).

    Returns (N * repeat^D, D), or (repeat, ..., repeat, N, D) if flatten=False.

    >>> x4 = blue.tile(blue.sample_points(N=1000), repeat=2)   # 4000 points
    """
    return utils.tile(x, repeat, flatten)


# ── Partitions ────────────────────────────────────────────────────────────────

def sample_tessels(N: int = 2 ** 15, D: int = 2, target: NDArray | None = None,
                   return_atoms: bool = False) -> NDArray | tuple[NDArray, NDArray]:
    """
    Split the unit square into N random quadrilaterals (N, 4, 2), N a power of 2 (2D only).

    Without target every cut halves the area; with target atoms (K, 2), K a multiple of N
    (K/N >= 100 recommended), every cut halves the atoms. return_atoms: also return the atoms
    grouped by quad (N, K/N, 2).

    >>> pts = blue.tessels_to_points(blue.sample_tessels(N=1024)).reshape(-1, 2)
    """
    depth = utils.depth(N)
    if target is None:
        return tessellate(depth)
    quads, atoms = tessellate(depth, utils.tessel_atoms(target, N))
    return (quads, atoms) if return_atoms else quads


def sample_clusters(N: int = 2 ** 15, D: int = 2, target: NDArray | WarmstartMethod = "Goodlattice",
                    atoms_per_cluster: int = 8) -> NDArray:
    """
    Split a point set into N balanced clusters (N, K/N, D), N a power of 2.

    target: atoms (K, D), K a multiple of N, or the generator ("Goodlattice", "Sobol",
    "Pinwheel") of N * atoms_per_cluster atoms.

    >>> pts = blue.clusters_to_points(blue.sample_clusters(N=1024)).reshape(-1, 2)
    """
    if not isinstance(target, np.ndarray):
        target = warmstart_points(N * atoms_per_cluster, D, target)
    return clusterise(utils.depth(N), D, target)


def clusters_to_points(clusters: NDArray, p: int = 3, verbose: int = 1, n_iter: int = 20) -> NDArray:
    """Replace each cluster (N, k, D) by m points matching its moments up to order p - 1
    (p=3: 3 points per cluster in 2D; p=5: 7). Returns (N, m, D)."""
    return fit_points(clusters, "clusters", p=p, n_iter=n_iter, verbose=verbose)


def tessels_to_points(tessels: NDArray, p: int = 3, verbose: int = 1, n_iter: int = 20) -> NDArray:
    """Replace each polygon (N, M, 2) by m points matching its moments up to order p - 1
    (p=3: 3 points per tessel; p=5: 7). Returns (N, m, 2)."""
    return fit_points(tessels, "polygons", p=p, n_iter=n_iter, verbose=verbose)


def image_to_quads(image: str, N: int = 2 ** 15, K: int = 4) -> NDArray:
    """
    Image stippling with N quads holding equal shares of the image density, plotted and
    returned (N, 4, 2). K: atoms per quad (higher is better, slower). N is rounded down to a power of 2.
    """
    N = 2 ** int(np.log2(N))
    quads = sample_tessels(N=N, target=utils.fast_stippling(image, K * N, verbose=0))
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
    return utils.pinwheel_map(points, depth)
