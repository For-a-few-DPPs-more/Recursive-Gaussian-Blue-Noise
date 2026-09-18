"""
Public API.

`blue.sample_points` is the main entry point of the package.
It provides a high-level interface for generating large point sets on the
periodic unit hypercube [0, 1)^D with sub-Poisson density fluctuations
(so-called blue noise).

The package also exposes utilities to sample tessellations (2D only) and
balanced clusters (arbitrary dimension).
Tessels or clusters are sampled with the following balance property:
uniform area repartition if no target is given, or uniform atom repartition
if target atoms are given.

Clusters and tessellations can subsequently be converted into low-discrepancy
point sets using `tessel2points` / `cluster2points`, which internally solve a
moment-matching problem.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing import Literal

from .run.run_bruteforce import _bruteforce_pipeline
from .run.run_recursive import _PRESETS, _recursive_pipeline
from .run.run_nufft import _nufft_pipeline
from .warm_start import _sobol_warmstart, _goodlattice_warmstart, _x_warmstart
from .progress import ProgressLogger

from .run.run_tessels import _tesselation
from .run.run_clusters import _clusterisation, _cstit_pipeline

from .grad.im2fields import _im2targ

from .pinwheels import _BASE, _subdivide, _full_transform

from .momentum.momentum import _from_geometry

from .viz import plot, plot_polygons

BlueNoiseMethod = Literal["rgbn", "nufft", "gaussian", "latjit", "cstit"]
WarmstartMethod = Literal["Goodlattice", "Sobol", "Pinwheel"]
ClusterMethod = Literal["Goodlattice", "Sobol", "Pinwheel"]

def im2points(image: str = "anything.jpg", N: int = 100_000) -> NDArray:
    """
    Image stippling: distribute N blue-noise points according to image brightness.

    A convenience wrapper around `sample_points` that reads an image file and
    uses its luminance as a target density, then plots the result.

    Parameters
    ----------
    image : str, default "anything.jpg"
        Path to the input image (any format supported by matplotlib/PIL).
    N : int, default 100_000
        Number of output points.

    Returns
    -------
    points : ndarray of shape (N, 2)
        The sampled point coordinates in [0, 1)^2.
    """
    points = sample_points(N=N, D=2, targets=image)
    plot(points, figsize=(10, 10))
    return points

def im2quads(image: str = "anything.jpg", N: int = 2**15, K: int = 100) -> NDArray:
    """
    Image stippling with quadrilaterals

    A convenience wrapper around `sample_tessels` that reads an image file and
    uses its luminance as a target density, then plots the result.

    Parameters
    ----------
    image : str, default "anything.jpg"
        Path to the input image (any format supported by matplotlib/PIL).
    N : int, default 100_000
        Number of output points.
    K : int, control quality of the quads by a better scanning of the image
    the bigger K the better, but slower

    Returns
    -------
    quads : ndarray of shape (N, 4, 2)
        The sampled ABCD coordinates of each quad in [0, 1)^2.
    
    Note
    ----
    N must be a power of 2, if not it is silently rounded 
    to the nearest power.
    """
    N = 2**int(np.log2(N))
    points = _im2targ(image, K*N)
    quads = sample_tessels(N = N, targets = points)
    plot_polygons(quads, color = "blue", linewidth = 0)
    return quads

def sample_points(
    N: int = 2**15,
    D: int = 2,
    lr: float = 1.0,
    method: BlueNoiseMethod = "rgbn",
    warmstart: NDArray | WarmstartMethod | None = None,
    n_iter_scale: int = 6,
    targets: NDArray | None = None,
    verbose: int = 1,
) -> NDArray:
    """
    Generate N stealthy (blue-noise) points in [0, 1)^D.

    Parameters
    ----------
    N : int, default 32768
        Number of output points.
    D : int, default 2
        Spatial dimension. Fastest for D=2-3, supported for D=4-5,
        experimental for D≥6.

    lr : float, default 1.0
        Global multiplier for the learning-rate. A default is provided, but
        fine tuning it might give better results

    method : {"gaussian", "rgbn", "nufft", "latjit", "cstit"}, default "rgbn"
        Sampling algorithm:
        - ``"gaussian"`` — Exact Gaussian Blue Noise (GBN), with no
                  neighbourhood truncation. High quality but slow for large N.
        - ``"rgbn"``     — Recursive Gaussian Blue Noise. Fast spatial
          optimisation with a truncated neighbourhood. Recommended.
        - ``"nufft"``    — Spectral optimisation using a Non-Uniform Fast
          Fourier Transform.
        - ``"cstit"``    — Optimisation based on a stable-partition
          criterion, inspired by the fair STIT method.
        - ``"latjit"``   — Randomly jittered lattice. Fast and simple.

    warmstart : {None, "Goodlattice", "Sobol", "Pinwheel", ndarray of shape (N, D)}, default None
        Initial point configuration. See :func:`blue.warmstart_points`.

    n_iter_scale : int, default 6
        Global multiplier for the number of iterations. A default is provided, but
        fine tuning it might give better results.

    targets : ndarray of shape (K, D) or str, optional
        Target points defining a non-uniform density for adaptive
        sampling. A path to an image file can also be given, e.g.
        ``targets="zebra.jpg"``. Only supported with ``method="gaussian, rgbn or cstit"``.

    verbose : int, default 1
        Verbosity level: ``0`` = silent, ``1`` = log progresses.

    Returns
    -------
    points : ndarray of shape (N, D)
        Sampled point coordinates in [0, 1)^D.

    Notes
    -----
    For N <= 1000, ``gaussian`` will be selected automatically because for small number of points there is 
    no need for a more complex method.

    When sampling with a target, be sure that it is normalised and belongs to [0, 1)**D or weird things will happen.
    """

    methods = ["rgbn", "gaussian", "nufft", "latjit", "cstit"]
    n_iter = n_iter_scale
    if method not in methods:
        raise ValueError(f"unknown method {method!r}, must be one of {methods}")


    def assert_valid_target(target, N, method, tol=1e-5):
        points_only = True if method == "cstit" else False
        D2_only = True if method != "cstit" else False
        if isinstance(target, str):
            assert Path(target).is_file(), f"Target image not found: {target!r}"
            if points_only:
                target = _im2targ(target, N, oversample = 32)
            return target
        target  = np.asarray(target)
        if D2_only:
            assert target.shape[-1] == 2, (
                f"only 2D targets are supported with method {method}, use method='cstit' instead"
            )
        a, b = target.min(), target.max()
        assert not (a < -tol or b > 1 + tol or b - a < 0.1), (
            f"Suspicious target range [{a:.3g}, {b:.3g}]. "
            "Expected values in [0, 1)^D with a reasonable spread."
        )
        return target
    
    has_target = targets is not None
    if has_target:        
        if method not in ["bruteforce","rgbn", "cstit"]:
            raise ValueError(
                f"a target density was given but method {method} does not support "
                "a custom target; use method='rgbn', 'bruteforce' or 'cstit' instead."
            )
        targets = assert_valid_target(targets, N, method = method)
        n_iter *= 2

    if method == "latjit":
        prefixD = 1
        suffix = 1
        for s in range(1, 11):
            prefixD = int((N / s) ** (1 / D) + 1e-6) ** D
            if N == prefixD * s:
                suffix = s
                break
        if N != prefixD * s:
            raise ValueError(
                f"for latjit, (N={N}) must be a power of 2 or more generally of the form"
                f"prefix**(D ={D}) * suffix with suffix in [1, 10]"
                "This is to build suffix different lattices with basis prefix"
            )
        return np.concatenate(
            [jitter(prefixD, D, verbose) for _ in range(suffix)]
        )

    if method == "cstit":
        prefix2 = int(np.log2(N) + 1e-6)
        assert N == 2**prefix2, (
            "for cstit, N must be a power of 2, because stit is based of recursive cuts of the space in two halfs "
        )
        return _cstit_pipeline(N, D, targets, verbose, 4*n_iter_scale, lr)

    bruteforce = method == "gaussian" or (N <= 1_000 if D  == 2 else N <= 3_000)
    nufft = method == "nufft"

    has_warmstart = warmstart is not None
    if has_warmstart:
        lr /= 2
        n_iter *= 2
        x = warmstart_points(N, D, warmstart)
    else:
        x = None

    if nufft:
        prefix2 = int(np.log2(N) + 1e-6)
        assert N == 2**prefix2, (
            "for nufft, N must be a power of 2 for performance." 
            "Non power of 2 case would be much slower for the kdtree part and is not implemented "
        )
        return _nufft_pipeline(N, D, lr=lr, warmstart=x,
                               verbose=verbose, n_iter= 20 * n_iter)

    if verbose >= 1:
        print(f"✦ {D}D blue-noise pipeline — sampling {N:,} points")

    if n_iter == 0:
        return x

    logger = ProgressLogger(D, verbose)
    if bruteforce:
        if D == 2:
            n_iter *= max(10, int(N/24))
        if D == 3:
            n_iter *= max(10, int(N/600))
        if D >= 4:
            n_iter *= max(10, int(N/2000))
        ctx = logger.enter_level(N, D, 0)
        ctx.start()
        blue = _bruteforce_pipeline(
            N, D, n_iter, ctx=ctx,
            lr=lr,
            target=targets,
        )
        sampled_points = np.array(blue(x))
        logger.exit_level()
    elif method == "rgbn":
        preset = _PRESETS[min(D, 5)]
        sampled_points = _recursive_pipeline(
            N=N,
            D=D,
            N_ITER=n_iter,
            logger=logger,
            S=preset["S"],
            expension_factor=preset["expension_factor"],
            LR_spatial=lr * preset["LR_spatial"],
            LR_spectral=lr * preset["LR_spectral"],
            spatial_radius=preset["spatial_radius"],
            spectral_radius=preset["spectral_radius"],
            N_PER_STEP=10,
            x=x,
            target=targets,
        )

    if verbose >= 1:
        print("Done — visualise with blue.plot(x)")

    return sampled_points


def sample_tessels(
    N: int = 2**15,
    D: int = 2,
    targets:  NDArray | None = None,
    return_atoms: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    """
    Recursively split the unit square into N random quadrilaterals (2D only).

    If ``targets`` is None, splits are chosen to achieve equal areas.
    If ``targets`` is provided, splits are chosen to achieve a balanced
    median separation of the atoms.

    Parameters
    ----------
    N : int, default 32768
        Number of output quadrilaterals. **Must be a power of 2.**
    D : int = 2
        STIT tesselation is ONLY for 2D, the D argument is only 
        for consistancy with other methods.
    targets : ndarray of shape (K, 2), optional
        Coordinates of atoms to split, in the [0, 1)² unit box.
        K must be a multiple of N. The more targets provided the better
        the approximation, but the slower the computation
        (K/N ≥ 100 is recommended for a decent tessellation).
        A typical use case is adaptive tessellation, e.g. with ``targets``
        being i.i.d. points sampled from a target density.
    return_atoms : bool, default False
        If True and ``targets`` is provided, also return the atoms
        redistributed among their final quadrilateral.

    Returns
    -------
    quad : ndarray of shape (N, 4, 2)
        A tessellation of N quadrilaterals with equal area (or equal atom count).
    atoms : ndarray of shape (N, K//N, 2)
        The input target atoms, redistributed among their quadrilateral.
        Only returned when ``targets`` is provided **and** ``return_atoms=True``.

    Notes
    -----
    Only 2D geometry is supported. N must be a power of 2 because each
    recursion step splits every quadrilateral into exactly two.

    To convert the tessellation into a flat point set, pass the output to
    `tessel2points` and then call `.reshape(-1, 2)`:

    >>> ts = blue.sample_tessels(N=1024)
    >>> pts = blue.tessel2points(ts).reshape(-1, 2)   # (1024 * m, 2)
    """
    depth = int(np.log2(N))
    if 2**depth != N:
        raise ValueError(
            f"N must be a power of 2 (got N={N}). "
            "Each recursion step splits every quadrilateral into exactly two, "
            "so only power-of-2 counts are supported."
        )

    if targets is not None:
        if targets.ndim == 2:
            targets = targets[None, ...]
        if targets.shape[1] % N != 0:
            raise ValueError(
                f"The number of target atoms ({targets.shape[1]}) must be a "
                f"multiple of N ({N})."
            )

    if return_atoms or targets is None:
        return _tesselation(depth, targets)
    return _tesselation(depth, targets)[0]


def sample_clusters(
    N: int = 2**15,
    D: int = 2,
    targets: NDArray | ClusterMethod  = "Goodlattice",
    n_per_cluster: int = 8,
) -> NDArray:
    """
    Recursively partition a point set into N balanced clusters.

    At each recursion step every cluster is split into two equal halves using
    a random median hyperplane. After log₂(N) levels exactly N clusters are
    obtained.

    Parameters
    ----------
    N : int, default 32768
        Number of output clusters. **Must be a power of two.**
    D : int, default 2
        Ambient dimension.
    targets : ndarray of shape (K, D) or method name, optional
        Initial atoms to cluster. Default is  a Good lattice = lattice
        with a basis found by Korobov optimization to properly match the shape (K, D).
        If a method name  is given, a sequence of K = N * n_per_cluster 
        atoms is generated automatically following the method.
        Supported methods are ("Goodlattice", "Sobol", "Pinwheel").
    n_per_cluster : int, default 16
        Number of atoms per final cluster. Only used when ``targets`` is
        not provided.

    Returns
    -------
    ndarray of shape (N, K//N, D)
        Collection of N balanced clusters, each containing K//N atoms.

    Notes
    -----
    The recursive splitting requires the total atom count K to be divisible
    by N.

    To convert clusters into a flat point set, pass the output to
    `cluster2points` and then call `.reshape(-1, D)`:

    >>> cl = blue.sample_clusters(N=1024, D=2)
    >>> pts = blue.cluster2points(cl).reshape(-1, 2)   # (1024 * m, 2)
    """
    depth = int(np.log2(N))
    if (1 << depth) != N:
        raise ValueError(
            f"N={N} must be a power of two. "
            "Each recursion step splits every cluster into exactly two."
        )

    if isinstance(targets, np.ndarray):
        if targets.ndim == 2:
            targets = targets[None, :, :]
        K = targets.shape[1]
        if K % N != 0:
            raise ValueError(
                f"The number of target atoms ({K}) must be divisible by N ({N})."
            )
    else:
        targets = warmstart_points(N*n_per_cluster, D, targets)[None, :, :]

    return _clusterisation(
        depth=depth,
        D=D,
        targets=targets,
        n_per_cluster=n_per_cluster,
    )


def tile(x: NDArray, repeat: int, flatoutput: bool = True) -> NDArray:
    """
    Tile points on the unit torus to cover [0, 1)^D periodically.

    Each of the ``repeat**D`` copies of ``x`` is rescaled by ``1/repeat``
    and shifted to its own sub-cube, so that the copies together pave the
    unit torus again. For example in 2D with ``repeat=2``: tile (0, 0) holds
    ``x/2``, tile (1, 1) holds ``x/2 + 0.5``, etc.

    Parameters
    ----------
    x : ndarray of shape (N, D)
        Points in [0, 1)^D (unit torus).
    repeat : int
        Number of repetitions per axis. The output contains
        ``N_final = N * repeat**D`` points.
    flatoutput : bool, default True
        If True, return shape is ``(N_final, D)``.
        If False, the tile structure is kept as leading axes:
        ``(repeat, ..., repeat, N, D)``.

    Returns
    -------
    ndarray of shape (N_final, D) if flatoutput else (repeat, ..., repeat, N, D)
        Tiled version of ``x``, periodised over [0, 1)^D.

    Examples
    --------
    >>> x = blue.sample_points(N=1000, D=2)
    >>> x4 = blue.tile(x, repeat=2)   # 4 000 points covering [0,1)^2
    """
    N, D = x.shape

    grids = np.meshgrid(*([np.arange(repeat)] * D), indexing="ij")
    idx = np.stack(grids, axis=-1)                             # (repeat,)*D + (D,)
    offset = (idx / repeat).reshape(*([repeat] * D), 1, D)    # (repeat,)*D + (1, D)

    x_scaled = x / repeat                                      # (N, D)
    xtiled = x_scaled + offset                                 # (repeat,)*D + (N, D)

    return xtiled.reshape(-1, D) if flatoutput else xtiled


def cluster2points(clusters: NDArray, p: int = 3, verbose: int = 1, n_iter: int = 20) -> NDArray:
    """
    Convert a batch of clusters into a low-discrepancy point set via moment matching.

    Each cluster is replaced by ``m`` representative points by solving a
    moment-matching problem (Levenberg–Marquardt) up to polynomial order ``p``.

    Parameters
    ----------
    clusters : ndarray of shape (N, k, D)
        Batch of N clusters, each containing k atoms in D dimensions.
        Typically the output of `sample_clusters`, or the ``atoms`` output
        of ``sample_tessels(..., return_atoms=True)``.
    p : int, default 3
        Maximum total moment order to match (centroid + central moments up
        to order ``p``). Higher ``p`` places more points per cluster (more
        constraints to satisfy) and yields a denser, more accurate result.
        ``p=3`` → 3 points per cluster in 2D; ``p=5`` → 7 points per cluster.
    verbose : int, default 1
            Verbosity level: ``0`` = silent, ``1`` = live progress bar with ETA.
    n_iter : int, default 20
            Number of iterations of the solver.
    Returns
    -------
    ndarray of shape (N, m, D)
        ``m`` representative points per cluster matching its moments up to
        order ``p``. To obtain a flat ``(N*m, D)`` point array, call
        ``.reshape(-1, D)`` on the result.

    Examples
    --------
    >>> cl = blue.sample_clusters(N=512, D=2)
    >>> pts = blue.cluster2points(cl).reshape(-1, 2)
    """
    return _from_geometry(clusters, "clusters", p, verbose, n_iter)


def tessel2points(tessels: NDArray, p: int = 3, verbose: int = 1, n_iter: int = 20) -> NDArray:
    """
    Convert a batch of quadrilateral tessels into a low-discrepancy point set.

    Each tessel is replaced by ``m`` representative points by solving a
    moment-matching problem (Levenberg–Marquardt) up to polynomial order ``p``.

    Parameters
    ----------
    tessels : ndarray of shape (N, 4, 2)
        Batch of N quadrilaterals, each defined by 4 vertices in 2D.
        Typically the ``quad`` output of `sample_tessels`.
    p : int, default 3
        Maximum total moment order to match (centroid + central moments up
        to order ``p``). Higher ``p`` places more points per tessel and
        yields a denser, more accurate result.
        ``p=3`` → 3 points per tessel; ``p=5`` → 7 points per tessel.
    verbose : int, default 1
                Verbosity level: ``0`` = silent, ``1`` = live progress bar with ETA.
    n_iter : int, default 20
            Number of iterations of the solver.

    Returns
    -------
    ndarray of shape (N, m, 2)
        ``m`` representative points per tessel matching its moments up to
        order ``p``. To obtain a flat ``(N*m, 2)`` point array, call
        ``.reshape(-1, 2)`` on the result.

    Examples
    --------
    >>> ts = blue.sample_tessels(N=512)
    >>> pts = blue.tessel2points(ts).reshape(-1, 2)
    """
    return _from_geometry(tessels, "polygons", p, verbose, n_iter)


def sobol(N: int = 2**15, D: int = 2) -> NDArray:
    """
    Generate N points in [0, 1)^D from a scrambled Sobol sequence.

    A lightweight wrapper around ``scipy.stats.qmc.Sobol``.

    Parameters
    ----------
    N : int, default 32768
        Number of output points. For best uniformity N should be a power of 2;
        a ``UserWarning`` is emitted by SciPy if it is not.
    D : int, default 2
        Spatial dimension.

    Returns
    -------
    points : ndarray of shape (N, D)
        Sobol points in [0, 1)^D.

    Notes
    -----
    Sobol sequences have much lower discrepancy than i.i.d. uniform samples
    and are useful as initialisations (warm starts) for other samplers.
    Requires ``scipy`` (``pip install scipy``).
    """
    return _sobol_warmstart(N=N, D=D)


def pinwheel_base() -> NDArray:
    """
    Return the base Conway triangle for pinwheel aperiodic tiling.

    Returns
    -------
    ndarray of shape (3, 2)
        The three vertices of the base 1-2-√5 right triangle in [0, 2] × [0, 1].
    """
    return _BASE.copy()


def pinwheel_transform(
    points: NDArray = pinwheel_base(),
    depth: int = 4,
) -> NDArray:
    """
    Apply a Pinwheel tiling transformation to a set of points.

    Recursively subdivides the Conway triangle (Pinwheel tiling) and maps
    the input points onto each resulting triangle, producing a fractal,
    aperiodic tiling.

    Parameters
    ----------
    points : ndarray, default ``pinwheel_base()``
        The coordinates of points defined on the base Conway triangle.

        - shape ``(2,)``    — one point projected onto every triangle.
        - shape ``(M, 2)``  — M points projected onto every triangle.
        - shape ``(N, M, 2)`` — each set of M points mapped to its own triangle.
    depth : int, default 4
        Number of subdivision iterations. The number of triangles grows as
        ``4 × 5^depth``.

    Returns
    -------
    ndarray of shape (T, M, 2)
        Transformed points for each of the T triangles at the given depth.
        T = 4 × 5^depth.

    Notes
    -----
    The Pinwheel tiling is non-periodic: each triangle is subdivided into 5
    smaller triangles, each rotated by arctan(1/2) relative to its parent.
    All orientations are dense on the circle, making the tiling isotropic.

    Examples
    --------
    >>> pw0 = blue.pinwheel_base()          # (3, 2) base triangle
    >>> pts = blue.tessel2points(pw0)       # (3, 3, 2) — 3 points per sub-region
    >>> tiling = blue.pinwheel_transform(pts, depth=4)  # (T, 3, 2)
    >>> flat = tiling.reshape(-1, 2)        # flatten to a point set
    """
    tiling = pinwheel_base()[None]
    for _ in range(depth):
        tiling = _subdivide(tiling)

    tiling = np.concatenate([tiling, -tiling + np.array([[2.0, 1.0]])], axis=0)
    tiling = np.concatenate(
        [tiling, tiling * np.array([[-1.0, 1.0]]) + np.array([[2.0, 1.0]])], axis=0
    )
    M, t = _full_transform(pinwheel_base(), tiling / 2.0)

    if points.ndim == 1:
        points = np.einsum("nij,j-> ni", M, points) + t
    elif points.ndim == 2:
        points = np.einsum("nij,kj->nki", M, points) + t[:, None, :]
    else:
        points = np.einsum("nij,nkj->nki", M, points) + t[:, None, :]
    return points


def _pinwheel_warmstart(N: int) -> NDArray:
    """
    Internal helper: sample exactly N 2D blue-noise points via Pinwheel tiling.

    Uses tessel2points + pinwheel_transform then crops/wraps to exactly N points.
    Only valid for D=2.
    """
    xbase = tessel2points(pinwheel_base(), p=3, verbose = 0)# (3, 3, 2) → 3 pts on base triangle
    depth = int(np.log(N / 3) / np.log(5) + 1)
    intensity = 3 * 4 * 5**depth
    x = pinwheel_transform(xbase, depth=depth)             # (4*5^depth, 3, 2)
    return _x_warmstart(x, N, intensity=intensity)         # (N, 2)


def warmstart_points(
    N: int,
    D: int,
    method: WarmstartMethod | NDArray | None = None,
    seed: int | None = None,
) -> NDArray:
    """
    Generate an initial point cloud in [0, 1)^D.

    Parameters
    ----------
    N : int
        Number of points.

    D : int
        Ambient dimension.

    method : {"Goodlattice", "Sobol", "Pinwheel"} or ndarray or None
        Warm start strategy. If an ndarray is provided, it must already
        contain points of shape (N, D).

    seed : int or None, optional
        Random seed used by stochastic warm start generators.

    Returns
    -------
    sample : np.ndarray of shape (N, D)
        Initial point cloud in the unit hypercube [0, 1)^D.

    Raises
    ------
    ValueError
        If the provided warm start array has an incorrect shape or if the
        requested method is unsupported.
    """
    if method is None:
        return _sobol_warmstart(N, D, seed=seed)

    if isinstance(method, np.ndarray):
        if method.shape != (N, D):
            raise ValueError(
                f"warmstart array must have shape {(N, D)}, "
                f"got {method.shape}"
            )
        return method.copy()

    if method == "Sobol":
        return _sobol_warmstart(N, D, seed=seed)

    if method == "Goodlattice":
        return _goodlattice_warmstart(N, D, seed=seed)

    if method == "Pinwheel":
        if D == 2:
            return _pinwheel_warmstart(N)
        warnings.warn(
            f"warmstart='Pinwheel' is only supported for D=2 (got D={D}); "
            "falling back to Sobol initialisation.",
            UserWarning,
            stacklevel=2,
        )
        return _sobol_warmstart(N, D, seed=seed)

    raise ValueError(
        f"unsupported warmstart={method!r}; expected None, "
        "'Goodlattice', 'Sobol', 'Pinwheel', or an ndarray of shape (N, D)."
    )

def jitter(N, D, verbose):
    """perturbed lattice based, latjit blue noise method"""
    n = int(N ** (1 / D) + 1e-6)
    if verbose >= 1 and (n**D != N):
        warnings.warn(
            f"user-given number of points N = {N} is not a power of dimension D = {D}; "
            f"N will be rounded to {n}^{D} = {n**D} to build the lattice",
            UserWarning,
            stacklevel=2,
        )
    axes = [
        (np.linspace(0, 1, n, endpoint = False) + 0.5) 
        for _ in range(D)
    ]
    lattice = np.stack(
        np.meshgrid(*axes, indexing="ij"),
        axis=-1
    ).reshape(-1, D) + np.random.rand(1, D)
    u = (np.random.rand(len(lattice), D) - 0.5)/n
    x = (lattice + u) % 1.0
    return x