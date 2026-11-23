"""
Public sampling functions: `sample_points` (uniform or along a target) and `image_to_points`.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import config
from .adaptive import adaptive_gbn, adaptive_rgbn
from .backend import keops_available
from .cstit import cstit
from .gbn import gbn, gbn_keops
from .progress import Progress, log
from .rgbn import rgbn
from .spectral import fast_stippling, spectral
from .targets import as_target
from .warmstart import WarmstartMethod, jittered_lattice, warmstart_points

Method = Literal["rgbn", "gaussian", "nufft", "nufft+", "latjit", "cstit"]
METHODS = ("rgbn", "gaussian", "nufft", "nufft+", "latjit", "cstit")
TARGET_METHODS = ("rgbn", "gaussian", "nufft", "nufft+", "cstit")


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
    N blue-noise points in the unit torus [0, 1)^D, as an (N, D) array.

    >>> x = blue.sample_points(N=10_000)
    >>> x = blue.sample_points(N=20_000, target="zebra.jpg")

    Parameters
    ----------
    N, D
        Number of points and dimension. D = 2-3 is fast, 4-5 is supported, above is experimental.
    method
        "rgbn"      recursive gaussian blue noise, fast (default).
        "gaussian"  exact gaussian blue noise over all pairs, O(N^2) per step.
                    Also used by "rgbn" when N <= config.SAMPLE_POINTS.bruteforce_max_n.
        "nufft"     spectral optimisation of the structure factor, by qmc_sampler.sample_l2
                    (see `blue_sampler.spectral`).
        "nufft+"    same, more modes constrained and an L-BFGS finish: slower, better.
        "cstit"     fair clustering + moment matching. Fast, lower quality, N a power of 2.
        "latjit"    a randomly jittered lattice. Instant, ignores every other option.
    lr
        Multiplier of the solver's step sizes. Lower it if the result looks noisy.
    warmstart
        Starting points of the descent (rgbn, gaussian, nufft): "Sobol", "Goodlattice" or an
        (N, D) array (in the target's frame if there is a target). Given a warm start, rgbn
        skips its coarse-to-fine recursion. Default: uniform random points.
    n_iter_scale
        Multiplier of the number of iterations. 0 returns the warm start (Sobol by default).
    target
        Non-uniform density to follow: an image path (D = 2, dark = dense) or a point set
        (K, D) read as samples of the density (see `blue_sampler.targets`). A point set outside
        [0, 1]^D is handled, the samples then come out in its frame.
        rgbn / gaussian switch to their adaptive versions (2D, `blue_sampler.adaptive`,
        constants in config.ADAPTIVE).
        cstit needs a point set of size a multiple of the number of cells.
    verbose
        0: silent, 1: progress bars.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")
    if target is not None:
        if method not in TARGET_METHODS:
            raise ValueError(f"method {method!r} does not take a target, use one of {TARGET_METHODS}")
        target = as_target(target, D)

    if method == "latjit":
        return jittered_lattice(N, D)

    if method == "cstit":
        if N & (N - 1):
            raise ValueError("cstit needs N to be a power of 2 (cells are split in two recursively)")
        n_iter = config.CSTIT.iter_factor * n_iter_scale
        return cstit(N, D, target, n_iter=n_iter, lr=lr, verbose=verbose)

    x0 = None
    if warmstart is not None or n_iter_scale == 0:
        x0 = warmstart_points(N, D, warmstart)
        if target is not None and not isinstance(warmstart, np.ndarray):
            x0 = target.from_unit(x0)       # generated in the unit cube, the solvers expect the target's frame
    if n_iter_scale == 0:
        return x0

    if target is not None and method in ("rgbn", "gaussian"):
        adaptive = adaptive_rgbn if method == "rgbn" else adaptive_gbn
        return adaptive(target, N, lr=lr, n_iter_scale=n_iter_scale, x=x0, verbose=verbose)

    if method in ("nufft", "nufft+"):
        cfg = config.SPECTRAL
        n_iter = (cfg.iter_factor[method] if D <= cfg.max_dim else cfg.iter_factor_high_d) * n_iter_scale
        if target is not None:
            n_iter *= config.SAMPLE_POINTS.target_iter_factor
        return spectral(N, D, n_iter=n_iter, chi=cfg.chi[method], lr=lr, init=x0, target=target,
                        lbfgs_finish=cfg.lbfgs_finish[method], verbose=verbose)

    log(f"✦ {D}D blue-noise pipeline — sampling {N:,} points", verbose)
    if method == "gaussian" or N <= config.SAMPLE_POINTS.bruteforce_max_n:
        if N >= config.SAMPLE_POINTS.keops_min_n and keops_available():
            return gbn_keops(N, D, x0, lr=lr, verbose=verbose)
        progress = Progress(f"[gaussian · N = {N}]", verbose=verbose)
        x = gbn(N, D, x0, lr=lr, progress=progress)
        progress.done()
        return x
    else:
        x = rgbn(N, D, n_iter_scale, lr=lr, x=x0, verbose=verbose)
    log("Done — visualise with blue.plot(x)", verbose)
    return x


def image_to_points(image: str | NDArray, N: int = 2 ** 14, fast: bool = False) -> NDArray:
    """
    Image stippling: N points (N, 2), dense where the image is dark. Plots and returns them.

    image: a path, or a 2D point set read as a density (see `blue_sampler.targets`).
    fast: spectral solver (qmc_sampler) instead of adaptive gaussian blue noise; faster, coarser.
    """
    from . import plot          # structure_factor_v2.plot, re-exported by blue_sampler/__init__

    points = fast_stippling(image, N) if fast else adaptive_rgbn(image, N)
    plot(points, figsize=(8, 8))
    return points
