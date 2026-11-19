"""
Helpers of the public API (`api`): argument checks and routing to the solvers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config, pinwheel
from .array_lib import keops_available
from .density import image_to_points as image_atoms
from .lattice import integer_grid
from .progress import Progress, log
from .solvers.bruteforce import bruteforce_sampler, bruteforce_sampler_keops
from .solvers.clusters import cstit
from .solvers.nufft import nufft
from .solvers.rgbn import rgbn
from .warmstart import jittered_lattice, warmstart_points
from .adaptive.rgbn import adaptive_gbn, adaptive_rgbn

CFG = config.SAMPLE_POINTS
METHODS = ("rgbn", "gaussian", "nufft", "nufft+", "latjit", "cstit")
TARGET_METHODS = ("rgbn", "gaussian", "nufft", "nufft+", "cstit")


# ── sample_points ─────────────────────────────────────────────────────────────

def check_target(target, N: int, method: str, tol: float = 1e-5):
    """Validate a target; images become atoms."""
    if method not in TARGET_METHODS:
        raise ValueError(f"method {method!r} does not support a target; use one of {TARGET_METHODS}, "
                         "or blue_sampler.adaptive for adaptive gaussian blue noise")
    if isinstance(target, str):
        if not Path(target).is_file():
            raise FileNotFoundError(f"target image not found: {target!r}")
        if method in ("nufft", "nufft+"):
            return target                  # nufft uses the image's FFT directly
        return image_atoms(target, N, oversample=CFG.image_oversample[method])
    target = np.asarray(target)
    lo, hi = target.min(), target.max()
    if lo < -tol or hi > 1 + tol or hi - lo < 0.1:
        raise ValueError(f"suspicious target range [{lo:.3g}, {hi:.3g}]: "
                         "expected values in [0, 1)^D with a reasonable spread")
    return target


def sample(N, D, lr, method, warmstart, n_iter_scale, target, verbose):
    """Routing of `api.sample_points`."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, must be one of {METHODS}")
    n_iter = n_iter_scale
    
    if target is not None:
        if method in ("gaussian", "rgbn"):
            adaptive = adaptive_rgbn if method == "rgbn" else adaptive_gbn
            return adaptive(target, N)
        target = check_target(target, N, method)
        n_iter *= CFG.target_iter_factor

    if method == "latjit":
        return jittered_lattice(N, D)
    if method == "cstit":
        if N & (N - 1):
            raise ValueError("cstit needs N to be a power of 2 (recursive cuts of the space in two halves)")
        return cstit(N, D, target, n_iter=config.CSTIT.iter_factor * n_iter_scale, lr=lr, verbose=verbose)

    x = None
    if warmstart is not None:
        x = warmstart_points(N, D, warmstart)

    if method in ("nufft", "nufft+"):
        factor = config.NUFFT.iter_factor[method] if D <= config.NUFFT.max_dim else config.NUFFT.iter_factor_high_d
        chi =config.NUFFT.chi[method]
        newton_finition = config.NUFFT.newton_finition[method]
        return nufft(N, D, n_iter=factor * n_iter, chi=chi, lr=lr, init=x,
                     target=target, verbose=verbose, newton_finition = newton_finition)

    log(f"✦ {D}D blue-noise pipeline — sampling {N:,} points", verbose)
    if n_iter == 0:
        return x
    if method == "gaussian" or N <= CFG.bruteforce_max_n:
        if N >= CFG.keops_min_n and keops_available():
            x = bruteforce_sampler_keops(N, D, lr=lr, verbose=verbose)(x)
        else:
            progress = Progress(f"[gaussian · N = {N}]", verbose=verbose)
            x = bruteforce_sampler(N, D, lr=lr, progress=progress)(x)
            progress.done()
    else:
        x = rgbn(N, D, n_iter, lr=lr, x=x, verbose=verbose)
    log("Done — visualise with blue.plot(x)", verbose)
    return x


def fast_stippling(image: str, N: int, verbose: int = 1):
    """N points following an image with the spectral solver (nufft) tuned for speed."""
    fs = config.NUFFT.fast_stippling
    return nufft(N, 2, target=image, chi=fs["chi"], n_iter=fs["n_iter"], lr=fs["lr"], verbose=verbose)


# ── Tiling, partitions, pinwheel ──────────────────────────────────────────────

def tile(x, repeat: int, flatten: bool):
    N, D = x.shape
    offsets = integer_grid(np.arange(repeat), D) / repeat
    tiled = x[None] / repeat + offsets[:, None, :]
    return tiled.reshape(-1, D) if flatten else tiled.reshape(*([repeat] * D), N, D)


def depth(N: int) -> int:
    """log2(N), N must be a power of 2."""
    d = int(np.log2(N))
    if 2 ** d != N:
        raise ValueError(f"N must be a power of 2 (got N={N}): every recursion step splits each cell in two")
    return d


def tessel_atoms(target, N: int):
    """Target atoms as (1, K, 2), K a multiple of N."""
    target = np.asarray(target)
    target = target[None] if target.ndim == 2 else target
    if target.shape[1] % N:
        raise ValueError(f"the number of target atoms ({target.shape[1]}) must be a multiple of N ({N})")
    return target


def pinwheel_map(points, depth: int):
    """Apply the affine maps of the pinwheel tiling to (2,), (M, 2) or (T, M, 2) points."""
    M, t = pinwheel.affine_maps(pinwheel.BASE_TRIANGLE, pinwheel.unit_square_tiling(depth))
    if points.ndim == 1:
        return np.einsum("nij,j->ni", M, points) + t
    if points.ndim == 2:
        return np.einsum("nij,kj->nki", M, points) + t[:, None, :]
    return np.einsum("nij,nkj->nki", M, points) + t[:, None, :]
