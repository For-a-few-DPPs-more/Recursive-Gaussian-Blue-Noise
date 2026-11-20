"""
Helpers of the public API (`api`): argument checks and routing to the solvers.
"""
from __future__ import annotations

from . import config
from .array_lib import keops_available
from .progress import Progress, log
from .solvers.bruteforce import bruteforce_sampler, bruteforce_sampler_keops
from .solvers.clusters import cstit
from .solvers.nufft import nufft
from .solvers.rgbn import rgbn
from .targets import as_target
from .warmstart import jittered_lattice, warmstart_points
from .adaptive.rgbn import adaptive_gbn, adaptive_rgbn

CFG = config.SAMPLE_POINTS
METHODS = ("rgbn", "gaussian", "nufft", "nufft+", "latjit", "cstit")
TARGET_METHODS = ("rgbn", "gaussian", "nufft", "nufft+", "cstit")


# ── sample_points ─────────────────────────────────────────────────────────────

def sample(N, D, lr, method, warmstart, n_iter_scale, target, verbose):
    """Routing of `api.sample_points`."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, must be one of {METHODS}")
    n_iter = n_iter_scale
    
    if target is not None:
        if method not in TARGET_METHODS:
            raise ValueError(f"method {method!r} does not support a target; use one of {TARGET_METHODS}")
        target = as_target(target, D)      # image path or point set, converted by each solver
        if method in ("gaussian", "rgbn"):
            adaptive = adaptive_rgbn if method == "rgbn" else adaptive_gbn
            return adaptive(target, N)
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
        if target is not None and isinstance(warmstart, str):
            x = target.from_unit(x)        # generated in the unit cube; solvers expect the target's frame

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
    """N points following an image (or a 2D point set) with the spectral solver (nufft) tuned for speed."""
    fs = config.NUFFT.fast_stippling
    return nufft(N, 2, target=image, chi=fs["chi"], n_iter=fs["n_iter"], lr=fs["lr"], verbose=verbose)
