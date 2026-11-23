"""
Spectral blue noise, borrowed from qmc_sampler (pip install qmc-sampler).

blue_sampler does not implement the spectral (NUFFT) solver any more: `sample_points(method=
"nufft" | "nufft+")`, `image_to_points(fast=True)` and `extras.image_to_quads` call
`qmc_sampler.sample_l2`. This module only translates blue_sampler's arguments:

- a `Target` becomes the target point cloud qmc_sampler expects, in the unit cube
  (an image is sampled i.i.d., config.SPECTRAL.image_oversample points per output point);
- the result is mapped back to the target's frame.

Backend, precision and NUFFT tolerance are qmc_sampler's own settings
(`blue.select_backend`, `blue.select_precision`, `blue.select_nufft_tolerance`).
"""
from __future__ import annotations

import numpy as np
import qmc_sampler

from . import config
from .targets import Target, as_target, sample_image

CFG = config.SPECTRAL


def target_points(target: Target, N: int) -> np.ndarray:
    """The target as a point cloud in the unit cube, as qmc_sampler reads it."""
    if target.kind == "image":
        return sample_image(target.path, CFG.image_oversample * N)
    return target.points


def spectral(N: int, D: int, *, chi: float, n_iter: int, lbfgs_finish: bool, lr: float = 1.0,
             init: np.ndarray | None = None, target=None, verbose: int = 1) -> np.ndarray:
    """
    N points (N, D) by qmc_sampler.sample_l2 (isotropic L2 ball of modes).

    init: warm start, in the target's frame. target: image path, point set or `Target`.
    With lbfgs_finish, n_iter is split evenly between gradient steps and L-BFGS.
    """
    target = None if target is None else as_target(target, D)
    tgt = None if target is None else target_points(target, N)
    x0 = None if init is None else (init if target is None else target.to_unit(init)) % 1.0
    x = qmc_sampler.sample_l2(
        N, D, Chi=chi, n_iter=max(1, n_iter // 2 if lbfgs_finish else n_iter), lr=lr,
        quasi_newton_finition=lbfgs_finish, warmstart=x0, target=tgt, verbose=verbose,
    )
    return x if target is None else target.from_unit(x)


def fast_stippling(target, N: int, verbose: int = 1) -> np.ndarray:
    """N 2D points following an image or a 2D point set, tuned for speed over quality."""
    fs = CFG.fast_stippling
    return spectral(N, 2, target=target, chi=fs["chi"], n_iter=fs["n_iter"], lr=fs["lr"],
                    lbfgs_finish=False, verbose=verbose)
