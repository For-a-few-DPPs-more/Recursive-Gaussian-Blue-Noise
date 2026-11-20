"""
A target in the form each solver family expects.

- nufft:     target Fourier coefficients on the model's modes
- cstit:     atoms, a multiple of the number of cells
- adaptive:  density grid (gbn / rgbn of `blue_sampler.adaptive`)
"""
from __future__ import annotations

import numpy as np

from .. import config
from .base import Target, as_target
from .image import image_fourier, load_image, sample_image
from .points import points_to_density

CFG = config.TARGETS


def for_nufft(target, model):
    """
    Target spectrum for a nufft model (`solvers.nufft`): image -> its Fourier coefficients on the
    model's modes; points -> their NUFFT. None -> 0.0 (uniform).
    """
    if target is None:
        return 0.0
    t = as_target(target, model.D)
    if t.kind == "image":
        return model.from_modes(image_fourier(t.path, model.modes, sign=model.sign))
    return model.spectrum(t.points)


def for_cstit(target, N: int, D: int, n_cells: int, n_atoms: int | None = None, rng=None) -> np.ndarray | None:
    """
    Atoms (K, D) in the unit cube for the partition samplers with `n_cells` cells (cstit, and the
    extras sample_clusters / sample_tessels): image -> n_atoms i.i.d. atoms (default
    CFG.cstit_oversample * N, rounded up to a multiple of n_cells); points -> the points, whose number
    must be at least N and a multiple of n_cells. None -> None (uniform).
    """
    if target is None:
        return None
    t = as_target(target, D)
    if t.kind == "image":
        K = CFG.cstit_oversample * N if n_atoms is None else n_atoms
        K += (-K) % n_cells
        return sample_image(t.path, K, rng=rng)
    K = len(t.points)
    if K < N:
        raise ValueError(f"need at least N = {N:,} target points (comfortably more), got {K:,}")
    if K % n_cells:
        raise ValueError(f"need a number of target points multiple of the number of cells "
                         f"({n_cells}), got {K:,}: keep the first {K - K % n_cells:,}")
    return t.points


def for_adaptive(target, shape: tuple[int, int], invert: bool = True) -> np.ndarray:
    """
    Density grid of `shape` indexed [x, y], summing to 1, for adaptive gbn / rgbn:
    image -> resampled image; points -> blurred histogram (`points_to_density`).
    """
    t = as_target(target, 2)
    if t.kind == "image":
        return load_image(t.path, shape, invert)
    return points_to_density(t.points, shape)


__all__ = ["Target", "for_nufft", "for_cstit", "for_adaptive"]
