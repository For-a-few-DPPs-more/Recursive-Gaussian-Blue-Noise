"""
Public API.

Points      sample_points, image_to_points
"""
from __future__ import annotations

from typing import Literal
from numpy.typing import NDArray

from . import samplers
from .viz import plot
from .warmstart import WarmstartMethod

Method = Literal["rgbn", "gaussian", "nufft", "nufft+", "latjit", "cstit"]

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
    warmstart   "Goodlattice", "Sobol" or an (N, D) array, see `warmstart_points`
    n_iter_scale  multiplier of the number of iterations (0: return the warm start)
    target      non-uniform density: an image path (D = 2, dark = dense) or a point set (K, D), any
                bounding box: moved into the unit cube if needed, the samples come out in its frame;
                rgbn / gaussian: adaptive gaussian blue noise (D = 2; points become a blurred pseudo image),
                nufft / nufft+: matches the target spectrum, cstit: clusters the target atoms
                (an image is sampled i.i.d., points must number a multiple of the cells)
    verbose     0: silent, 1: progress

    >>> x = blue.sample_points(N=10_000)
    """
    return samplers.sample(N, D, lr, method, warmstart, n_iter_scale, target, verbose)


def image_to_points(image: str | NDArray, N: int = 2 ** 14, fast: bool = False) -> NDArray:
    """
    Image stippling: N points, dense where the image is dark, plotted and returned (N, 2).
    image: path, or a 2D point set (K, 2) used as a target (see `blue_sampler.targets`).

    Default: adaptive Gaussian blue noise (`blue_sampler.adaptive`); fast: spectral solver (nufft).
    """
    if fast:
        points = samplers.fast_stippling(image, N)
    else:
        from .adaptive import adaptive_rgbn
        points = adaptive_rgbn(image, N)
    plot(points, figsize=(8, 8))
    return points