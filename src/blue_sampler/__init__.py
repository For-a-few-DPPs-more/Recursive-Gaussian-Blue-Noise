"""
blue_sampler: blue noise (hyperuniform point patterns) on the unit torus [0, 1)^D,
uniform or following a target density.

>>> import blue_sampler as blue
>>> x = blue.sample_points(N=10_000, D=2)          # (10000, 2)
>>> blue.plot(x)
>>> blue.plot_structure_factor(x)
>>> x = blue.image_to_points("zebra.jpg")          # stippling

Layout
------
api               sample_points, image_to_points: checks the arguments, picks the solver
gbn               gaussian blue noise, all pairs (exact, O(N^2))
rgbn              recursive gaussian blue noise (coarse to fine, neighbours only)
cstit             fair clustering + moment matching
adaptive          gaussian blue noise following a density (stippling, 2D)
spectral          spectral solver ("nufft", "nufft+", fast stippling), delegated to qmc_sampler
targets           target densities: image path or point set
moments           moment matching: cells -> a few representative points
warmstart         Sobol, and the lattices of structure_factor_v2.lattices
torus             wrap, integer balls of Z^D
config            tunable constants, editable in place
backend           CuPy / KeOps detection, check_gpu
progress          console output
extras/           tessellations, clusters, pinwheel tilings and their plots (import separately)
"""
from structure_factor_v2 import (
    plot,
    plot_structure_factor,
    structure_factor,
    structure_factor_and_average,
)

from . import config
from .api import image_to_points, sample_points
from .backend import check_gpu

__all__ = [
    "sample_points", "image_to_points",
    "structure_factor", "structure_factor_and_average",
    "plot", "plot_structure_factor",
    "check_gpu", "config",
]

__version__ = "5.0.1"
