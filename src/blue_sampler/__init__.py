"""
blue_sampler
============

Stealthy point patterns: low-discrepancy, spectrally isotropic samples on the
unit torus [0, 1)^D.

Quick start
-----------
>>> import blue_sampler as blue
>>> x = blue.sample_points(N=10_000, D=2)   # (10000, 2) array
>>> blue.plot(x)
>>> blue.plot_structure_factor(x)

Package layout
--------------
api               public sampling functions
samplers          routing from api to the solvers for sampling
config            every tunable constant
structure_factor  structure factor estimation (trusted or tapered boundary)
viz               plot, plot_structure_factor
diagnostics       check_gpu: optional GPU stack
warmstart         initial point clouds and cheap low-discrepancy samplers
targets/          target densities (image path or point set) for every solver
solvers/          bruteforce, rgbn, nufft, tessels, clusters (cstit), kernels, geometry
adaptive/         adaptive gaussian blue noise (image stippling), 2D jax, standalone
moments/          moment matching (cells -> representative points)
lattice/          integer balls, Fourier modes, Korobov lattices
array_lib/        numpy / cupy / jax / KeOps primitives
progress          console output

Extras
------
Tessellations, clusters, pinwheels and their plots, not imported by default:
>>> from blue_sampler import extras
"""
from .api import sample_points, image_to_points
from .viz import plot, plot_structure_factor
from .structure_factor import structure_factor, structure_factor_and_average
from .diagnostics import check_gpu

__all__ = [
    "image_to_points", "sample_points",
    "structure_factor", "structure_factor_and_average", 
    "plot", "plot_structure_factor", 
    "check_gpu",
]

__version__ = "4.0.0"
