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
utils             helpers of api (checks, routing to the solvers)
config            every tunable constant
solvers/          bruteforce, rgbn, nufft, tessels, clusters (cstit)
adaptive/         adaptive gaussian blue noise (image stippling), 2D jax, standalone
moments/          moment matching (cells -> representative points)
lattice/          integer balls, Fourier modes, Korobov lattices
array_lib/        numpy / cupy / jax / KeOps primitives
progress          console output
"""
from .api import (
    image_to_points,
    image_to_quads,
    tessels_to_points,
    clusters_to_points,
    sample_points,
    sample_tessels,
    sample_clusters,
    sobol,
    pinwheel_base,
    pinwheel_transform,
    tile,
    warmstart_points,
)
from .viz import plot, plot_structure_factor, plot_tessels, plot_clusters, plot_polygons
from .structure_factor import structure_factor, structure_factor_and_average
from .datasets import generate_dataset
from .diagnostics import check_gpu

__all__ = [
    "image_to_points", "image_to_quads", "tessels_to_points", "clusters_to_points",
    "sample_points", "sample_tessels", "sample_clusters", "sobol",
    "pinwheel_base", "pinwheel_transform", "tile", "warmstart_points",
    "structure_factor", "structure_factor_and_average", "generate_dataset",
    "plot", "plot_structure_factor", "plot_tessels", "plot_clusters", "plot_polygons",
    "check_gpu",
]

__version__ = "4.0.0"
