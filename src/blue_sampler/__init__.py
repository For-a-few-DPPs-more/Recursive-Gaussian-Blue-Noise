"""
blue_sampler
============

Generate stealthy point patterns — low-discrepancy, spectrally isotropic
samples on the unit torus [0, 1)^D.

Quick start
-----------
>>> import blue_sampler as blue
>>> x = blue.sample_points(N=10_000, D=2)   # (10000, 2) array
>>> blue.plot(x)
>>> blue.plot_structure_factor(x)
"""
from .api import (
    im2points,
    im2quads,
    tessel2points,
    cluster2points,
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
from .structurefactor import structure_factor, structure_factor_and_average
from .datasets import generate_dataset
from .gpu_setup import check_gpu

__all__ = [
    "im2points",
    "im2quads",
    "tessel2points",
    "cluster2points",
    "sample_points",
    "sample_tessels",
    "sample_clusters",
    "sobol",
    "pinwheel_base",
    "pinwheel_transform",
    "structure_factor",
    "structure_factor_and_average",
    "generate_dataset",
    "plot",
    "plot_structure_factor",
    "plot_tessels",
    "plot_clusters",
    "plot_polygons",
    "tile",
    "warmstart_points",
    "check_gpu",
]

__version__ = "2.1.4"
