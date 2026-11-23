"""
Extras: fair partitions (tessellations, clusters), pinwheel tilings, and their plots.
Not imported by `blue_sampler` itself:

>>> import blue_sampler.extras as bluex
>>> t = bluex.sample_tessels(N=2 ** 12)
>>> bluex.plot_tessels(t)
>>> x = bluex.tessels_to_points(t)

Points      tile, warmstart_points, sobol
Partitions  sample_tessels, sample_clusters, tessels_to_points, clusters_to_points, image_to_quads
Pinwheel    pinwheel_base, pinwheel_transform, pinwheel_points
Plots       plot_tessels, plot_clusters, plot_polygons
"""
from ..warmstart import WarmstartMethod, sobol, warmstart_points
from .pinwheel import pinwheel_base, pinwheel_points, pinwheel_transform
from .sampling import (
    clusters_to_points,
    image_to_quads,
    sample_clusters,
    sample_tessels,
    tessels_to_points,
    tile,
)
from .viz import plot_clusters, plot_polygons, plot_tessels

__all__ = [
    "tile", "warmstart_points", "sobol", "WarmstartMethod",
    "sample_tessels", "sample_clusters", "tessels_to_points", "clusters_to_points", "image_to_quads",
    "pinwheel_base", "pinwheel_transform", "pinwheel_points",
    "plot_tessels", "plot_clusters", "plot_polygons",
]
