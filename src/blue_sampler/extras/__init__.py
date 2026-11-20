"""
Extras: tessellations, clusters, pinwheels and their plots.

Not imported by `blue_sampler` itself; import it explicitly.

>>> from blue_sampler import extras
>>> t = extras.sample_tessels(N=2 ** 12)
>>> extras.plot_tessels(t)
>>> x = extras.tessels_to_points(t)

Points      tile, warmstart_points, sobol
Partitions  sample_tessels, sample_clusters, tessels_to_points, clusters_to_points, image_to_quads
Pinwheel    pinwheel_base, pinwheel_transform, pinwheel_points
Plots       plot_tessels, plot_clusters, plot_polygons
"""
from .extras import (
    WarmstartMethod,
    clusters_to_points,
    image_to_quads,
    pinwheel_base,
    pinwheel_transform,
    sample_clusters,
    sample_tessels,
    sobol,
    tessels_to_points,
    tile,
    warmstart_points,
)
from .pinwheel import pinwheel_points
from .viz import plot_clusters, plot_polygons, plot_tessels

__all__ = [
    "tile", "warmstart_points", "sobol", "WarmstartMethod",
    "sample_tessels", "sample_clusters", "tessels_to_points", "clusters_to_points", "image_to_quads",
    "pinwheel_base", "pinwheel_transform", "pinwheel_points",
    "plot_tessels", "plot_clusters", "plot_polygons",
]
