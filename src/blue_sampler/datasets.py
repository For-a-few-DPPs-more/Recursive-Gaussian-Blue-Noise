"""Example point clouds from the `squarenet` package (optional dependency)."""
from __future__ import annotations


def generate_dataset(size=(1000, 2), dataset: str = "barbara", plot_data: bool = True, list_method: bool = True):
    """Points of a squarenet dataset, rescaled to [0, 1]."""
    from squarenet.sampler import list_methods, samplepoints

    if list_method:
        print("source: squarenet.sampler from module squarenet")
        print("available datasets:", list_methods())
    points = samplepoints(method=dataset, size=size, plot_points=plot_data)
    points -= points.min()
    return points / points.max()
