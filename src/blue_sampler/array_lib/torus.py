"""Geometry of the unit torus [0, 1)^D, for any array type (numpy, cupy, jax, KeOps)."""
from .ops import JAX, Ops


def wrap(x):
    """Map coordinates back into [0, 1)^D."""
    return x - x // 1


def min_image(delta, ops: Ops = JAX):
    """Shortest signed displacement on the torus (minimum image convention)."""
    return delta - ops.round(delta)
