"""
Pairwise interaction forces on the unit torus.

Each force takes the raw displacement ``delta = x_i - x_j`` and returns the
repulsive force exerted by j on i (same shape, to be summed over j). The `ops`
argument selects the array library (jax by default, or KeOps LazyTensors).
"""
from __future__ import annotations

from math import pi

from .array_lib import JAX, Ops, min_image


def gaussian_force(delta, sigma2: float, ops: Ops = JAX):
    """delta * exp(-|delta|^2 / (2 sigma2)), minimum-image distances."""
    delta = min_image(delta, ops)
    return delta * ops.exp(-ops.sum(delta ** 2) / (2.0 * sigma2))


def periodic_gaussian_force(delta, sigma2: float, ops: Ops = JAX):
    """Smooth periodic version of `gaussian_force` (|delta| -> |sin(pi delta)| / pi).
    Preferred when sigma is not small, where the minimum image becomes discontinuous."""
    s2 = ops.sum(ops.sin(pi * delta) ** 2) / (2.0 * pi ** 2 * sigma2)
    return ops.sin(2.0 * pi * delta) * ops.exp(-s2) / (2.0 * pi)


def pick_force(sigma2: float, periodic_min_sigma2: float):
    return periodic_gaussian_force if sigma2 >= periodic_min_sigma2 else gaussian_force
