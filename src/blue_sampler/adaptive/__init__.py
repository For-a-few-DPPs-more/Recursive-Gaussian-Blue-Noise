"""
Adaptive Gaussian blue noise (image stippling), 2D only, jax.

Port of stipple (https://github.com/adecacqueray/stipple, src/rgbn.rs), which is the
reference for every parameter. Kept apart from the rest of the package on purpose.

>>> from blue_sampler.adaptive import adaptive_rgbn, adaptive_refine
>>> x = adaptive_rgbn("zebra.jpg", N=20_000)
>>> x = adaptive_refine("zebra.jpg", x)          # "optimise further"
"""
from .field import Density
from .rgbn import adaptive_gbn, adaptive_refine, adaptive_rgbn

__all__ = ["Density", "adaptive_gbn", "adaptive_rgbn", "adaptive_refine"]
