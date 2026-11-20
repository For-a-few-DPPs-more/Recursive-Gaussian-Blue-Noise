"""
Targets: non-uniform densities given to `sample_points(target=...)` and to the solvers.

A target is either
- an image path (2D): dark pixels = dense (``invert=True``), or
- a point set (K, D), read as samples of the density. Outside [0, 1]^D it is moved into the
  unit cube (uniform scaling, margin); solvers map their output back with `Target.from_unit`,
  so the samples come out in the frame of the target.

`as_target` validates the input once; each family of solvers then reads it in its own form:

    solver               image                               point set
    for_nufft            image Fourier coefficients          NUFFT of the points
    for_cstit            oversampled i.i.d. atoms            the points (count checked)
    for_adaptive         density grid (native)               blurred histogram (pseudo image)

Layout
------
base       Target, as_target: classification, checks, normalisation to the unit cube
image      images as densities: load_image, image_fourier, sample_image, periodic blur
points     point sets as densities: check_points, points_to_density
convert    for_nufft, for_cstit, for_adaptive
"""
from .base import Target, as_target
from .convert import for_adaptive, for_cstit, for_nufft
from .image import blur, image_fourier, load_image, sample_image
from .points import check_points, points_to_density

__all__ = [
    "Target", "as_target",
    "for_nufft", "for_cstit", "for_adaptive",
    "load_image", "image_fourier", "sample_image", "blur",
    "check_points", "points_to_density",
]
