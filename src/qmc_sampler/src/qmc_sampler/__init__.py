MESSAGE = """
qmc_sampler: QMC point sets by spectral optimisation, see
     https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/src/qmc_sampler (sampler)
     https://github.com/For-a-few-DPPs-more/hyperuniform-samplers (datasets)

Available methods:
- sample_l2: uses a classical isotropic L2-ball as the spectral exclusion region.
- sample_l1: uses an L1-ball, which is anisotropic and can be useful for QMC
  in settings with low effective dimensionality.
- sample_wd: uses the wrap-around discrepancy, further exploiting the
  low-effective-dimensionality assumption.

Backends: keops (default if available), torch, finufft (D <= 3, O(N log N) instead of O(N^2)).
finufft is not enabled by default (exact sums up to machine precision), but is strongly
recommended in 2D / 3D:
    pip install finufft
    qmc_sampler.select_backend(dimensions={2, 3}, backend="finufft")
"""

print(MESSAGE)


from ._backend import available_backends, get_backend, select_backend
from ._viz import plot, plot_structure_factor
from .qmc_sampler import sample_l1, sample_l2, sample_wd

__all__ = [
    "sample_l1",
    "sample_l2",
    "sample_wd",
    "plot",
    "plot_structure_factor",
    "select_backend",
    "get_backend",
    "available_backends",
]
__version__ = "3.0.0"
