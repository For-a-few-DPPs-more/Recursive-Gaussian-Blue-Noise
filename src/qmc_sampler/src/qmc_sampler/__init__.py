MESSAGE = """
qmc_sampler: lightweight wrapper around the blue_sampler python package for generating QMC 
point set, see
     https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/src/qmc_sampler (sampler)
     https://github.com/For-a-few-DPPs-more/hyperuniform-samplers (datasets)

Available methods:
- sample_l2: uses a classical isotropic L2-ball as the spectral exclusion region.
- sample_l1: uses an L1-ball, which is anisotropic and can be useful for QMC
  in settings with low effective dimensionality.
- sample_wd: uses the wrap-around discrepancy, further exploiting the
  low-effective-dimensionality assumption.

For faster L2 sampling in 2D or 3D, a FINUFFT backend with O(N log N) complexity is available (instead of O(N^2)). 
It requires GPU support and additional
dependencies:
    pip install "blue_sampler[gpu3]"

Once installed, you will have to overwright the default sample_l2 implementation with this new version, which will use FINUFFT:

    def sample_l2(N, D, **kwargs):
        from blue_sampler.run.run_nufft import _nufft_pipeline
        kwargs.setdefault("verbose", 0)
        return _nufft_pipeline(N, D, **kwargs)
"""

print(MESSAGE)


from .qmc_sampler import sample_l1, sample_wd

def sample_l2(N, D, **kwargs):
    from blue_sampler.run.run_nufft_keops import _nufft_pipeline_keops
    kwargs.setdefault("verbose", 0)
    return _nufft_pipeline_keops(N, D, **kwargs)

def plot(*args, **kwargs):
    from blue_sampler import plot as _plot
    return _plot(*args, **kwargs)

def plot_structure_factor(*args, **kwargs):
    from blue_sampler import plot_structure_factor as _psf
    return _psf(*args, **kwargs)

__all__ = ["sample_l1", "sample_l2", "sample_wd", "plot", "plot_structure_factor"]
__version__ = "1.0.4"