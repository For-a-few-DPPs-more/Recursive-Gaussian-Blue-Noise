# qmc-sampler

**Quasi-Monte Carlo point sampling from spectral optimisation.**

`qmc-sampler` is a lightweight wrapper around [`blue_sampler`](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers) that provides several sampling methods for generating quasi-Monte Carlo (QMC) point sets, by minimising the spectral loss over all wave vectors within an exclusion region. 3 kinds of exclusion regions, with different geometrical shape, are available: l2, l1, wd.

## Installation

Install from PyPI:

```bash
pip install qmc-sampler
```

## Available sampling methods

The package provides three sampling methods:

- **`sample_l2(N, D, **kwargs)`** — uses a classical isotropic L_2-ball as the spectral exclusion region.
- **`sample_l1(N, D, **kwargs)`** — uses an L_1-ball, producing an anisotropic spectral exclusion region that can be useful when exploiting low effective dimensionality.
- **`sample_wd(N, D, **kwargs)`** — uses the wrap-around discrepancy (WD), a criterion designed to further exploit low-effective-dimensionality structure.

Here, `N` denotes the number of points and `D` the dimension.

## Basic usage

```python
from qmc_sampler import sample_l1, sample_l2, sample_wd

N = 1000
D = 2

points_l2 = sample_l2(N, D)
points_l1 = sample_l1(N, D)
points_wd = sample_wd(N, D)

```

Additional keyword arguments e.g. "Chi = 0.4" can be passed to the sampling functions and forwarded to the underlying implementation.

## Accelerated L2 sampling with FINUFFT

For supported 2D and 3D problems, `blue_sampler` provides an accelerated NUFFT-based implementation that can reduce the computational complexity from approximately O(N^2) to O(N\log N).

This backend requires a compatible GPU environment and additional dependencies. Install the appropriate extras supported by your version of `blue_sampler`, for example:

```bash
pip install "blue-sampler[gpu13]"
```

To use the accelerated backend, overwright the default `sample_l2` implementation with:

```python
def sample_l2(N, D, **kwargs):
    from blue_sampler.run.run_nufft import _nufft_pipeline

    kwargs.setdefault("verbose", 0)
    return _nufft_pipeline(N, D, **kwargs)
```

## Plotting utilities

The package also exposes plotting functions from `blue_sampler`:

```python
from qmc_sampler import plot, plot_structure_factor

plot(...)
plot_structure_factor(...)
```

Refer to the upstream documentation for the accepted arguments.

## Links

- **Source code:** [hyperuniform-samplers on GitHub](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers)
- **Underlying package:** [blue_sampler on PyPI](https://pypi.org/project/blue-sampler/)
- **Upstream repository:** [Recursive Gaussian Blue Noise](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise)
- **Documentation:** [blue-sampler.readthedocs.io](https://blue-sampler.readthedocs.io)

## License

MIT
