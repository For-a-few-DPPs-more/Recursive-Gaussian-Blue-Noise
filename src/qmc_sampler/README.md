# qmc-sampler

**Quasi-Monte Carlo point sampling from spectral optimisation.**

`qmc-sampler` is a lightweight, standalone package (numpy, scipy, matplotlib, pykeops; optionally torch or finufft) that provides several sampling methods for generating quasi-Monte Carlo (QMC) point sets, by minimising the spectral loss over all wave vectors within an exclusion region. 3 kinds of exclusion regions, with different geometrical shape, are available: l2, l1, wd.

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

## Backends (KeOps, torch, FINUFFT)

All heavy computations go through one of three backends, which the user can controll.
e.g., to use the finufft backend in 2D and 3D:

```bash
pip install finufft          # or: pip install "qmc-sampler[finufft]"
```

```python
import qmc_sampler

qmc_sampler.select_backend(dimensions={2, 3}, backend="finufft")
```

| backend     | method                           | complexity          | dimensions | accuracy                                   |
|-------------|----------------------------------|---------------------|------------|--------------------------------------------|
| `"keops"`   | direct sums, pykeops (GPU if available) | O(N M)       | any        | exact up to machine precision              |
| `"torch"`   | direct sums, torch (cuda if available)  | O(N M)       | any        | exact up to machine precision              |
| `"finufft"` | non-uniform FFT                  | O(N log N + M)      | D <= 3     | relative tolerance 1e-4 (float32), 1e-8 (float64) |

Here `M` is the number of wave vectors, which grows like `N`.

**By default, FINUFFT is not enabled**: the backend is `"keops"` if pykeops is available, otherwise
`"torch"`, and `"finufft"` only if neither is installed. This guarantees that, unless you ask
otherwise, every Fourier sum is computed exactly, up to machine precision.

**FINUFFT is nevertheless strongly recommended in 2D and 3D**: it reduces the cost of each
iteration from O(N^2) to O(N log N), which typically makes sampling one to several orders of
magnitude faster, and makes large `N` reachable. Enable it manually:

## Plotting utilities

The package also exposes plotting functions (adapted from `blue_sampler`):

```python
from qmc_sampler import plot, plot_structure_factor

plot(...)
plot_structure_factor(...)
```

Like the samplers, `plot_structure_factor` uses the backend selected for the dimension of the points (see above).

## Links

- **Source code:** [hyperuniform-samplers on GitHub](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers)
- **Underlying package:** [blue_sampler on PyPI](https://pypi.org/project/blue-sampler/)
- **Upstream repository:** [Recursive Gaussian Blue Noise](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise)
- **Documentation:** [blue-sampler.readthedocs.io](https://blue-sampler.readthedocs.io)

## License

MIT
