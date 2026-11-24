# Documentation

```{include} ../../README.md
```

## Main API

Everything below is available directly from `import blue_sampler as blue`.

### Sampling

```{eval-rst}
.. autosummary::
   :toctree: generated/

   blue_sampler.sample_points
   blue_sampler.image_to_points
```

### Analysis

```{eval-rst}
.. autosummary::
   :toctree: generated/

   blue_sampler.structure_factor
   blue_sampler.structure_factor_and_average
```

### Visualization

```{eval-rst}
.. autosummary::
   :toctree: generated/

   blue_sampler.plot
   blue_sampler.plot_structure_factor
```

### Diagnostics

```{eval-rst}
.. autosummary::
   :toctree: generated/

   blue_sampler.check_gpu
```

### Configuration

Every tunable constant lives in `blue_sampler.config`, as dataclass instances that can be
edited in place before sampling:

```python
import blue_sampler as blue
blue.config.RGBN.presets[2].lr_spatial = 0.2
blue.config.ADAPTIVE.grid_size = 1024
```

| Object                     | Used by                                                  |
|----------------------------|----------------------------------------------------------|
| `config.SAMPLE_POINTS`     | routing rules of `sample_points` (exact solver, KeOps)   |
| `config.BRUTEFORCE`        | `gaussian` (exact O(N²) gaussian blue noise)             |
| `config.RGBN`              | `rgbn` (per-dimension presets, neighbour search)         |
| `config.ADAPTIVE`          | adaptive samplers, `image_to_points`                     |
| `config.SPECTRAL`          | `nufft`, `nufft+`, `image_to_points(fast=True)`          |
| `config.CSTIT`             | `cstit`                                                  |
| `config.TESSELS`           | `extras.sample_tessels`                                  |
| `config.MOMENTS`           | moment matching (`cstit`, `*_to_points`, pinwheel)       |
| `config.TARGETS`           | images and point sets read as densities                  |

See {doc}`blue_sampler` for every field.

## Adaptive sampling (image stippling)

Adaptive Gaussian blue noise, 2D only, kept apart from the rest of the package
(`from blue_sampler.adaptive import ...`). It is what `image_to_points` and
`sample_points(target=..., method="rgbn" | "gaussian")` use.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   blue_sampler.adaptive.adaptive_rgbn
   blue_sampler.adaptive.adaptive_refine
   blue_sampler.adaptive.adaptive_gbn
   blue_sampler.adaptive.Density
```

## Targets

Non-uniform densities given to `sample_points(target=...)`: an image path (2D) or a
point set `(K, D)`.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   blue_sampler.targets.Target
   blue_sampler.targets.as_target
```

## Extras

Tessellations, clusters, pinwheels and their plots. Not imported by default:
`from blue_sampler import extras`.

### Points

```{eval-rst}
.. autosummary::
   :toctree: generated/

   blue_sampler.extras.tile
   blue_sampler.extras.warmstart_points
   blue_sampler.extras.sobol
```

### Partitions

```{eval-rst}
.. autosummary::
   :toctree: generated/

   blue_sampler.extras.sample_tessels
   blue_sampler.extras.sample_clusters
   blue_sampler.extras.tessels_to_points
   blue_sampler.extras.clusters_to_points
   blue_sampler.extras.image_to_quads
```

### Pinwheel

```{eval-rst}
.. autosummary::
   :toctree: generated/

   blue_sampler.extras.pinwheel_base
   blue_sampler.extras.pinwheel_transform
   blue_sampler.extras.pinwheel_points
```

### Plots

```{eval-rst}
.. autosummary::
   :toctree: generated/

   blue_sampler.extras.plot_tessels
   blue_sampler.extras.plot_clusters
   blue_sampler.extras.plot_polygons
```

```{toctree}
:hidden:

modules
```
