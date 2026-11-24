"""
All Tunable constants of the package. 
If the provided defaults feel bad, any parameter can be tuned:

>>> import blue_sampler as blue
>>> blue.config.RGBN.presets[2].lr_spatial = 0.2
"""
from __future__ import annotations

from dataclasses import dataclass, field


def by_dim(table: dict, D: int):
    """table[D], D clamped to the range of dimensions present in the table."""
    return table[min(max(D, min(table)), max(table))]


@dataclass
class SamplePointsConfig:
    """Routing rules of `sample_points`."""
    bruteforce_max_n: int = 1_000          # N <= this: exact gaussian solver, whatever the method
    keops_min_n: int = 5_000               # KeOps kernels are only worth it above this N
    target_iter_factor: int = 2            # a target (nufft) needs more iterations


@dataclass
class BruteforceConfig:
    """Exact O(N^2) gaussian blue noise."""
    learning_rate: dict = field(default_factory=lambda: {2: 0.3, 3: 0.1, 4: 0.05, 5: 0.01})
    learning_rate_high_d: float = 0.001    # D > 5
    sigma_scale: float = 1.0               # kernel width, in mean spacings N^(-1/D)
    periodic_kernel_min_sigma: float = 0.1 # wider kernels use the smooth periodic formula
    min_iter: int = 20
    iter_divisor: dict = field(default_factory=lambda: {2: 4, 3: 100})  # n_iter = max(min_iter, N // divisor)


@dataclass
class RGBNPreset:
    spatial_radius: float      # neighbourhood of the short-range kernel, in grid cells
    spectral_radius: float     # radius of the wave vectors of the long-range term
    lr_spatial: float
    lr_spectral: float
    expansion_factor: float    # size of the simplex a parent is cloned on, in mean spacings
    sigma_scale: float         # kernel width: sigma^2 = 2 * sigma_scale * spacing^2


@dataclass
class RGBNConfig:
    """Recursive Gaussian Blue Noise."""
    presets: dict = field(default_factory=lambda: {
        2: RGBNPreset(7, 7, 0.100, 0.1, 0.3, 1.0),
        3: RGBNPreset(5, 5, 0.030, 0.1, 0.3, 1.0),
        4: RGBNPreset(4, 4, 0.010, 0.1, 1.0, 0.5),
        5: RGBNPreset(3, 3, 0.003, 0.1, 1.0, 0.5),
    })
    bruteforce_max_n: dict = field(default_factory=lambda: {2: 1_000, 3: 2_000, 4: 3_000, 5: 4_000})
    root_max_n: int = 3_000                # a level this small is not split further
    root_n_iter: int = 20                  # neighbour updates of a root level
    steps_per_iter: int = 10               # descent steps between two neighbour updates
    grid_max_dim: int = 4                  # grid neighbours up to this D, Verlet lists above
    periodic_kernel_min_sigma2: float = 0.03
    verlet_skin_factor: int = 3            # skin size = verlet_skin_factor x neighbour list size
    verlet_chunk_elems: int = 2 ** 25      # memory budget (floats) of the neighbour searches


@dataclass
class AdaptiveConfig:
    """Adaptive gaussian blue noise (stippling). Defaults from stipple, the reference."""
    grid_size: int = 512                   # side of the density grid
    levels: int = 14                       # pyramid of blurred gradients: one octave of variance per level,
                                           # from half a pixel of standard deviation up to ~45 pixels
    lr: float = 0.3
    max_move: float = 0.3                  # longest move of a dot in one step, in local spacings
    cutoff: float = 30.0                   # pair kernel truncated at exp(-cutoff)
    a_min: float = 1 / 32                  # kernels stop growing below this density (x the mean)
    steps_per_iter: int = 10               # descent steps between two re-sorts of the grid
    root_max_n: int = 3_000                # size of the coarsest level
    root_n_iter: int = 20                  # passes of the coarsest level (and of a warm start)
    iter_factor: int = 2                   # passes per level = iter_factor * n_iter_scale
    gbn_iter_factor: int = 7               # adaptive_gbn: passes = gbn_iter_factor * n_iter_scale
    expansion_factor: float = 0.3          # size of the cloning triangle, in local spacings
    stencil: int = 7                       # grid neighbourhood of the pair sum (~6 % density error
                                           # where pale areas meet dark ones)
    refine_stencil: int = 10               # adaptive_refine: ~1 % error, ~2x the cost per step


@dataclass
class SpectralConfig:
    """Spectral blue noise, delegated to qmc_sampler.sample_l2 (`blue_sampler.spectral`)."""
    chi: dict = field(default_factory=lambda: {"nufft": 0.35, "nufft+": 0.43})
    lbfgs_finish: dict = field(default_factory=lambda: {"nufft": False, "nufft+": True})
    iter_factor: dict = field(default_factory=lambda: {"nufft": 20, "nufft+": 80})
    iter_factor_high_d: int = 10           # D > 3 converges faster
    max_dim: int = 3                       # D <= max_dim uses iter_factor, above iter_factor_high_d
    image_oversample: int = 32             # an image target is sampled i.i.d. at this many points per output point
    fast_stippling: dict = field(default_factory=lambda: {"chi": 1.0, "n_iter": 40, "lr": 3.0})


@dataclass
class CstitConfig:
    """Fair clustering + moment matching."""
    iter_factor: int = 4
    atoms_per_cluster: int = 32
    moment_order: int = 3
    lr_to_lambda: float = 0.01             # Levenberg-Marquardt damping = lr * lr_to_lambda


@dataclass
class TesselsConfig:
    """Fair quadrilateral splits."""
    pool: int = 5                          # random directions tried at once per split
    max_retries: int = 8
    pool_growth: int = 2
    eps: float = 1e-9


@dataclass
class MomentsConfig:
    """Levenberg-Marquardt moment matching."""
    max_p: int = 100
    lambda0: float = 1e-2
    lambda_increase: float = 3.0
    lambda_decrease: float = 0.5
    min_lambda_rel: float = 1e-7           # lambda >= min_lambda_rel * trace(J^T J) / dim
    tol: float = 1e-30


@dataclass
class TargetsConfig:
    """Images and point sets read as densities (`blue_sampler.targets`)."""
    image_shape: tuple = (512, 512)
    image_blur_sigma: float = 1.0          # pixels, before i.i.d. sampling of an image
    image_jitter: float = 1 / 2000
    cstit_oversample: int = 32             # i.i.d. atoms per output point when cstit reads an image
    points_blur_spacing: float = 1.0       # pseudo image of a point set: blur in mean spacings...
    points_blur_min: float = 1.0           # ... and at least this many pixels
    points_margin: float = 0.05            # point sets moved into the unit cube keep this margin empty
                                           # (the cube is a torus: the support must not touch itself

SAMPLE_POINTS = SamplePointsConfig()
BRUTEFORCE = BruteforceConfig()
RGBN = RGBNConfig()
ADAPTIVE = AdaptiveConfig()
SPECTRAL = SpectralConfig()
CSTIT = CstitConfig()
TESSELS = TesselsConfig()
MOMENTS = MomentsConfig()
TARGETS = TargetsConfig()