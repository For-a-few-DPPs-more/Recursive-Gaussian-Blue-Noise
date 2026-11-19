"""
Every tunable constant of the package, grouped by component.

The values are module-level frozen dataclasses, e.g. ``config.RGBN.root_n_iter``.
Per-dimension tables are indexed by D and clamped to their range by ``by_dim``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def by_dim(table: dict, D: int):
    """table[D], with D clamped to the dimensions present in the table."""
    return table[min(max(D, min(table)), max(table))]


@dataclass(frozen=True)
class SamplePointsConfig:
    """Routing rules of `sample_points`."""
    bruteforce_max_n: int = 1_000          # N <= this: exact gaussian solver, whatever the method
    keops_min_n: int = 5_000               # KeOps kernels are only worth it above this N
    target_iter_factor: int = 2            # a target (nufft, cstit) needs more iterations
    image_oversample: dict = field(default_factory=lambda: {"cstit": 32})


@dataclass(frozen=True)
class BruteforceConfig:
    """Exact O(N^2) gaussian blue noise."""
    learning_rate: dict = field(default_factory=lambda: {2: 0.3, 3: 0.1, 4: 0.05, 5: 0.01})
    learning_rate_high_d: float = 0.001    # D > 5
    sigma_scale: float = 1.0               # kernel width, in units of the mean spacing N^(-1/D)
    periodic_kernel_min_sigma: float = 0.1 # wide kernels use the smooth periodic formulation
    min_iter: int = 20
    iter_divisor: dict = field(default_factory=lambda: {2: 4, 3: 100})  # n_iter = max(min_iter, N // divisor)


@dataclass(frozen=True)
class RGBNPresetConfig:
    spatial_radius: float      # neighbourhood (in grid cells) of the short-range kernel
    spectral_radius: float     # radius of the wave vectors of the long-range kernel
    lr_spatial: float
    lr_spectral: float
    expansion_factor: float    # size of the simplex used to clone a parent into children
    sigma_scale: float         # kernel width, sigma^2 = 2 * sigma_scale * spacing^2


@dataclass(frozen=True)
class RGBNConfig:
    """Recursive Gaussian Blue Noise."""
    presets: dict = field(default_factory=lambda: {
        2: RGBNPresetConfig(7, 7, 0.100, 0.1, 0.3, 1.0),
        3: RGBNPresetConfig(5, 5, 0.030, 0.1, 0.3, 1.0),
        4: RGBNPresetConfig(4, 4, 0.010, 0.1, 1.0, 0.5),
        5: RGBNPresetConfig(3, 3, 0.003, 0.1, 1.0, 0.5),
    })
    bruteforce_max_n: dict = field(default_factory=lambda: {2: 1_000, 3: 2_000, 4: 3_000, 5: 4_000})
    root_max_n: int = 3_000                # below this a level is solved directly (no recursion)
    root_n_iter: int = 20                  # iterations of a root level
    steps_per_iter: int = 10               # descent steps between two neighbour updates
    grid_max_dim: int = 4                  # grid neighbours up to this D, Verlet lists above
    periodic_kernel_min_sigma2: float = 0.03
    verlet_skin_factor: int = 3            # skin = skin_factor x neighbour list size
    verlet_chunk_elems: int = 2 ** 25      # memory budget (floats) of the neighbour searches


@dataclass(frozen=True)
class NufftConfig:
    """Spectral optimisation of the structure factor."""
    chi: dict = field(default_factory=lambda: {"nufft": 0.35, "nufft+": 0.43})
    newton_finition: dict = field(default_factory= lambda: {"nufft": False, "nufft+": True})
    iter_factor: dict = field(default_factory=lambda: {"nufft": 20, "nufft+": 80})
    iter_factor_high_d: int = 10           # D > 3 converges faster
    max_dim: int = 3                       # finufft supports D <= 3; brute-force kernels above
    eps: dict = field(default_factory=lambda: {"float32": 1e-4, "float64": 1e-8})
    initial_step: float = 0.1              # in units of the mean spacing
    step_increase: float = 1.05            # "bold driver" adaptive step
    step_decrease: float = 0.8
    fast_stippling: dict = field(default_factory=lambda: {"chi": 1.0, "n_iter": 40, "lr": 3.0})

@dataclass(frozen=True)
class CstitConfig:
    """Fair STIT clustering + moment matching."""
    iter_factor: int = 4
    atoms_per_cluster: int = 32
    moment_order: int = 3
    lr_to_lambda: float = 0.01             # Levenberg-Marquardt damping = lr * lr_to_lambda


@dataclass(frozen=True)
class TesselsConfig:
    """Fair quadrilateral splits."""
    pool: int = 5                          # random directions tried per split
    max_retries: int = 8
    pool_growth: int = 2
    eps: float = 1e-9


@dataclass(frozen=True)
class MomentsConfig:
    """Levenberg-Marquardt moment matching."""
    max_p: int = 100
    lambda0: float = 1e-2
    lambda_increase: float = 3.0
    lambda_decrease: float = 0.5
    min_lambda_rel: float = 1e-7           # lambda >= min_lambda_rel * trace(J^T J) / dim
    tol: float = 1e-30


@dataclass(frozen=True)
class DensityConfig:
    """Target images."""
    image_shape: tuple = (512, 512)
    image_blur_sigma: float = 1.0          # pixels
    image_jitter: float = 1 / 2000


@dataclass(frozen=True)
class StructureFactorConfig:
    margin: float = 0.1
    max_beta: float = 15.0                 # Kaiser taper: leakage ~exp(-2 beta) ~ 1e-13
    k_max: float = 2.0                     # in units of the inter-particle wave number
    eps: dict = field(default_factory=lambda: {"float32": 1e-5, "float64": 1e-12})
    nufft_chunk: int = 400_000             # points per 3D NUFFT call
    bruteforce_chunk_elems: int = 8_000_000
    low_mode_fraction: float = 0.25        # D >= 4: share of exhaustive low modes
    radial_bins: dict = field(default_factory=lambda: {1: (10, 50), 2: (5, 25), 3: (2, 6)})  # (width, start) in |n|^2
    smoothing: float = 0.01                # gaussian smoothing width, fraction of the log-k range
    resolution_divisor: int = 10           # D >= 4: wave vectors evaluated = resolution / divisor


SAMPLE_POINTS = SamplePointsConfig()
BRUTEFORCE = BruteforceConfig()
RGBN = RGBNConfig()
NUFFT = NufftConfig()
CSTIT = CstitConfig()
TESSELS = TesselsConfig()
MOMENTS = MomentsConfig()
DENSITY = DensityConfig()
STRUCTURE_FACTOR = StructureFactorConfig()
