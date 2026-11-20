"""
Recursive Gaussian Blue Noise (rgbn).

A level with N points first solves a coarse problem with ~N / (k+1) points
(recursively), then clones every parent into k+1 children placed on a randomly
rotated simplex (k = min(D, 3)), and finally runs a gradient descent of
    E = short-range gaussian repulsion + spectral energy on the lowest wave vectors.
Small levels are solved directly,
by the exact bruteforce solver when N is small enough.

The short-range sum only visits neighbours, found in one of two ways
(module `neighbours`): a grid re-sorted by gridpoints (D <= 4), or Verlet lists.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .. import config
from ..array_lib import wrap
from ..config import RGBNPresetConfig, by_dim
from .geometry import random_rotations, simplex
from .kernels import gaussian_force, periodic_gaussian_force
from ..lattice import half_ball
from ..progress import Progress, log
from .bruteforce import bruteforce_sampler
from .neighbours import grid_descent, verlet_descent

CFG = config.RGBN


def spectral_gradient(spectral_radius: float, D: int):
    """Gradient of sum_k |S(k)|^2 / |k|^D over the half-ball of wave vectors (NaN rows ignored)."""
    import jax
    import jax.numpy as jnp

    K = 2.0 * jnp.pi * half_ball(int(spectral_radius ** 2), D) * 1j
    K_dual = -K / (jnp.abs(K) ** D).sum(axis=-1, keepdims=True)

    def one_mode(x, k, k_dual):
        e = jnp.nan_to_num(jnp.exp(jnp.sum(k * x, axis=-1, keepdims=True)), nan=0.0)
        return jnp.real(jnp.sum(e, axis=0, keepdims=True) * k_dual * jnp.conjugate(e))

    def grad(x):
        flat = x.reshape(-1, D)
        out, _ = jax.lax.scan(lambda acc, kk: (acc + one_mode(flat, *kk), None), jnp.zeros_like(flat), (K, K_dual))
        return out.reshape(x.shape)

    return grad


@dataclass(frozen=True)
class Energy:
    """Scales of one level: x <- x + lr_short * short_range(x - y) - lr_long * long_range_grad(x)."""
    sigma2: float
    lr_short: float
    lr_long: float
    short_range: callable       # pair force f(x_i - x_j), NaN-safe
    long_range_grad: callable   # x -> spectral gradient, same shape as x

    @classmethod
    def build(cls, N: int, D: int, preset: RGBNPresetConfig, lr: float):
        import jax.numpy as jnp

        n_axis = N ** (1.0 / D)                             # points per axis
        S = preset.sigma_scale
        sigma2 = S * 2.0 * (1.0 / n_axis) ** 2
        if sigma2 >= CFG.periodic_kernel_min_sigma2:
            short = lambda d: periodic_gaussian_force(d, sigma2 / 2) / S
        else:
            short = lambda d: jnp.nan_to_num(gaussian_force(d, sigma2 / 2), nan=0.0)
        lr_short = lr * (preset.lr_spatial / S)
        lr_long = lr * (preset.lr_spectral / (N ** 0.5 * n_axis))
        return cls(sigma2, lr_short, lr_long, short, spectral_gradient(preset.spectral_radius, D))


class _Recursion:
    def __init__(self, D, preset, lr, neighbours, verbose, width):
        self.D, self.preset, self.lr = D, preset, lr
        self.descent = {"grid": grid_descent, "verlet": verlet_descent}[neighbours]
        self.verbose, self.width = verbose, width
        self.n_children = min(D, 3) + 1                     # vertices of the cloning simplex

    def level(self, N: int, n_iter: int, x, depth: int) -> np.ndarray:
        D = self.D
        bruteforce = N <= by_dim(CFG.bruteforce_max_n, D)
        is_root = x is not None or bruteforce or N <= CFG.root_max_n
        if is_root:
            n_iter = CFG.root_n_iter
        progress = Progress(f"[level {depth} · N = {N:>{self.width}}]", total=n_iter, verbose=self.verbose)

        if is_root:
            x = np.random.rand(N, D) if x is None else np.asarray(x).reshape(N, D)
        else:
            k = self.n_children
            x = self.clone(self.level(N // k + N % k, n_iter, None, depth + 1), N)

        progress.start()
        if bruteforce:
            x = bruteforce_sampler(N, D, lr=self.lr, progress=progress)(x)
        else:
            energy = Energy.build(N, D, self.preset, self.lr)
            x = self.descent(x, n_iter, energy, self.preset.spatial_radius, progress)
        progress.done()
        return x

    def clone(self, parents: np.ndarray, N: int) -> np.ndarray:
        """N children from N // k parents on random simplices (+ N % k parents kept as is)."""
        D, k = self.D, self.n_children
        parents = np.random.permutation(parents)
        n_parents = N // k
        offsets = random_rotations(simplex(k - 1), n_parents, D) * (self.preset.expansion_factor / N ** (1.0 / D))
        children = (parents[:n_parents, None, :] + offsets).reshape(-1, D)
        children = np.concatenate([parents[n_parents:], children], axis=0)
        return wrap(children.astype(np.float32))


def rgbn(N: int, D: int, n_iter: int, *, lr: float = 1.0, x=None,
         neighbours: str | None = None, verbose: int = 1, **preset_overrides) -> np.ndarray:
    """
    Recursive gaussian blue noise: N points in [0, 1)^D.

    n_iter: neighbour updates per non-root level (root levels use config.RGBN.root_n_iter).
    x: initial points (N, D); the recursion is then skipped.
    neighbours: "grid" or "verlet" (default: grid up to config.RGBN.grid_max_dim).
    preset_overrides: any field of config.RGBNPresetConfig (e.g. spatial_radius=5).
    """
    preset = replace(by_dim(CFG.presets, D), **{k: v for k, v in preset_overrides.items() if v is not None})
    neighbours = neighbours or ("grid" if D <= CFG.grid_max_dim else "verlet")
    log(f"neighbour search: {neighbours}", verbose)
    return _Recursion(D, preset, lr, neighbours, verbose, len(str(N))).level(N, n_iter, x, 0)
