"""
Blue noise sampling solver.

Samples random hyperuniform (= sub-Poisson density fluctuation) point clouds
of shape (N, D). Hyperuniformity is achieved through gradient descent on
interaction kernels.
"""

from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

from ..math import torus_wrap
from ..grad.fields import make_target
from ..progress import _LevelCtx


LEARNING_RATE = {2: 0.3, 3: 0.1, 4: 0.05, 5: 0.01}


def gauss_kernel(x, y, sigma2):
    """Gaussian interaction kernel with minimum-image distances on the torus.

    Args:
        x: Array of shape (N, 1, D).
        y: Array of shape (1, M, D).
        sigma2: Squared Gaussian bandwidth (σ²).

    Returns:
        Array of shape (N, D), containing summed interaction forces.
    """
    delta = x - y
    delta = delta - jnp.round(delta)          # minimum image

    dist2 = jnp.sum(delta**2, axis=-1, keepdims=True)
    weights = jnp.exp(-dist2 / (2.0 * sigma2))

    return jnp.sum(delta * weights, axis=1)


def gauss_sin_kernel(x, y, sigma2):
    """Periodic Gaussian interaction kernel (sinusoidal).

    Args:
        x: Array of shape (N, 1, D).
        y: Array of shape (1, M, D).
        sigma2: Squared bandwidth (σ²).

    Returns:
        Array of shape (N, D), containing summed interaction forces.
    """
    delta = x - y
    sin_pi = jnp.sin(jnp.pi * delta)
    s2 = jnp.sum(sin_pi**2, axis=-1, keepdims=True) / (2.0 * jnp.pi**2 * sigma2)

    weights = jnp.exp(-s2)
    force = jnp.sin(2.0 * jnp.pi * delta)

    return jnp.sum(force * weights, axis=1) / (2.0 * jnp.pi)


def _default_steps(N: int, D: int) -> int:
    if D == 2:
        return max(20, N // 4)
    if D == 3:
        return max(20, N // 100)
    return 20


def _bruteforce_pipeline(
    N: int,
    D: int,
    n_iter: int | None = None,
    lr: float = 1.0,
    ctx: _LevelCtx | None = None,
    target=None,
    s: float = 1.0,
):
    """Brute-force O(N²) gradient-descent sampler for moderate N.

    Args:
        N: Number of sample points.
        D: Spatial dimension.
        n_iter: Number of iterations; defaults to _default_steps(N, D).
        lr: Learning-rate multiplier.
        ctx: Optional progress logger.
        target: Optional target density (points of shape (K, D) or image path).
        s: Interaction length-scale multiplier.

    Returns:
        A function sample_fn(init=None) that returns an array of shape (N, D).
    """
    DX = N ** (-1.0 / D)              
    sigma = s * DX
    sigma2 = sigma ** 2               

    Niter = _default_steps(N, D) if n_iter is None else n_iter
    base_coef = LEARNING_RATE.get(D, 0.001) / s**2
    coef = lr * base_coef

    target_lr = 0.03 * DX * lr

    if sigma < 0.1:
        kernel = lambda x, y: gauss_kernel(x, y, sigma2)
    else:
        kernel = lambda x, y: gauss_sin_kernel(x, y, sigma2)

    target_grad = make_target(target, sigma2, D)

    def force(x):
        g = kernel(x[:, None, :], x[None, :, :])  
        return g + target_lr * target_grad(x)

    def step(x):
        return torus_wrap(x + coef * force(x))

    one_step = jax.jit(step)

    @jax.jit
    def _run(x):
        return jax.lax.fori_loop(0, Niter, lambda _, x: step(x), x)

    def eta():
        x = jnp.asarray(np.random.rand(N, D), dtype=jnp.float32)
        one_step(x).block_until_ready()          # warm-up
        t0 = time.perf_counter()
        one_step(x).block_until_ready()
        return (time.perf_counter() - t0) * Niter

    def sample_fn(init: np.ndarray | None = None) -> jnp.ndarray:
        if ctx is not None:
            ctx.on_bruteforce_start(eta_seconds=eta())

        if init is None:
            init = np.random.rand(N, D)

        init = np.asarray(init, dtype=np.float32)
        if init.shape != (N, D):
            raise ValueError(f"init must have shape {(N, D)}, got {init.shape}")

        out = _run(jnp.asarray(init, dtype=jnp.float32))
        out.block_until_ready()

        if ctx is not None:
            ctx.on_bruteforce_done()

        return out

    return sample_fn