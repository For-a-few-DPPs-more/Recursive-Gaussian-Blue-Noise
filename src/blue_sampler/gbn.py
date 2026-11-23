"""
Gaussian blue noise (Ahmed, Ren, Wonka 2022): gradient descent of a gaussian pair energy,
over all O(N^2) pairs at every step. For moderate N, and as the root solver of rgbn.

The pair forces are written once for jax arrays and for KeOps LazyTensors (which spell
elementwise math as methods, `x.exp()`), through the small `ops` namespaces below.
"""
from __future__ import annotations

import time
from math import pi
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from . import config
from .backend import keops_error
from .progress import Progress, fmt_duration
from .torus import wrap

CFG = config.BRUTEFORCE

JAX_OPS = SimpleNamespace(exp=jnp.exp, sin=jnp.sin, round=jnp.round,
                          sum=lambda x: jnp.sum(x, axis=-1, keepdims=True))
KEOPS_OPS = SimpleNamespace(exp=lambda x: x.exp(), sin=lambda x: x.sin(), round=lambda x: x.round(),
                            sum=lambda x: x.sum(dim=2))


# ── Pair forces ───────────────────────────────────────────────────────────────
# delta = x_i - x_j (raw, not wrapped); the force of j on i, to be summed over j.

def gaussian_force(delta, sigma2: float, ops=JAX_OPS):
    """delta exp(-|delta|^2 / (2 sigma2)), delta taken as the shortest displacement on the torus."""
    delta = delta - ops.round(delta)
    return delta * ops.exp(-ops.sum(delta ** 2) / (2.0 * sigma2))


def periodic_gaussian_force(delta, sigma2: float, ops=JAX_OPS):
    """Smooth periodic version of `gaussian_force` (|delta| -> |sin(pi delta)| / pi), for wide
    kernels, where the shortest displacement jumps across the torus."""
    s2 = ops.sum(ops.sin(pi * delta) ** 2) / (2.0 * pi ** 2 * sigma2)
    return ops.sin(2.0 * pi * delta) * ops.exp(-s2) / (2.0 * pi)


def _schedule(N: int, D: int, n_iter: int | None, lr: float):
    """(force, sigma2, step, n_iter): kernel width ~ the mean spacing N^(-1/D)."""
    spacing = N ** (-1.0 / D)
    s = CFG.sigma_scale
    sigma2 = (s * spacing) ** 2
    step = lr * CFG.learning_rate.get(D, CFG.learning_rate_high_d) / s ** 2
    if n_iter is None:
        n_iter = max(CFG.min_iter, N // CFG.iter_divisor[D]) if D in CFG.iter_divisor else CFG.min_iter
    periodic = sigma2 >= CFG.periodic_kernel_min_sigma ** 2
    return (periodic_gaussian_force if periodic else gaussian_force), sigma2, step, n_iter


# ── Samplers ──────────────────────────────────────────────────────────────────

def gbn(N: int, D: int, x=None, n_iter: int | None = None, lr: float = 1.0,
        progress: Progress | None = None) -> np.ndarray:
    """
    N points (N, D) of gaussian blue noise, jax implementation.
    x: starting points, uniform random by default. n_iter: default from config.BRUTEFORCE.
    progress: shows an estimate of the duration (the descent itself is one jitted loop).
    """
    force, sigma2, step, n_iter = _schedule(N, D, n_iter, lr)

    def one_step(x):
        f = jnp.sum(force(x[:, None, :] - x[None, :, :], sigma2), axis=1)
        return wrap(x + step * f)

    if progress is not None and progress.verbose >= 1:
        progress.note(f"bruteforce — ETA ~{fmt_duration(_time_one_step(one_step, N, D) * n_iter)}")
    x = np.random.rand(N, D) if x is None else x
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (N, D):
        raise ValueError(f"starting points must have shape {(N, D)}, got {x.shape}")
    run = jax.jit(lambda x: jax.lax.fori_loop(0, n_iter, lambda _, x: one_step(x), x))
    return np.asarray(run(jnp.asarray(x)).block_until_ready())


def _time_one_step(one_step, N, D) -> float:
    one_step = jax.jit(one_step)
    x = jnp.asarray(np.random.rand(N, D), dtype=jnp.float32)
    one_step(x).block_until_ready()                         # compile
    t0 = time.perf_counter()
    one_step(x).block_until_ready()
    return time.perf_counter() - t0


def gbn_keops(N: int, D: int, x=None, n_iter: int | None = None, lr: float = 1.0,
              verbose: int = 1) -> np.ndarray:
    """Same as `gbn` with KeOps reductions: no N x N matrix in memory, fast on GPU."""
    if keops_error() is not None:
        raise RuntimeError(f"KeOps is not working: {keops_error()}")
    from pykeops.numpy import LazyTensor

    force, sigma2, step, n_iter = _schedule(N, D, n_iter, lr)
    x = np.array(np.random.rand(N, D) if x is None else x, dtype=np.float32, order="C")
    bar = Progress(f"[gaussian · keops · N={N} D={D}]", total=n_iter, verbose=verbose)
    for _ in range(n_iter):
        delta = LazyTensor(x[:, None, :]) - LazyTensor(x[None, :, :])
        f = force(delta, sigma2, KEOPS_OPS).sum_reduction(dim=1)
        x = np.ascontiguousarray(wrap(x + step * f), dtype=np.float32)
        bar.tick()
    bar.done()
    return x
