"""
Exact gaussian blue noise: gradient descent of a gaussian pair energy, all O(N^2)
pairs at every step. For moderate N, or as the root solver of rgbn.

Both implementations (jax, KeOps) share the scales and the step size.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .. import config
from ..array_lib import KEOPS, keops_error, wrap
from ..kernels import pick_force
from ..progress import Progress, fmt_duration

CFG = config.BRUTEFORCE


@dataclass(frozen=True)
class Schedule:
    sigma2: float       # kernel variance
    step: float         # x <- x + step * force
    n_iter: int
    spacing: float      # mean inter-point distance N^(-1/D)

    @classmethod
    def build(cls, N: int, D: int, n_iter: int | None = None, lr: float = 1.0) -> "Schedule":
        spacing = N ** (-1.0 / D)
        s = CFG.sigma_scale
        base_lr = CFG.learning_rate.get(D, CFG.learning_rate_high_d)
        if n_iter is None:
            n_iter = max(CFG.min_iter, N // CFG.iter_divisor[D]) if D in CFG.iter_divisor else CFG.min_iter
        return cls(sigma2=(s * spacing) ** 2, step=lr * base_lr / s ** 2, n_iter=n_iter, spacing=spacing)

    @property
    def force(self):
        return pick_force(self.sigma2, CFG.periodic_kernel_min_sigma ** 2)


def bruteforce_sampler(N: int, D: int, n_iter: int | None = None, lr: float = 1.0,
                       progress: Progress | None = None):
    """Returns sample(init=None) -> (N, D) numpy array (jax implementation)."""
    import jax
    import jax.numpy as jnp

    sch = Schedule.build(N, D, n_iter, lr)
    pair_force = sch.force

    def step(x):
        f = jnp.sum(pair_force(x[:, None, :] - x[None, :, :], sch.sigma2), axis=1)
        return wrap(x + sch.step * f)

    run = jax.jit(lambda x: jax.lax.fori_loop(0, sch.n_iter, lambda _, x: step(x), x))

    def estimated_duration() -> float:
        one_step = jax.jit(step)
        x = jnp.asarray(np.random.rand(N, D), dtype=jnp.float32)
        one_step(x).block_until_ready()                     # compile
        t0 = time.perf_counter()
        one_step(x).block_until_ready()
        return (time.perf_counter() - t0) * sch.n_iter

    def sample(init=None) -> np.ndarray:
        if progress is not None and progress.verbose >= 1:
            progress.note(f"bruteforce — ETA ~{fmt_duration(estimated_duration())}")
        init = np.random.rand(N, D) if init is None else init
        init = np.asarray(init, dtype=np.float32)
        if init.shape != (N, D):
            raise ValueError(f"init must have shape {(N, D)}, got {init.shape}")
        return np.asarray(run(jnp.asarray(init)).block_until_ready())

    return sample


def bruteforce_sampler_keops(N: int, D: int, n_iter: int | None = None, lr: float = 1.0, verbose: int = 1):
    """Returns sample(init=None) -> (N, D) numpy array (KeOps implementation)."""
    if keops_error() is not None:
        raise RuntimeError(f"KeOps is not working: {keops_error()}")
    from pykeops.numpy import LazyTensor

    sch = Schedule.build(N, D, n_iter, lr)
    pair_force = sch.force

    def sample(init=None) -> np.ndarray:
        x = np.array(np.random.rand(N, D) if init is None else init, dtype=np.float32, order="C")
        bar = Progress(f"[gaussian · keops · N={N} D={D}]", total=sch.n_iter, verbose=verbose)
        for _ in range(sch.n_iter):
            delta = LazyTensor(x[:, None, :]) - LazyTensor(x[None, :, :])
            f = KEOPS.reduce(pair_force(delta, sch.sigma2, KEOPS))
            x = np.ascontiguousarray(wrap(x + sch.step * f), dtype=np.float32)
            bar.tick()
        bar.done()
        return x

    return sample
