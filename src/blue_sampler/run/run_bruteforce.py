"""
blue noise sampling solver.
samples random hyperuniform (= sub-poisson density flucation) point clouds (N, D)
hyperuniformity is achieved through standard gradient descent on energy kernels
"""


from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import time
from ..math import torus_wrap

from ..grad.kernels import gauss_kernel, gauss_sin_kernel
from ..grad.fields import make_target

from ..progress import _LevelCtx

# ── Bruteforce (small N) ──────────────────────────────────────────────────────

def _bruteforce_pipeline(
    N: int,
    D: int,
    n_iter: int,
    lr: float = 1.0,
    ctx: _LevelCtx | None = None,
    target=None,
):
    """bruteforce (complexity O(N2)) gradient-descent sampler for N ≤ ~3 000 points.
    ctx is an optional logger for printing algorithm progress

    target : ndarray of shape (K, D), or "anything.jpg" optional
        Atoms describing a target density. When given, an extra gradient
        term (built from `fields.make_multi_scales_field_fun`) is added
        to make the density of the sample match the target distribution
    """
    DX     = 1.0 / N ** (1.0 / D)
    S      = 1.0 if target is None else 0.5
    sigma2 = S * 2.0 * DX ** 2
    high_D = sigma2 >= 0.03

    lr_table = {2: 0.4, 3: 0.1, 4: 0.05, 5: 0.01}
    lr    = lr * lr_table.get(D, 0.001) / S
    Niter = n_iter//2 if high_D else n_iter

    if high_D:
        a = 2.0 * np.pi
        b = 2.0 / (sigma2 * a ** 2)
        c = 1.0 / (2.0 * S * np.pi)
        kernel = lambda x, y: gauss_sin_kernel(x, y, a, b, c)
    else:
        kernel = lambda x, y: gauss_kernel(x, y, sigma2)

    target_grad =  make_target(target, sigma2, D)

    def grad(x):
        g = jax.vmap(lambda xi: kernel(xi[None], x).sum(axis=0))(x)
        g = g + lr * target_grad(x)
        return g

    @jax.jit
    def _run(x):
        def step(_, x):
            return torus_wrap(x - lr * grad(x))
        return jax.lax.fori_loop(0, Niter, step, x)
    
    def eta():
        """
        estimate remaining time
        """
        @jax.jit
        def one_step(x):
            return torus_wrap(x - lr * grad(x))

        x = one_step(np.random.rand(N, D))
        x.block_until_ready()
        t0 = time.perf_counter()
        x = one_step(x)
        x.block_until_ready()
        elapsed = time.perf_counter() - t0

        eta_seconds = elapsed * Niter
        return eta_seconds



    def sample_fn(init: np.ndarray | None = None) -> jnp.ndarray:
        if ctx is not None:
            ctx.on_bruteforce_start(eta_seconds = eta())
        if init is None:
            init = np.random.rand(N, D)
        out = _run(jnp.asarray(init))
        out.block_until_ready()
        if ctx is not None:
            ctx.on_bruteforce_done()
        return out

    return sample_fn