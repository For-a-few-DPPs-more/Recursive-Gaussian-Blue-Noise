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
    n_iter: int = 60,
    lr: float = 1.0,
    ctx: _LevelCtx | None = None,
    target=None,
):
    """bruteforce (complexity O(N2)) gradient-descent sampler for N ≤ ~3 000 points.
    ctx is an optional logger for printing algorithm progress

    target : ndarray of shape (K, D), or "anything.jpg" optional
        Atoms describing a target density. When given, an extra gradient
        term (built from `fields.make_multi_scales_field_fun`) is added to
        make the density of the sample match the target distribution
    """
    DX     = N ** (-1.0 / D)
    sigma2 = 2.0 * DX ** 2
    high_D = sigma2 >= 0.03

    lr    = 0.03 * DX * lr
    scale = {2: 240, 3: 6000, 4: 60_000}.get(D, 100_000)
    Niter = max(40, int(n_iter * N / scale))

    if high_D:
        a = 2.0 * np.pi
        b = 4.0 / (sigma2 * a ** 2)
        c = 1.0 / (2.0 * np.pi)
        kernel = lambda x, y: gauss_sin_kernel(x, y, a, b, c)
    else:
        kernel = lambda x, y: gauss_kernel(x, y, sigma2)

    target_grad =  make_target(target, sigma2, D)
    target_lr = lr


    if target is not None:
        def target_grad_idx(x, idx):
            return target_grad(x)[idx]
    else:
        def target_grad_idx(x, idx):
            return 0.0

    def grad(x):
        g = jax.vmap(lambda xi: kernel(xi[None], x).sum(axis=0))(x)
        g = g + target_lr * target_grad(x)
        return g

    def grad_idx(x, idx):
        g = jax.vmap(lambda xi: kernel(xi[None], x).sum(axis=0))(x[idx])
        g = g + target_lr * target_grad_idx(x, idx)
        return g

    steps = jnp.array([0.4, 0.7, 1.0, 1.3, 1.6])
    sample_size = min(N, 100)

    @jax.jit
    def _run(x, idxs):
        def step(it, state):
            x, lr = state

            gradx = grad(x)
            gradnorm = jnp.linalg.norm(gradx, axis=-1).mean()

            candidates = jax.vmap(
                lambda step: torus_wrap(
                    x - step * lr * gradx / gradnorm
                )
            )(steps)

            idx = idxs[it]

            scores = jax.vmap(
                lambda candidate: (grad_idx(candidate, idx) ** 2).sum()
            )(candidates)

            best = jnp.argmin(scores)
            lr = lr * steps[best]
            x = candidates[best]

            return x, lr

        return jax.lax.fori_loop(0, Niter, step, (x, lr))[0]
    
    def eta():
        """
        estimate remaining time (optional)
        """
        idxs = np.stack([
            np.arange(N)
            if sample_size == N
            else np.random.choice(N, sample_size, replace=False)
            for _ in range(Niter)
        ])

        @jax.jit
        def one_step(x, idxs):
            def step(_, state):
                x, lr = state

                gradx = grad(x)
                gradnorm = jnp.linalg.norm(gradx, axis=-1).mean()

                candidates = jax.vmap(
                    lambda step: torus_wrap(
                        x - step * lr * gradx / gradnorm
                    )
                )(steps)

                idx = idxs[0]

                scores = jax.vmap(
                    lambda candidate: (grad_idx(candidate, idx) ** 2).sum()
                )(candidates)

                best = jnp.argmin(scores)

                return candidates[best], lr * steps[best]

            return jax.lax.fori_loop(0, 1, step, (x, lr))[0]

        x = one_step(np.random.rand(N, D), jnp.asarray(idxs))
        x.block_until_ready()
        t0 = time.perf_counter()
        x = one_step(x, jnp.asarray(idxs))
        x.block_until_ready()
        elapsed = time.perf_counter() - t0

        eta_seconds = elapsed * Niter
        return eta_seconds



    def sample_fn(init: np.ndarray | None = None) -> jnp.ndarray:
        if ctx is not None:
            ctx.on_bruteforce_start(eta_seconds = eta())
        if init is None:
            init = np.random.rand(N, D)

        idxs = np.stack([
            np.arange(N)
            if sample_size == N
            else np.random.choice(N, sample_size, replace=False)
            for _ in range(Niter)
        ])

        out = _run(jnp.asarray(init), jnp.asarray(idxs))
        out.block_until_ready()
        if ctx is not None:
            ctx.on_bruteforce_done()
        return out

    return sample_fn