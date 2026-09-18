from __future__ import annotations

import itertools
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ..math import kdtree_order, torus_delta, torus_wrap


def get_shifts(radius: int, dims: int) -> jnp.ndarray:
    ranges = [range(-radius, radius + 1)] * dims
    shifts = [s for s in itertools.product(*ranges) if sum(v * v for v in s) <= radius**2]
    return jnp.array(shifts, dtype=jnp.int32)


def _nufft_pipeline(
    N: int, D: int, lr: float = 1.0, kfrac: float = 1.0,
    warmstart: NDArray | None = None, verbose: int = 1,
    n_iter: int = 200, regrid_every: int = 1000,
) -> NDArray:
    """Generalised NUFFT-style point set optimisation."""
    G = 2 ** int(np.log2(N ** (1 / D)) + 1e-6)
    r = 5 if D == 2 else 4 if D == 3 else 3
    shifts, cell_size = get_shifts(r, D), G**D

    assert N % cell_size == 0, (
        f"N={N} must be divisible by G**D with G power of 2 "
        f"(got G={G}, cell_size={cell_size})"
    )
    n_cells = N // cell_size
    log = print if verbose >= 1 else lambda *_: None

    K_RADIUS, SIGMA2 = max(1, int(G * kfrac)), (1.0 / G) ** 2
    base_delta, AXES = 0.01 / G * lr, tuple(range(D))

    g_axis = jnp.linspace(0.0, 1.0, G, endpoint=False)
    GRID = jnp.stack(jnp.meshgrid(*([g_axis] * D), indexing="ij"), axis=-1)[..., None, :]

    freqs = jnp.fft.fftfreq(G) * G
    freq_r2 = sum(f**2 for f in jnp.meshgrid(*([freqs] * D), indexing="ij"))
    MASK = (freq_r2 > K_RADIUS**2) | (freq_r2 <= 0.01)
    K_WEIGHT = (((1.0 / freq_r2))).at[MASK].set(0.0)[..., None]

    def _loss_and_grad(x_grid, shifts):
        def _fwd(acc, shift):
            delta = torus_delta(GRID - jnp.roll(x_grid, shift, axis=AXES))
            return acc + jnp.exp(-jnp.sum(delta**2, axis=-1) / SIGMA2), None

        dens, _ = jax.lax.scan(_fwd, jnp.zeros((G,) * D + (n_cells,)), shifts)
        F = jnp.fft.fftn(dens, axes=AXES)
        loss = jnp.sum(jnp.abs(F)**3 * K_WEIGHT)
        grad_dens = jnp.fft.ifftn(K_WEIGHT * F * jnp.abs(F), axes=AXES).real

        def _bwd(acc, shift):
            delta = torus_delta(GRID - jnp.roll(x_grid, shift, axis=AXES))
            dist2 = jnp.sum(delta**2, axis=-1, keepdims=True)
            gpix = grad_dens[..., None] * jnp.exp(-dist2 / SIGMA2) * delta
            return acc + jnp.roll(gpix, -shift, axis=AXES), None

        grad_x, _ = jax.lax.scan(_bwd, jnp.zeros_like(x_grid), shifts)
        return loss, grad_x

    @partial(jax.jit, static_argnums=(1,))
    def _run_chunk(x_init, n_steps, adap, prev_loss):
        def step(carry, _):
            x, adap, prev = carry
            loss, g = _loss_and_grad(x, shifts)
            x = torus_wrap(x - g / jnp.mean(jnp.abs(g) + 1e-30) * adap)
            improved = loss < prev
            return (x, jnp.where(improved, adap * 1.05, adap * 0.8), loss), None

        (x, adap, prev), _ = jax.lax.scan(step, (x_init, adap, prev_loss), None, length=n_steps)
        return x, adap, prev

    def _loss(x):
        return _loss_and_grad(x, shifts)[0]

    if warmstart is not None:
        if warmstart.shape != (N, D):
            raise ValueError(f"warmstart must be ({N}, {D}), got {warmstart.shape}")
        x_np = warmstart.astype(np.float32)
    else:
        x_np = np.random.rand(N, D).astype(np.float32)

    def to_grid(pts: NDArray) -> jnp.ndarray:
        return jnp.asarray(kdtree_order(pts, G=G))

    log(f"[nufft] {D}D | {N} pts | {n_iter} iters")

    t0 = time.time()
    x_grid = to_grid(x_np)
    flat = np.asarray(x_grid).reshape(-1, D)
    log(f"kdtree built (elapsed {time.time() - t0:.2f}s)")
    loss0 = _loss(x_grid)
    log(f"loss 0: {loss0:.2e}")

    t0 = time.time()

    for start in range(0, n_iter, regrid_every):
        adap, prev_loss = jnp.asarray(base_delta, dtype=x_grid.dtype), jnp.asarray(jnp.inf)
        nit = min(n_iter - start, regrid_every)
        x_grid = to_grid(flat)
        x_grid, adap, prev_loss = _run_chunk(x_grid, nit, adap, prev_loss)
        curloss = _loss(x_grid)
        log(f"loss {start + nit}: {curloss:.2e}")
        flat = np.asarray(x_grid).reshape(-1, D)


    log(f"done in {time.time() - t0:.2f}s | loss {float(loss0):.2e} → {float(curloss):.2e}")

    return flat
