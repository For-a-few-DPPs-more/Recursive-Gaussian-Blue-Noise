from __future__ import annotations

import itertools
import time
from functools import partial

import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from ..math import torus_delta, torus_wrap, kdtree_order

def get_shifts(radius: int, dims: int) -> jnp.ndarray:
    ranges = [range(-radius, radius + 1)] * dims
    shifts = [
        s for s in itertools.product(*ranges)
        if sum(v * v for v in s) <= radius * radius
    ]
    return jnp.array(shifts, dtype=jnp.int32)


def _nufft_pipeline(
    N: int,
    D: int,
    lr: float = 1.0,
    kfrac: float = 1.0,
    warmstart: NDArray | None = None,
    verbose: int = 1,
    n_iter: int = 200,
    regrid_every: int = 30,
) -> NDArray:
    """
    Generalised NUFFT-style point set optimisation.
    """
    # Determine grid side G (power of two)
    G = 2 ** int(np.log2(N ** (1 / D)) + 1e-6)
    if D == 2:
        r = 5
    elif D == 3:
        r = 4
    elif D >= 4:
        r = 3

    shifts = get_shifts(r, D)
    cell_size = G ** D
    assert N % cell_size == 0, (
        f"N={N} must be divisible by G**D with G power of 2 "
        f"(got G={G}, cell_size={cell_size})"
    )
    n_cells = N // cell_size

    log = (lambda msg: print(msg)) if verbose >= 1 else (lambda msg: None)

    K_RADIUS = max(1, int(G * kfrac))
    SIGMA2 = (1.0 / G) ** 2
    base_delta = 0.01 / G * lr
    # Spatial axes only; the cell dimension is the last one before the coordinate dim
    AXES = tuple(range(D))

    g_axis = jnp.linspace(0.0, 1.0, G, endpoint=False)
    # GRID: (G,)*D + (1, D)  → broadcasts over the n_cells axis
    GRID = jnp.stack(
        jnp.meshgrid(*([g_axis] * D), indexing="ij"),
        axis=-1,
    )[..., None, :]          # shape (G,)*D + (1, D)

    freqs = jnp.fft.fftfreq(G) * G
    freq_r2 = sum(
        f ** 2 for f in jnp.meshgrid(*([freqs] * D), indexing="ij")
    )
    MASK = (freq_r2 > K_RADIUS ** 2) | (freq_r2 <= 0.5)
    K_WEIGHT = (1.0 / freq_r2)
    K_WEIGHT = K_WEIGHT.at[MASK].set(0.0)
    # Broadcast-ready over cells: (G,)*D + (1,)
    K_WEIGHT = K_WEIGHT[..., None]

    def _loss_and_grad(x_grid, shifts):
        # x_grid: (G,)*D + (n_cells, D)
        def _fwd(acc, shift):
            rolled = jnp.roll(x_grid, shift, axis=AXES)
            delta = torus_delta(GRID - rolled)          # (G,)*D + (n_cells, D)
            dens_contrib = jnp.exp(-jnp.sum(delta ** 2, axis=-1) / SIGMA2)
            return acc + dens_contrib, None

        dens, _ = jax.lax.scan(
            _fwd,
            jnp.zeros((G,) * D + (n_cells,)),
            shifts,
        )
        # dens: (G,)*D + (n_cells,)
        F = jnp.fft.fftn(dens, axes=AXES)
        loss = jnp.sum((jnp.abs(F) ** 2) * K_WEIGHT)
        grad_F = K_WEIGHT * F
        grad_dens = jnp.fft.ifftn(grad_F, axes=AXES).real   # (G,)*D + (n_cells,)

        def _bwd(acc, shift):
            rolled = jnp.roll(x_grid, shift, axis=AXES)
            delta = torus_delta(GRID - rolled)              # (G,)*D + (n_cells, D)
            dist2 = jnp.sum(delta ** 2, axis=-1, keepdims=True)
            gpix = grad_dens[..., None] * jnp.exp(-dist2 / SIGMA2) * delta
            return acc + jnp.roll(gpix, -shift, axis=AXES), None

        grad_x, _ = jax.lax.scan(
            _bwd,
            jnp.zeros_like(x_grid),
            shifts,
        )
        return loss, grad_x

    @partial(jax.jit, static_argnums=(1,))
    def _run_chunk(x_init, n_steps, adap, prev_loss):
        def step(carry, _):
            x, adap, prev = carry

            loss, g = _loss_and_grad(x, shifts)
            rms = jnp.sqrt(jnp.mean(g ** 2) + 1e-30)
            x_new = torus_wrap(x - (g / rms) * adap)

            improved = loss < prev
            adap = jnp.where(improved, adap * 1.01, adap * 0.9)

            return (x_new, adap, loss), None

        (x_f, adap_f, prev_f), _ = jax.lax.scan(
            step,
            (x_init, adap, prev_loss),
            None,
            length=n_steps,
        )
        return x_f, adap_f, prev_f


    def _loss(x):
        return _loss_and_grad(x, shifts)[0]

    if warmstart is not None:
        if warmstart.shape != (N, D):
            raise ValueError(
                f"warmstart must be ({N}, {D}), got {warmstart.shape}"
            )
        x_np = warmstart.astype(np.float32)
    else:
        x_np = np.random.rand(N, D).astype(np.float32)


    def to_grid(pts: NDArray) -> jnp.ndarray:
        """Re-order flat points into the multi-cell grid layout."""
        return jnp.asarray(kdtree_order(pts, G=G))


    log(f"[nufft] {D}D  | {N} pts | {n_iter} iters ")

    t0 = time.time()
    x_grid = to_grid(x_np)
    log(f"kdtree built (elapsed {time.time() - t0:.2f}s)")
    loss0 = _loss(x_grid)
    log(f"loss 0: {loss0:.2e}")

    adap = jnp.asarray(base_delta, dtype=x_grid.dtype)
    prev_loss = jnp.asarray(jnp.inf)
    
    n_steps = regrid_every

    x_grid, adap, prev_loss = _run_chunk(
        x_grid, n_steps, adap, prev_loss
    )

    flat = np.asarray(x_grid).reshape(-1, D)

    # Main optimisation loop
    t0 = time.time()

    adap = jnp.asarray(base_delta, dtype=x_grid.dtype)
    prev_loss = jnp.asarray(jnp.inf)

    for start in range(0, n_iter, regrid_every):

        log(f"loss {start + regrid_every}: {_loss(x_grid):.2e}")
        x_grid = to_grid(flat)
        
        n_steps = regrid_every

        x_grid, adap, prev_loss = _run_chunk(
            x_grid, n_steps, adap, prev_loss
        )

        flat = np.asarray(x_grid).reshape(-1, D)

    x_grid = to_grid(flat)

    loss2 = _loss(x_grid)
    loss2.block_until_ready()

    log(
        f"done in {time.time() - t0:.2f}s | "
        f"loss {float(loss0):.2e} → {float(loss2):.2e}"
    )

    return np.asarray(x_grid).reshape(-1, D)