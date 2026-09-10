"""
blue noise sampling solver.
samples random hyperuniform (= sub-poisson density fluctuation) point clouds (N, D)
hyperuniformity is achieved through standard gradient descent on energy kernels
"""


from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from squarenet import SquareNet

from ..math import (
    integers_in_half_ball,
    simplex,
    grid_shape,
    torus_wrap,
    prepare_wave_vectors,
    prepare_points,
    random_rotations,
)
from ..grad.kernels import (
    gauss_kernel,
    gauss_sin_kernel,
    spectral_kernel,
)

from ..grad.fields import make_target
from ..progress import ProgressLogger
from .run_bruteforce import _bruteforce_pipeline

_PRESETS = {
    2: dict(spatial_radius=7, spectral_radius=7, LR_spatial=0.100, LR_spectral=0.1, expension_factor=0.3, S=1.0),
    3: dict(spatial_radius=5, spectral_radius=5, LR_spatial=0.030, LR_spectral=0.1, expension_factor=0.3, S=1.0),
    4: dict(spatial_radius=4, spectral_radius=4, LR_spatial=0.010, LR_spectral=0.1, expension_factor=1.0, S=0.5),
    5: dict(spatial_radius=4, spectral_radius=4, LR_spatial=0.003, LR_spectral=0.1, expension_factor=1.5, S=0.5),
}

# ── Core pipeline ─────────────────────────────────────────────────────────────

def _recursive_pipeline(
    N: int,
    D: int,
    N_ITER: int,
    logger: ProgressLogger,
    *,
    S: float,
    expension_factor: float,
    LR_spatial: float,
    LR_spectral: float,
    spatial_radius: float,
    spectral_radius: float,
    N_PER_STEP: int,
    x: np.ndarray | None = None,
    target=None,
    _bruteforce: bool = False,
    _is_root: bool = False,
    _is_leaf: bool = True,
) -> np.ndarray:
    """Recursive stealthy-sampling pipeline. Spawns child pipelines when N is large."""
    try:
        has_target = target is not None
        is_root    = _is_root or (N <= 3_000) or (x is not None)
        brute_thresh = 1000 if D == 2 else 3_000
        brute_ITER = 600 if D == 2 else 60
        if x is None:
            x = np.random.rand(N, D)
        if has_target and D == 2:
            #spatial_radius = 8
            S = 0.5
        if is_root:
            N_ITER = 50

        ctx = logger.enter_level(N, D, N_ITER)
        Dsimp      = min(D, 3)
        IJK, _, Axes = grid_shape(N, D)
        Nsqrt      = N ** 0.5
        Ncbrt      = N ** (1.0 / D)
        bruteforce = _bruteforce or (N <= brute_thresh)
        sigma2     = S * 2.0 * (1.0 / Ncbrt) ** 2
        high_D     = sigma2 >= 0.03

        SHIFTS        = integers_in_half_ball(spatial_radius, D)
        Ks            = integers_in_half_ball(spectral_radius, D)
        K_w, K_       = prepare_wave_vectors(Ks)
        Clone_simplex = simplex(Dsimp)

        lr_micro = (LR_spatial / S) 
        lr_macro = (LR_spectral / (Nsqrt * Ncbrt))


        if high_D:
            a = 2.0 * np.pi
            b = 2.0 / (sigma2 * a ** 2)
            c = 1.0 / (2.0 * S * np.pi)
            micro_kernel = lambda x_val, y_val: gauss_sin_kernel(x_val, y_val, a, b, c)
        else:
            micro_kernel = lambda x_val, y_val: gauss_kernel(x_val, y_val, sigma2)
        
        def true_cells(x):
            return np.isfinite(x).all(axis = -1)
        
        def empty_cells(x):
            return ~true_cells(x)
        
        def micro_grad(x_val):
            def body(acc, shift):
                contrib = micro_kernel(x_val, jnp.roll(x_val, shift, axis=Axes))
                return acc + contrib - jnp.roll(contrib, -shift, axis=Axes), None
            out, _ = jax.lax.scan(body, jnp.zeros_like(x_val), SHIFTS)
            return out

        if has_target:
            macro_grad = make_target(target, sigma2, D)
            slow_down = 0.5
            lr_micro *= slow_down
            lr_macro = lr_micro
  
        else:
            def macro_grad(x_val):
                x_flat = x_val.reshape(-1, D)
                def body(acc, args):
                    k, k_ = args
                    return acc + spectral_kernel(x_flat, k, k_), None
                out, _ = jax.lax.scan(body, jnp.zeros_like(x_flat), (K_w, K_))
                return out.reshape(*IJK, D)
    
        sn = SquareNet(gridshape=IJK, max_iter=50, verbose=0)

        shake_offset = 0.5 if not has_target else 0.0

        def _gridify_numpy(x_val: np.ndarray) -> np.ndarray:
            ctx.tick()
            x_val = np.array(x_val)  
            flat = np.array(torus_wrap(np.random.permutation(x_val.reshape(-1, D)) - shake_offset))  
            empty_mask = empty_cells(flat)
            flat[empty_mask] = np.random.rand(*flat.shape)[empty_mask]
            sn.fit(flat, method="ultimate")
            flat[empty_mask] = np.nan
            x_new = sn.map(flat)
            return x_new

        def gridify(x_val: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(
                _gridify_numpy,
                jax.ShapeDtypeStruct(x_val.shape, x_val.dtype),
                x_val,
            )

        def clone(x_val: np.ndarray) -> np.ndarray:
            """Expand N//(Dsimp+1) parents into N children via simplex offsets."""
            x_val     = np.asarray(x_val).reshape(-1, D) 
            x_val     = x_val[true_cells(x_val)]
            x_val     = np.random.permutation(x_val)
            N_parents = N // (Dsimp + 1)
            N_keep    = N - (Dsimp + 1) * N_parents
            offsets   = random_rotations(Clone_simplex, N_parents, D, Dsimp) * (expension_factor / Ncbrt)
            children  = (x_val[:N_parents, None, :] + offsets).reshape(-1, D)
            if N_keep > 0:
                children = np.concatenate([x_val[N_parents:], children], axis=0)
            return np.asarray(torus_wrap(jnp.array(children)))  # no in-place write after -> asarray fine (unchanged)
        
        def full_grad(x):
            g = lr_micro * micro_grad(x) + lr_macro * macro_grad(x)
            return g

        @jax.jit
        def run_iters(x_val: jnp.ndarray) -> jnp.ndarray:
            def step(i, x_val):
                x_val = jax.lax.cond(
                    i % N_PER_STEP == 0,
                    gridify,
                    lambda val: val,
                    x_val,
                )
                return torus_wrap(x_val - full_grad(x_val))

            return jax.lax.fori_loop(0, N_ITER * N_PER_STEP, step, x_val)

        if is_root:
            if bruteforce:
                ctx.start()
                x_pts = _bruteforce_pipeline(
                    N, D, brute_ITER, ctx = ctx, 
                    target=target, 
                )(x)
                x_pts   = prepare_points(np.asarray(x_pts), N, IJK, D)
            else:
                ctx.start()
                x_pts = prepare_points(np.asarray(x), N, IJK, D)
                x_pts = run_iters(x_pts)
        else:
            N_child = N // (Dsimp + 1) + N % (Dsimp + 1)
            xparent = clone(
                _recursive_pipeline(
                    N=N_child,
                    D=D,
                    logger=logger,
                    S=S,
                    expension_factor=expension_factor,
                    LR_spatial=LR_spatial,
                    LR_spectral=LR_spectral,
                    spatial_radius=spatial_radius,
                    spectral_radius=spectral_radius,
                    N_ITER=N_ITER,
                    N_PER_STEP=N_PER_STEP,
                    target=target,
                    _is_root=False,
                    _is_leaf=False,
                )
            )
            ctx.start()
            if bruteforce:
                x_pts = _bruteforce_pipeline(
                    N, D, brute_ITER, ctx = ctx, 
                    target=target, 
                )(xparent)
            else:
                x_pts = prepare_points(xparent, N, IJK, D)
                x_pts = run_iters(x_pts)
        if not bruteforce:
            ctx.done()

        if _is_leaf:
            x_pts = np.array(x_pts.reshape(-1, D))  
            x_pts = x_pts[true_cells(x_pts)]

        return x_pts

    finally:
        logger.exit_level()