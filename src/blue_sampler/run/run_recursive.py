"""
blue noise sampling solver.
samples random hyperuniform (= sub-poisson density fluctuation) point clouds (N, D)
hyperuniformity is achieved through standard gradient descent on energy kernels
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import gridpoints

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


# ── Presets ───────────────────────────────────────────────────────────────────

_PRESETS = {
    2: dict(spatial_radius=7, spectral_radius=7, LR_spatial=0.100, LR_spectral=0.1, expension_factor=0.3, S=1.0),
    3: dict(spatial_radius=5, spectral_radius=5, LR_spatial=0.030, LR_spectral=0.1, expension_factor=0.3, S=1.0),
    4: dict(spatial_radius=4, spectral_radius=4, LR_spatial=0.010, LR_spectral=0.1, expension_factor=1.0, S=0.5),
    5: dict(spatial_radius=3, spectral_radius=3, LR_spatial=0.003, LR_spectral=0.1, expension_factor=1.0, S=0.5),
}

# N below which we skip the grid pipeline and go straight to the brute-force solver
_BRUTE_THRESHOLDS = {2: 1_000, 3: 2_000, 4: 3_000, 5: 4_000}
_BRUTE_ITER = 60       # iterations of the brute-force solver
_ROOT_N_ITER = 20      # iterations at the root of the recursion


def _preset_for(D: int) -> dict:
    """Preset for dimension D (D < 2 uses the 2D preset, D > 5 uses the 5D one)."""
    return _PRESETS[min(max(D, 2), 5)]


def _resolve(D: int, **given) -> dict:
    """Replace every None in `given` by the preset value for dimension D."""
    preset = _preset_for(D)
    return {k: (preset[k] if v is None else v) for k, v in given.items()}


def _brute_threshold(D: int) -> int:
    return _BRUTE_THRESHOLDS[min(max(D, 2), 5)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _array_backend():
    """(xp, to_xp, to_numpy): cupy if available (gridpoints runs on GPU), numpy otherwise."""
    try:
        import cupy as cp
        return cp, cp.asarray, cp.asnumpy
    except Exception:
        return np, np.asarray, np.asarray


def _finite_rows(x: np.ndarray) -> np.ndarray:
    """Mask of real points (empty grid cells are stored as NaN rows)."""
    return np.isfinite(x).all(axis=-1)


def _make_micro_kernel(sigma2: float, S: float):
    """Pairwise kernel: gaussian at low D, gaussian*sin at high D (sigma2 >= 0.03)."""
    if sigma2 >= 0.03:
        a = 2.0 * np.pi
        b = 2.0 / (sigma2 * a ** 2)
        c = 1.0 / (2.0 * S * np.pi)
        return lambda x, y: gauss_sin_kernel(x, y, a, b, c)
    return lambda x, y: gauss_kernel(x, y, sigma2)

def _make_macro_kernel(spectral_radius, sigma2, N, D, target, IJK = None):
    Ks = integers_in_half_ball(spectral_radius, D)
    K_w, K_ = prepare_wave_vectors(Ks)

    if IJK is None:
        IJK = (N,)
    has_target = target is not None

    if has_target:
        macro_grad = make_target(target, sigma2, D)
    else:
        def macro_grad(x_val):
            """Long-range spectral force (sum over wave vectors)."""
            x_flat = x_val.reshape(-1, D)
            def body(acc, args):
                k, k_ = args
                return acc + spectral_kernel(x_flat, k, k_), None
            out, _ = jax.lax.scan(body, jnp.zeros_like(x_flat), (K_w, K_))
            return out.reshape(*IJK, D)
    return macro_grad

# ── Core pipeline ─────────────────────────────────────────────────────────────

def _recursive_pipeline(
    N: int,
    D: int,
    N_ITER: int,
    *,
    S: float | None = None,
    expension_factor: float | None = None,
    LR_spatial: float | None = None,
    LR_spectral: float | None = None,
    spatial_radius: float | None = None,
    spectral_radius: float | None = None,
    N_PER_STEP: int = 10,
    verbose: int = 1,
    logger: ProgressLogger | None = None,
    x: np.ndarray | None = None,
    target=None,
    _bruteforce: bool = False,
    _is_root: bool = False,
    _is_leaf: bool = True,
) -> np.ndarray:
    """Recursive stealthy-sampling pipeline. Spawns child pipelines when N is large.

    Any hyper-parameter left to None is taken from `_PRESETS[D]`.
    """
    if logger is None:
        logger = ProgressLogger(D, verbose)

    try:
        # ── Parameters ────────────────────────────────────────────────────────
        p = _resolve(
            D, S=S, expension_factor=expension_factor,
            LR_spatial=LR_spatial, LR_spectral=LR_spectral,
            spatial_radius=spatial_radius, spectral_radius=spectral_radius,
        )
        S, expension_factor = p["S"], p["expension_factor"]
        LR_spatial, LR_spectral = p["LR_spatial"], p["LR_spectral"]
        spatial_radius, spectral_radius = p["spatial_radius"], p["spectral_radius"]

        has_target = target is not None
        if has_target and D == 2:
            S = 0.5  # applied after the presets so it is not overwritten

        bruteforce = _bruteforce or (N <= _brute_threshold(D))
        is_root = _is_root or (x is not None) or bruteforce or N <= 3000
        if is_root:
            N_ITER = _ROOT_N_ITER
        if x is None:
            x = np.random.rand(N, D)

        xp, to_xp, to_numpy = _array_backend()
        ctx = logger.enter_level(N, D, N_ITER)

        # ── Geometry and scales ───────────────────────────────────────────────
        Dsimp = min(D, 3)
        IJK, _, Axes = grid_shape(N, D)
        Nsqrt = N ** 0.5
        Ncbrt = N ** (1.0 / D)
        sigma2 = S * 2.0 * (1.0 / Ncbrt) ** 2

        SHIFTS = integers_in_half_ball(spatial_radius, D)
        Clone_simplex = simplex(Dsimp)

        lr_micro = LR_spatial / S
        lr_macro = LR_spectral / (Nsqrt * Ncbrt)

        micro_kernel = _make_micro_kernel(sigma2, S)

        # ── Gradients ─────────────────────────────────────────────────────────
        def micro_grad(x_val):
            """Short-range pair forces over the half-ball of grid shifts (action / reaction)."""
            def body(acc, shift):
                contrib = micro_kernel(x_val, jnp.roll(x_val, shift, axis=Axes))
                return acc + contrib - jnp.roll(contrib, -shift, axis=Axes), None
            out, _ = jax.lax.scan(body, jnp.zeros_like(x_val), SHIFTS)
            return out

        macro_grad = _make_macro_kernel(
            spectral_radius, sigma2, N, D, target, IJK = IJK)
        if has_target:
            lr_micro *= 0.5
            lr_macro = lr_micro

        def full_grad(x_val):
            return lr_micro * micro_grad(x_val) + lr_macro * macro_grad(x_val)

        # ── Grid re-sorting (host callback) ───────────────────────────────────
        shake_offset = 0.0 if has_target else 0.5  # shifts the grid cells between re-sorts

        def _gridify_numpy(x_val: np.ndarray) -> np.ndarray:
            ctx.tick()
            flat = np.random.permutation(np.array(x_val).reshape(-1, D))
            flat = np.array(torus_wrap(flat - shake_offset))

            # empty cells get a random position for the sort, then go back to NaN
            empty = ~_finite_rows(flat)
            flat[empty] = np.random.rand(*flat.shape)[empty]
            flat = to_xp(flat)
            order = gridpoints.argsort(flat, IJK, verbose=0, level=2)
            flat[empty] = xp.nan
            return to_numpy(flat[order].reshape(*IJK, D))

        def gridify(x_val: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(
                _gridify_numpy,
                jax.ShapeDtypeStruct(x_val.shape, x_val.dtype),
                x_val,
            )

        # ── Descent loop ──────────────────────────────────────────────────────
        @jax.jit
        def run_iters(x_val: jnp.ndarray) -> jnp.ndarray:
            def step(i, x_val):
                x_val = jax.lax.cond(i % N_PER_STEP == 0, gridify, lambda v: v, x_val)
                return torus_wrap(x_val - full_grad(x_val))
            return jax.lax.fori_loop(0, N_ITER * N_PER_STEP, step, x_val)

        # ── Cloning (coarse -> fine) ──────────────────────────────────────────
        def clone(x_val: np.ndarray) -> np.ndarray:
            """Expand N//(Dsimp+1) parents into N children via simplex offsets."""
            x_val = np.asarray(x_val).reshape(-1, D)
            x_val = np.random.permutation(x_val[_finite_rows(x_val)])

            N_parents = N // (Dsimp + 1)
            N_keep = N - (Dsimp + 1) * N_parents
            offsets = random_rotations(Clone_simplex, N_parents, D, Dsimp) * (expension_factor / Ncbrt)
            children = (x_val[:N_parents, None, :] + offsets).reshape(-1, D)
            if N_keep > 0:
                children = np.concatenate([x_val[N_parents:], children], axis=0)
            return np.asarray(torus_wrap(jnp.array(children)))

        def solve(x_init: np.ndarray) -> np.ndarray:
            """Brute-force solver for small N, grid descent otherwise."""
            ctx.start()
            if bruteforce:
                x_pts = _bruteforce_pipeline(N, D, _BRUTE_ITER, ctx=ctx, target=target)(x_init)
                return prepare_points(np.asarray(x_pts), N, IJK, D) if is_root else x_pts
            return run_iters(prepare_points(np.asarray(x_init), N, IJK, D))

        # ── Recursion ─────────────────────────────────────────────────────────
        if is_root:
            x_pts = solve(x)
        else:
            x_parent = clone(
                _recursive_pipeline(
                    N=N // (Dsimp + 1) + N % (Dsimp + 1),
                    D=D,
                    N_ITER=N_ITER,
                    S=S,
                    expension_factor=expension_factor,
                    LR_spatial=LR_spatial,
                    LR_spectral=LR_spectral,
                    spatial_radius=spatial_radius,
                    spectral_radius=spectral_radius,
                    N_PER_STEP=N_PER_STEP,
                    logger=logger,
                    target=target,
                    _is_root=False,
                    _is_leaf=False,
                )
            )
            x_pts = solve(x_parent)

        if not bruteforce:
            ctx.done()

        if _is_leaf:
            x_pts = np.array(x_pts).reshape(-1, D)
            x_pts = x_pts[_finite_rows(x_pts)]
        return x_pts

    finally:
        logger.exit_level()