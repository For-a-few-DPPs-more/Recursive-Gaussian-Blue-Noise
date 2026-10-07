"""
blue noise sampling solver (KNN / Verlet-list version).
samples random hyperuniform (= sub-poisson density fluctuation) point clouds (N, D)
hyperuniformity is achieved through standard gradient descent on energy kernels

Neighbour lists:
  - a skin of `skin_factor * knn` neighbours is built once per level (brute force, cupy if available else numpy)
    and kept until the end of the level,
  - every N_PER_STEP steps the (N, knn) list is re-selected inside the skin (top_k on the
    current distances), fully under jit.

Array convention: numpy at the boundaries (inputs / outputs, Morton sort, cloning, bruteforce),
jax.numpy only inside the jitted descent.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..math import (
    integers_in_half_ball,
    simplex,
    torus_wrap,
    random_rotations,
)

from ..progress import ProgressLogger
from .run_bruteforce import _bruteforce_pipeline

from .run_recursive import (
    _make_micro_kernel, _make_macro_kernel, _brute_threshold,
    _resolve, _BRUTE_ITER,
    _ROOT_N_ITER,
)


def _as_np(a, D: int) -> np.ndarray:
    """Any array-like (numpy or jax) -> numpy (N, D)."""
    return np.asarray(a).reshape(-1, D)


# ── Neighbour lists ───────────────────────────────────────────────────────────

def _array_backend():
    """(xp, to_xp, to_numpy): cupy if available (gridpoints runs on GPU), numpy otherwise."""
    try:
        import cupy as cp
        return cp, cp.asarray, cp.asnumpy
    except Exception:
        return np, np.asarray, np.asarray


def _build_skin(x, n_skin: int, max_elems: int = 2 ** 25) -> jnp.ndarray:
    """Skin: n_skin nearest neighbours on the torus [0,1)^D. Returns (N, n_skin) int32."""
    x = jnp.mod(jnp.asarray(x), 1.0)
    N, D = x.shape
    if n_skin >= N:
        raise ValueError(f"n_skin={n_skin} must be < N={N}")

    # Skin saturé : tous les autres points
    if n_skin == N - 1:
        ids = jnp.broadcast_to(jnp.arange(N - 1, dtype=jnp.int32), (N, N - 1))
        return ids + (ids >= jnp.arange(N, dtype=jnp.int32)[:, None])

    chunk = int(max(1, min(N, max_elems // N)))
    n_blocks = -(-N // chunk)
    pad = n_blocks * chunk - N

    xq = jnp.pad(x, ((0, pad), (0, 0))).reshape(n_blocks, chunk, D)
    idx = jnp.arange(n_blocks * chunk, dtype=jnp.int32).reshape(n_blocks, chunk)

    def block(args):
        xi, gi = args                                   # (chunk, D), (chunk,)
        d = xi[:, None, :] - x[None, :, :]              # (chunk, N, D)
        d = d - jnp.round(d)
        d2 = jnp.sum(d * d, axis=-1)                    # (chunk, N)
        # exclut le point lui-même (les lignes de padding ont gi >= N : aucun effet)
        d2 = jnp.where(gi[:, None] == jnp.arange(N)[None, :], jnp.inf, d2)
        _, nn = jax.lax.top_k(-d2, n_skin)
        return nn.astype(jnp.int32)

    return jax.lax.map(block, (xq, idx)).reshape(-1, n_skin)[:N]


def _make_knn_refresh(N: int, D: int, K: int, n_skin: int, max_elems: int = 2 ** 25):
    """Returns refresh(x, skin) -> (N, K): the K nearest points *inside the skin*, at current positions.
    Processed in blocks of points so memory stays around max_elems floats."""
    chunk = int(max(1, min(N, max_elems // (n_skin * D))))
    n_blocks = -(-N // chunk)
    pad = n_blocks * chunk - N

    def refresh(x, skin):
        xb = jnp.pad(x, ((0, pad), (0, 0))).reshape(n_blocks, chunk, D)
        sb = jnp.pad(skin, ((0, pad), (0, 0))).reshape(n_blocks, chunk, n_skin)

        def block(args):
            xi, si = args
            d = x[si] - xi[:, None, :]          # (chunk, n_skin, D)
            d = d - jnp.round(d)                # minimum image on the torus
            d2 = jnp.sum(d * d, axis=-1)
            _, loc = jax.lax.top_k(-d2, K)      # K smallest d2
            return jnp.take_along_axis(si, loc, axis=1)

        return jax.lax.map(block, (xb, sb)).reshape(-1, K)[:N]

    return refresh


# ── Morton (spatial sorting) ─────────────────────────────────────────────────────

def _morton_order(x: np.ndarray) -> np.ndarray:
    """Permutation sorting the points of the torus [0,1)^D along a Morton (Z-order) curve.
    Points close in space end up close in memory: the gathers x[skin] / x[knn] become cache friendly."""
    N, D = x.shape
    bits = min(16, 63 // D)                       # bits per axis, D * bits <= 63 (uint64 code)
    xs = np.mod(np.asarray(x, dtype=np.float32), 1.0)
    q = np.minimum((xs * (1 << bits)).astype(np.uint64), np.uint64((1 << bits) - 1))
    code = np.zeros(N, dtype=np.uint64)
    for b in range(bits):                         # interleave bits: axis d, bit b -> position b*D + d
        for d in range(D):
            code |= ((q[:, d] >> np.uint64(b)) & np.uint64(1)) << np.uint64(b * D + d)
    return np.argsort(code)


# ── Core pipeline ─────────────────────────────────────────────────────────────

def _recursive_pipeline_verlet(
    N: int,
    D: int,
    N_ITER: int,
    *,
    skin_factor: int = 3,
    N_PER_STEP: int = 10,
    verbose: int = 1,
    S: float | None = None,
    expension_factor: float | None = None,
    LR_spatial: float | None = None,
    LR_spectral: float | None = None,
    spatial_radius: float | None = None,
    spectral_radius: float | None = None,
    logger: ProgressLogger | None = None,
    x: np.ndarray | None = None,
    target=None,
    _bruteforce: bool = False,
    _is_root: bool = False,
) -> np.ndarray:
    """Recursive stealthy-sampling pipeline (KNN/Verlet). Spawns child pipelines when N is large.

    Any hyper-parameter left to None is taken from `_PRESETS[D]`.
    The skin is built once per level; N_ITER is the total number of KNN refreshes
    (the descent runs N_ITER * N_PER_STEP steps).
    Takes and returns numpy arrays of shape (N, D).
    """
    if logger is None:
        logger = ProgressLogger(D, verbose)

    try:
        # ── Parameters ────────────────────────────────────────────────────────
        p = _resolve(
            D, S=S, expension_factor=expension_factor,
            LR_spatial=LR_spatial, LR_spectral=LR_spectral,
            spectral_radius=spectral_radius, spatial_radius=spatial_radius,
        )
        S, expension_factor = p["S"], p["expension_factor"]
        LR_spatial, LR_spectral = p["LR_spatial"], p["LR_spectral"]
        spatial_radius = p["spatial_radius"]
        spectral_radius = p["spectral_radius"]

        knn = 2 * len(integers_in_half_ball(spatial_radius, D))

        has_target = target is not None
        if has_target and D == 2:
            S = 0.5  # applied after the presets so it is not overwritten

        bruteforce = _bruteforce or (N <= _brute_threshold(D))
        is_root = _is_root or (x is not None) or bruteforce or N <= 3_000
        if is_root:
            N_ITER = _ROOT_N_ITER

        if is_root:
            x = np.random.rand(N, D) if x is None else _as_np(x, D)
            x = x.astype(np.float32)
            x = x[_morton_order(x)]

        ctx = logger.enter_level(N, D, N_ITER)

        # ── Geometry and scales ───────────────────────────────────────────────
        Dsimp = min(D, 3)
        Nsqrt = N ** 0.5
        Ncbrt = N ** (1.0 / D)
        sigma2 = S * 2.0 * (1.0 / Ncbrt) ** 2

        K = int(min(knn, N - 1))                    # neighbours in the list
        n_skin = int(min(skin_factor * K, N - 1))   # neighbours in the skin
        K = min(K, n_skin)

        Clone_simplex = simplex(Dsimp)

        lr_micro = LR_spatial / S
        lr_macro = LR_spectral / (Nsqrt * Ncbrt)

        micro_kernel = _make_micro_kernel(sigma2, S)
        refresh = _make_knn_refresh(N, D, K, n_skin)

        # ── Gradients ─────────────────────────────────────────────────────────
        def micro_grad(x_val, knn_idx):
            """Short-range pair forces, pure gather: point i sums k(x_i, x_j) over its own list.
            Non-mutual pairs sit at the edge of the list, where the kernel is negligible."""
            def body(acc, j):  # j: (N,) = k-th neighbour of every point
                return acc + micro_kernel(x_val, x_val[j]), None
            out, _ = jax.lax.scan(body, jnp.zeros_like(x_val), knn_idx.T)
            return out

        macro_grad = _make_macro_kernel(spectral_radius, sigma2, N, D, target)
        if has_target:
            lr_micro *= 0.5
            lr_macro = lr_micro

        # ── Descent loop ──────────────────────────────────────────────────────
        n_steps = max(1, N_ITER) * N_PER_STEP

        def _tick_and_refresh(x_val, knn_idx, skin):
            jax.debug.callback(ctx.tick)
            return refresh(x_val, skin)

        @jax.jit
        def descend(x_val: jnp.ndarray, skin: jnp.ndarray) -> jnp.ndarray:
            """Full descent with the fixed skin; the skin is sorted by distance when built,
            so its first K columns are the initial KNN list."""
            def step(i, carry):
                x_val, knn_idx = carry
                knn_idx = jax.lax.cond(
                    i % N_PER_STEP == 0,
                    _tick_and_refresh,
                    lambda x_, k_, s_: k_,
                    x_val, knn_idx, skin,
                )
                g = lr_micro * micro_grad(x_val, knn_idx) + lr_macro * macro_grad(x_val)
                return torus_wrap(x_val - g), knn_idx

            x_val, _ = jax.lax.fori_loop(0, n_steps, step, (x_val, skin[:, :K]))
            return x_val

        def run_iters(x_init: np.ndarray) -> np.ndarray:
            skin = _build_skin(x_init, n_skin)      # numpy, built once, kept until the end
            out = descend(jnp.asarray(x_init), jnp.asarray(skin))  # np -> jnp only here
            return _as_np(out, D)                   # single jnp -> np conversion

        def run_bruteforce(x_init: np.ndarray) -> np.ndarray:
            out = _bruteforce_pipeline(N, D, _BRUTE_ITER, ctx=ctx, target=target)(x_init)
            return _as_np(out, D)

        # ── Cloning (coarse -> fine), numpy in / numpy out ────────────────────
        def clone(x_val: np.ndarray) -> np.ndarray:
            """Expand N//(Dsimp+1) parents into N children via simplex offsets."""
            x_val = np.random.permutation(x_val)

            N_parents = N // (Dsimp + 1)
            N_keep = N - (Dsimp + 1) * N_parents
            offsets = np.asarray(
                random_rotations(Clone_simplex, N_parents, D, Dsimp)
            ) * (expension_factor / Ncbrt)
            children = (x_val[:N_parents, None, :] + offsets).reshape(-1, D)
            if N_keep > 0:
                children = np.concatenate([x_val[N_parents:], children], axis=0)
            children = _as_np(torus_wrap(jnp.asarray(children)), D)
            return children[_morton_order(children)]

        def solve(x_init: np.ndarray) -> np.ndarray:
            """Brute-force solver for small N, KNN descent otherwise. Always returns numpy (N, D)."""
            ctx.start()
            return run_bruteforce(x_init) if bruteforce else run_iters(x_init)

        # ── Recursion ─────────────────────────────────────────────────────────
        if is_root:
            x_pts = solve(x)
        else:
            x_parent = clone(
                _recursive_pipeline_verlet(
                    N=N // (Dsimp + 1) + N % (Dsimp + 1),
                    D=D,
                    N_ITER=N_ITER,
                    S=S,
                    expension_factor=expension_factor,
                    LR_spatial=LR_spatial,
                    LR_spectral=LR_spectral,
                    spatial_radius=spatial_radius,
                    spectral_radius=spectral_radius,
                    skin_factor=skin_factor,
                    N_PER_STEP=N_PER_STEP,
                    logger=logger,
                    target=target,
                    _is_root=False,
                )
            )
            x_pts = solve(x_parent)

        if not bruteforce:
            ctx.done()

        return x_pts

    finally:
        logger.exit_level()