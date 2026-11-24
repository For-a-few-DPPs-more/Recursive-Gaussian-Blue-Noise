"""
Recursive gaussian blue noise (rgbn).

A level of N points first solves a coarse level of ~N / k points (recursively), clones every
point into k = min(D, 3) + 1 children on a small randomly rotated simplex, then descends

    E = short-range gaussian repulsion + spectral energy on the lowest wave vectors.

Small levels are solved directly, by the exact `gbn` solver when N is small enough.

The short-range sum only visits neighbours, found in one of two ways:
- grid (D <= 4): the points sit on a (I,)^D grid (empty cells = NaN rows), re-sorted by
  gridpoints so that grid neighbours are spatial neighbours; the pair forces are computed
  against rolled copies of the grid.
- Verlet lists: a "skin" of nearest neighbours is found once per level, and the K-nearest
  lists are re-selected inside it, fully under jit.
Both run n_iter blocks of config.RGBN.steps_per_iter steps, refreshing the neighbours at the
start of every block.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from . import config
from .backend import array_module
from .config import RGBNPreset, by_dim
from .gbn import gaussian_force, gbn, periodic_gaussian_force
from .progress import Progress, log
from .torus import half_ball, wrap

CFG = config.RGBN


def rgbn(N: int, D: int, n_iter: int, *, lr: float = 1.0, x=None,
         neighbours: str | None = None, verbose: int = 1, **preset_overrides) -> np.ndarray:
    """
    N points (N, D) of recursive gaussian blue noise in [0, 1)^D.

    n_iter: neighbour updates per level (root levels use config.RGBN.root_n_iter).
    lr: multiplier of both step sizes.
    x: initial points (N, D); the recursion is then skipped.
    neighbours: "grid" or "verlet" (default: grid up to config.RGBN.grid_max_dim).
    preset_overrides: fields of config.RGBNPreset replacing the preset of dimension D
                      (e.g. spatial_radius=5).
    """
    preset = replace(by_dim(CFG.presets, D), **{k: v for k, v in preset_overrides.items() if v is not None})
    neighbours = neighbours or ("grid" if D <= CFG.grid_max_dim else "verlet")
    if neighbours not in ("grid", "verlet"):
        raise ValueError(f"neighbours must be 'grid' or 'verlet', got {neighbours!r}")
    descent = grid_descent if neighbours == "grid" else verlet_descent
    log(f"neighbour search: {neighbours}", verbose)
    width = len(str(N))
    n_children = min(D, 3) + 1                              # vertices of the cloning simplex

    def level(N: int, n_iter: int, x, depth: int) -> np.ndarray:
        bruteforce = N <= by_dim(CFG.bruteforce_max_n, D)
        is_root = x is not None or bruteforce or N <= CFG.root_max_n
        if is_root:
            n_iter = CFG.root_n_iter
        progress = Progress(f"[level {depth} · N = {N:>{width}}]", total=n_iter, verbose=verbose)

        if is_root:
            x = np.random.rand(N, D) if x is None else np.asarray(x).reshape(N, D)
        else:
            coarse = level(N // n_children + N % n_children, n_iter, None, depth + 1)
            x = _clone(coarse, N, n_children, preset.expansion_factor)

        progress.start()
        if bruteforce:
            x = gbn(N, D, x, lr=lr, progress=progress)
        else:
            x = descent(x, n_iter, Energy.build(N, D, preset, lr), preset.spatial_radius, progress)
        progress.done()
        return x

    return level(N, n_iter, x, 0)


def _clone(parents: np.ndarray, N: int, k: int, expansion: float) -> np.ndarray:
    """N points from N // k parents, each replaced by the k vertices of a random small simplex
    (the N % k remaining parents are kept as they are)."""
    D = parents.shape[1]
    parents = np.random.permutation(parents)
    n_parents = N // k
    offsets = _random_rotations(_simplex(k - 1), n_parents, D) * (expansion / N ** (1.0 / D))
    children = (parents[:n_parents, None, :] + offsets).reshape(-1, D)
    children = np.concatenate([parents[n_parents:], children], axis=0)
    return wrap(children.astype(np.float32))


def _simplex(D: int) -> np.ndarray:
    """Vertices (D+1, D) of a regular simplex centred at the origin, unit circumradius."""
    if D == 1:
        return np.array([[-1.0], [1.0]])
    tip = np.zeros((1, D))
    tip[0, -1] = 1.0
    base = np.hstack((_simplex(D - 1), np.zeros((D, 1))))
    return np.vstack((np.sqrt(1.0 - (1.0 / D) ** 2) * base - tip / D, tip))


def _random_rotations(x: np.ndarray, n: int, d_out: int) -> np.ndarray:
    """n random orthonormal embeddings of the point set x (M, d_in) into R^d_out: (n, M, d_out)."""
    Q, _ = np.linalg.qr(np.random.randn(n, d_out, x.shape[-1]))
    return np.einsum("nij,kj->nki", Q, x)


# ── Energy ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Energy:
    """One descent step: x <- x + lr_short * sum_j short_range(x - x_j) - lr_long * long_range_grad(x)."""
    lr_short: float
    lr_long: float
    short_range: callable       # pair force f(x_i - x_j), NaN-safe
    long_range_grad: callable   # x -> gradient of the spectral energy, same shape as x

    @classmethod
    def build(cls, N: int, D: int, preset: RGBNPreset, lr: float) -> "Energy":
        n_axis = N ** (1.0 / D)                             # points per axis
        S = preset.sigma_scale
        sigma2 = S * 2.0 * (1.0 / n_axis) ** 2
        if sigma2 >= CFG.periodic_kernel_min_sigma2:
            short = lambda d: periodic_gaussian_force(d, sigma2 / 2) / S
        else:
            short = lambda d: jnp.nan_to_num(gaussian_force(d, sigma2 / 2), nan=0.0)
        lr_short = lr * (preset.lr_spatial / S)
        lr_long = lr * (preset.lr_spectral / (N ** 0.5 * n_axis))
        return cls(lr_short, lr_long, short, _spectral_gradient(preset.spectral_radius, D))


def _spectral_gradient(spectral_radius: float, D: int):
    """x -> gradient of sum_k |S(k)|^2 / |k|^D over the half-ball of wave vectors (NaN rows ignored)."""
    K = 2.0 * jnp.pi * half_ball(int(spectral_radius ** 2), D) * 1j
    K_dual = -K / (jnp.abs(K) ** D).sum(axis=-1, keepdims=True)

    def one_mode(x, k, k_dual):
        e = jnp.nan_to_num(jnp.exp(jnp.sum(k * x, axis=-1, keepdims=True)), nan=0.0)
        return jnp.real(jnp.sum(e, axis=0, keepdims=True) * k_dual * jnp.conjugate(e))

    def grad(x):
        flat = x.reshape(-1, D)
        out, _ = jax.lax.scan(lambda acc, kk: (acc + one_mode(flat, *kk), None), jnp.zeros_like(flat), (K, K_dual))
        return out.reshape(x.shape)

    return grad


# ── Descent ───────────────────────────────────────────────────────────────────

def _descend(x, state, n_iter: int, energy: Energy, short_range_force, refresh, progress):
    """The jitted descent shared by both neighbour strategies.
    refresh(x, state) -> (x, state); short_range_force(x, state) -> summed pair forces."""
    started = [False]

    def on_block():
        if started[0]:
            progress.tick()
        else:                                               # first block: compilation is over
            started[0] = True
            progress.start()
            progress.note("calibrating…")

    def refresh_and_tick(x, state):
        jax.debug.callback(on_block)
        return refresh(x, state)

    def step(i, carry):
        x, state = jax.lax.cond(i % CFG.steps_per_iter == 0, refresh_and_tick, lambda x, s: (x, s), *carry)
        # the short-range repulsion is a force (+), the long-range term a gradient (-)
        g = energy.lr_long * energy.long_range_grad(x) - energy.lr_short * short_range_force(x, state)
        return wrap(x - g), state

    n_steps = max(1, n_iter) * CFG.steps_per_iter
    return jax.jit(lambda x, s: jax.lax.fori_loop(0, n_steps, step, (x, s))[0])(x, state)


def grid_descent(x: np.ndarray, n_iter: int, energy: Energy, spatial_radius: float, progress) -> np.ndarray:
    """Descent with grid neighbours: every grid shift within spatial_radius."""
    import gridpoints

    N, D = x.shape
    shape = (int(np.ceil(N ** (1.0 / D))),) * D
    axes = tuple(range(D))
    shifts = half_ball(int(spatial_radius ** 2), D)
    shake = 0.5                                             # moves the cells between two re-sorts
    xp, to_xp, to_numpy = array_module()

    def short_range_force(x, _):
        """Pair forces over the half-ball of grid shifts, each pair once (action / reaction)."""
        def body(acc, shift):
            f = energy.short_range(x - jnp.roll(x, shift, axis=axes))
            return acc + f - jnp.roll(f, -shift, axis=axes), None
        return jax.lax.scan(body, jnp.zeros_like(x), shifts)[0]

    def resort(x):
        flat = wrap(np.random.permutation(np.array(x).reshape(-1, D)) - shake)
        empty = ~np.isfinite(flat).all(axis=-1)
        flat[empty] = np.random.rand(*flat.shape)[empty]    # empty cells: random position for the sort...
        flat = to_xp(flat)
        order = gridpoints.argsort(flat, shape, verbose=0, level=2)
        flat[empty] = xp.nan                                # ... then empty again
        return to_numpy(flat[order].reshape(*shape, D))

    def refresh(x, state):
        return jax.pure_callback(resort, jax.ShapeDtypeStruct(x.shape, x.dtype), x), state

    padded = np.full((int(np.prod(shape)), D), np.nan)
    padded[:N] = x
    out = np.asarray(_descend(jnp.asarray(padded.reshape(*shape, D)), None, n_iter, energy,
                              short_range_force, refresh, progress)).reshape(-1, D)
    return out[np.isfinite(out).all(axis=-1)]


def verlet_descent(x: np.ndarray, n_iter: int, energy: Energy, spatial_radius: float, progress) -> np.ndarray:
    """Descent with Verlet lists of K neighbours, K = twice the grid stencil of spatial_radius."""
    x = np.asarray(x, dtype=np.float32)
    x = x[_morton_order(x)]
    N, D = x.shape
    K = int(min(2 * len(half_ball(int(spatial_radius ** 2), D)), N - 1))   # neighbours in a list
    n_skin = int(min(CFG.verlet_skin_factor * K, N - 1))                     # neighbours in the skin
    K = min(K, n_skin)
    knn = _knn_in_skin(N, D, K, n_skin)

    def short_range_force(x, state):
        """Each point sums the forces of its own list (non-mutual pairs sit at the edge of the
        lists, where the kernel is negligible)."""
        neighbours, _ = state
        return jax.lax.scan(lambda acc, j: (acc + energy.short_range(x - x[j]), None),
                            jnp.zeros_like(x), neighbours.T)[0]

    def refresh(x, state):
        _, skin = state
        return x, (knn(x, skin), skin)

    skin = _build_skin(x, n_skin)                           # sorted by distance: first K columns = initial lists
    out = _descend(jnp.asarray(x), (skin[:, :K], skin), n_iter, energy, short_range_force, refresh, progress)
    return np.asarray(out).reshape(-1, D)


def _blocks(n: int, chunk: int):
    """(n_blocks, padding) to process n rows by blocks of `chunk`."""
    n_blocks = -(-n // chunk)
    return n_blocks, n_blocks * chunk - n


def _build_skin(x, n_skin: int) -> np.ndarray:
    """(N, n_skin) int32: the n_skin nearest neighbours of every point on the torus, sorted by distance."""
    x = jnp.mod(jnp.asarray(x), 1.0)
    N, D = x.shape
    if n_skin == N - 1:                                     # every other point
        ids = jnp.broadcast_to(jnp.arange(N - 1, dtype=jnp.int32), (N, N - 1))
        return ids + (ids >= jnp.arange(N, dtype=jnp.int32)[:, None])

    chunk = int(max(1, min(N, CFG.verlet_chunk_elems // N)))
    n_blocks, pad = _blocks(N, chunk)
    xq = jnp.pad(x, ((0, pad), (0, 0))).reshape(n_blocks, chunk, D)
    idx = jnp.arange(n_blocks * chunk, dtype=jnp.int32).reshape(n_blocks, chunk)

    def block(args):
        xi, gi = args
        d = xi[:, None, :] - x[None, :, :]
        d = d - jnp.round(d)
        d2 = jnp.sum(d * d, axis=-1)
        d2 = jnp.where(gi[:, None] == jnp.arange(N)[None, :], jnp.inf, d2)   # not itself
        return jax.lax.top_k(-d2, n_skin)[1].astype(jnp.int32)

    return jax.lax.map(block, (xq, idx)).reshape(-1, n_skin)[:N]


def _knn_in_skin(N: int, D: int, K: int, n_skin: int):
    """refresh(x, skin) -> (N, K): the K nearest points within each skin, at the current positions."""
    chunk = int(max(1, min(N, CFG.verlet_chunk_elems // (n_skin * D))))
    n_blocks, pad = _blocks(N, chunk)

    def refresh(x, skin):
        xb = jnp.pad(x, ((0, pad), (0, 0))).reshape(n_blocks, chunk, D)
        sb = jnp.pad(skin, ((0, pad), (0, 0))).reshape(n_blocks, chunk, n_skin)

        def block(args):
            xi, si = args
            d = x[si] - xi[:, None, :]
            d = d - jnp.round(d)
            loc = jax.lax.top_k(-jnp.sum(d * d, axis=-1), K)[1]
            return jnp.take_along_axis(si, loc, axis=1)

        return jax.lax.map(block, (xb, sb)).reshape(-1, K)[:N]

    return refresh


def _morton_order(x: np.ndarray) -> np.ndarray:
    """Permutation along a Z-order curve: spatial neighbours end up close in memory, which keeps
    the gathers x[skin] and x[knn] cache friendly."""
    N, D = x.shape
    bits = min(16, 63 // D)                                 # D * bits <= 63 (uint64 code)
    q = np.minimum((np.mod(np.asarray(x, dtype=np.float32), 1.0) * (1 << bits)).astype(np.uint64),
                   np.uint64((1 << bits) - 1))
    code = np.zeros(N, dtype=np.uint64)
    for b in range(bits):                                   # axis d, bit b -> position b * D + d
        for d in range(D):
            code |= ((q[:, d] >> np.uint64(b)) & np.uint64(1)) << np.uint64(b * D + d)
    return np.argsort(code)
