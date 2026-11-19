"""
Short-range neighbour search strategies for the rgbn descent.

grid_descent    points live on a D-dim grid of shape (I,)^D (empty cells = NaN rows),
                re-sorted by gridpoints so that grid neighbours are spatial neighbours;
                pair forces are computed against rolled copies of the grid.
verlet_descent  a "skin" of nearest neighbours is computed once per level, and the
                K-nearest list is re-selected inside the skin, fully under jit.

Both take and return numpy (N, D) points. The descent runs n_iter blocks of
config.RGBN.steps_per_iter steps; the neighbours are refreshed (and progress
ticked) at the start of every block.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..array_lib import gpu_array_module, wrap
from ..lattice import half_ball

CFG = config.RGBN


def _descend(x, state, n_iter: int, energy, short_range_force, refresh, progress):
    """Jitted descent shared by both strategies.
    refresh(x, state) -> (x, state); short_range_force(x, state) -> summed pair forces."""
    import jax

    started = [False]

    def on_block():
        if started[0]:
            progress.tick()
        else:                                    # first block: compilation is over
            started[0] = True
            progress.start()
            progress.note("calibrating…")

    def refresh_and_tick(x, state):
        jax.debug.callback(on_block)
        return refresh(x, state)

    def step(i, carry):
        x, state = jax.lax.cond(i % CFG.steps_per_iter == 0, refresh_and_tick, lambda x, s: (x, s), *carry)
        # short-range repulsion is a force (+), the long-range term a gradient (-)
        g = energy.lr_long * energy.long_range_grad(x) - energy.lr_short * short_range_force(x, state)
        return wrap(x - g), state

    n_steps = max(1, n_iter) * CFG.steps_per_iter
    return jax.jit(lambda x, s: jax.lax.fori_loop(0, n_steps, step, (x, s))[0])(x, state)


# ── Grid ──────────────────────────────────────────────────────────────────────

def grid_descent(x: np.ndarray, n_iter: int, energy, spatial_radius: float, progress) -> np.ndarray:
    import gridpoints
    import jax
    import jax.numpy as jnp

    N, D = x.shape
    shape = (int(np.ceil(N ** (1.0 / D))),) * D
    axes = tuple(range(D))
    shifts = half_ball(int(spatial_radius ** 2), D)
    shake = 0.5                                      # shifts the cells between two re-sorts
    xp, to_xp, to_numpy = gpu_array_module()

    def short_range_force(x, _):
        """Pair forces over the half-ball of grid shifts (action / reaction)."""
        def body(acc, shift):
            f = energy.short_range(x - jnp.roll(x, shift, axis=axes))
            return acc + f - jnp.roll(f, -shift, axis=axes), None
        return jax.lax.scan(body, jnp.zeros_like(x), shifts)[0]

    def resort(x):
        flat = wrap(np.random.permutation(np.array(x).reshape(-1, D)) - shake)
        empty = ~np.isfinite(flat).all(axis=-1)
        flat[empty] = np.random.rand(*flat.shape)[empty]     # empty cells: random position for the sort...
        flat = to_xp(flat)
        order = gridpoints.argsort(flat, shape, verbose=0, level=2)
        flat[empty] = xp.nan                                  # ... then empty again
        return to_numpy(flat[order].reshape(*shape, D))

    def refresh(x, state):
        return jax.pure_callback(resort, jax.ShapeDtypeStruct(x.shape, x.dtype), x), state

    padded = np.full((int(np.prod(shape)), D), np.nan)
    padded[:N] = x
    out = np.asarray(_descend(jnp.asarray(padded.reshape(*shape, D)), None, n_iter, energy,
                              short_range_force, refresh, progress)).reshape(-1, D)
    return out[np.isfinite(out).all(axis=-1)]


# ── Verlet lists ──────────────────────────────────────────────────────────────

def _blocks(n: int, chunk: int):
    """(n_blocks, padding) to process n rows by blocks of `chunk`."""
    n_blocks = -(-n // chunk)
    return n_blocks, n_blocks * chunk - n


def build_skin(x, n_skin: int, max_elems: int = CFG.verlet_chunk_elems):
    """(N, n_skin) int32: the n_skin nearest neighbours of every point on the torus, sorted by distance."""
    import jax
    import jax.numpy as jnp

    x = jnp.mod(jnp.asarray(x), 1.0)
    N, D = x.shape
    if n_skin >= N:
        raise ValueError(f"n_skin={n_skin} must be < N={N}")
    if n_skin == N - 1:                                       # every other point
        ids = jnp.broadcast_to(jnp.arange(N - 1, dtype=jnp.int32), (N, N - 1))
        return ids + (ids >= jnp.arange(N, dtype=jnp.int32)[:, None])

    chunk = int(max(1, min(N, max_elems // N)))
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


def knn_in_skin(N: int, D: int, K: int, n_skin: int, max_elems: int = CFG.verlet_chunk_elems):
    """refresh(x, skin) -> (N, K): the K nearest points among each skin, at the current positions."""
    import jax
    import jax.numpy as jnp

    chunk = int(max(1, min(N, max_elems // (n_skin * D))))
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


def morton_order(x: np.ndarray) -> np.ndarray:
    """Permutation sorting the points along a Z-order curve: spatial neighbours end up
    close in memory, which makes the gathers x[skin] / x[knn] cache friendly."""
    N, D = x.shape
    bits = min(16, 63 // D)                                   # D * bits <= 63 (uint64 code)
    q = np.minimum((np.mod(np.asarray(x, dtype=np.float32), 1.0) * (1 << bits)).astype(np.uint64),
                   np.uint64((1 << bits) - 1))
    code = np.zeros(N, dtype=np.uint64)
    for b in range(bits):                                     # axis d, bit b -> position b*D + d
        for d in range(D):
            code |= ((q[:, d] >> np.uint64(b)) & np.uint64(1)) << np.uint64(b * D + d)
    return np.argsort(code)


def verlet_descent(x: np.ndarray, n_iter: int, energy, spatial_radius: float, progress) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    x = np.asarray(x, dtype=np.float32)
    x = x[morton_order(x)]
    N, D = x.shape
    K = int(min(2 * len(half_ball(int(spatial_radius ** 2), D)), N - 1))    # neighbours in the list
    n_skin = int(min(CFG.verlet_skin_factor * K, N - 1))                     # neighbours in the skin
    K = min(K, n_skin)
    knn = knn_in_skin(N, D, K, n_skin)

    def short_range_force(x, state):
        """Point i sums the forces of its own list (non-mutual pairs sit at the edge of the
        lists, where the kernel is negligible)."""
        neighbours, _ = state
        return jax.lax.scan(lambda acc, j: (acc + energy.short_range(x - x[j]), None),
                            jnp.zeros_like(x), neighbours.T)[0]

    def refresh(x, state):
        _, skin = state
        return x, (knn(x, skin), skin)

    skin = build_skin(x, n_skin)                  # sorted by distance: its first K columns are the initial list
    out = _descend(jnp.asarray(x), (skin[:, :K], skin), n_iter, energy, short_range_force, refresh, progress)
    return np.asarray(out).reshape(-1, D)
