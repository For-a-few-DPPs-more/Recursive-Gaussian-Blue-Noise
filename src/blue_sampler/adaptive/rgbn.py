"""
Adaptive Gaussian blue noise (2D, jax), ported from stipple (src/rgbn.rs).

Energy of GBN (Ahmed, Ren, Wonka 2022): every dot carries a Gaussian whose size follows the
local density (variance sigma^2 / a, a = target density at the dot, mean 1), and the kernels
must add up to the image. Its gradient is a pair repulsion with the combined kernel of the two
dots, c = 2 a_i a_j / (a_i + a_j), plus an attraction to the image blurred by the dot's own
kernel (read from the pyramid of `field.Density`).

- `adaptive_gbn`: all pairs, O(N^2) per step. Small N.
- `adaptive_rgbn`: neighbours from a grid sorted by gridpoints (the sorted grid stretches with
  the density like the kernels, so one stencil serves dark and pale areas), dots optimised
  coarse to fine, each level cloned x3 from the previous one.
- `adaptive_refine`: more rgbn steps on existing dots, with a wider stencil.
"""
from __future__ import annotations

import numpy as np

from ..array_lib import wrap
from ..lattice import half_ball
from .field import LEVELS, T0, Density, bilinear, interpolate

EXPANSION = 0.3        # size of the cloning triangle, in local spacings
LR = 0.3
STENCIL = 7            # grid neighbourhood of the pair sum (pale areas next to dark ones: ~6 % density error)
REFINE_STENCIL = 10    # wider neighbourhood for `adaptive_refine` (~1 %, ~2x the cost per step)
ROOT_N_ITER = 20       # passes of the root level
N_ITER = 12            # passes of the other levels
N_PER_STEP = 10        # descent steps per pass (grid re-sorted and kernel sizes re-read between passes)
ROOT_MAX_N = 3000      # coarsest level size
A_MIN = 1.0 / 32.0     # kernels stop growing below this density (x the mean): near-white areas stay finite
MAX_MOVE = 0.3         # longest move of a dot in one step, in local spacings
CUTOFF = 30.0          # pair kernel truncated at exp(-30)


def _density(image, invert: bool) -> Density:
    return image if isinstance(image, Density) else Density(image, invert)


def _amap(density: Density, n: int) -> np.ndarray:
    """Local density map a, seen at the scale of the mean spacing of n dots."""
    return np.maximum(density.smooth(0.25 / n), A_MIN)


def _level_sizes(n: int) -> list[int]:
    sizes = [n]
    while sizes[-1] > ROOT_MAX_N:
        m = sizes[-1]
        sizes.append(m // 3 + m % 3)
    return sizes[::-1]


def _draw(density: Density, m: int, rng) -> np.ndarray:
    """m dots drawn from the density (rejection sampling)."""
    G = density.rho.shape[0]
    rho_max = density.rho.max()
    out = []
    while sum(len(o) for o in out) < m:
        p = rng.random((2 * m, 2))
        c = np.minimum((p * G).astype(np.int64), G - 1)
        out.append(p[density.rho[c[:, 0], c[:, 1]] > rng.random(2 * m) * rho_max])
    return np.concatenate(out)[:m].astype(np.float32)


def _clone(parents: np.ndarray, n: int, amap: np.ndarray, rng) -> np.ndarray:
    """N/3 parents -> N children on small randomly rotated triangles, sized by the local spacing
    (the remaining parents are kept as they are)."""
    x = rng.permutation(parents)
    n_par = n // 3
    p = x[:n_par]
    scale = EXPANSION / np.sqrt(n * interpolate(amap, p))
    theta = rng.random(n_par) * 2 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    tri = np.array([(-np.sqrt(0.75), -0.5), (np.sqrt(0.75), -0.5), (0.0, 1.0)])
    off = np.stack([c[:, None] * tri[:, 0] - s[:, None] * tri[:, 1],
                    s[:, None] * tri[:, 0] + c[:, None] * tri[:, 1]], -1) * scale[:, None, None]
    children = (p[:, None, :] + off).reshape(-1, 2)
    return wrap(np.concatenate([x[n_par:], children]).astype(np.float32))


def _attraction(grad, p, k, f):
    """Gradient of the image blurred by each dot's own kernel: levels k and k + 1 blended by f."""
    idx, w = bilinear(p)
    g0 = grad[k[None], idx[..., 0], idx[..., 1]]                # (4, n, 2)
    g1 = grad[k[None] + 1, idx[..., 0], idx[..., 1]]
    return (w[..., None] * (g0 * (1 - f)[None, :, None] + g1 * f[None, :, None])).sum(0)


def _move(p, a, at, pair, sigma2):
    """One descent step, preconditioned by a^-2 (every dot moves in units of its own spacing)."""
    import jax.numpy as jnp
    d = LR / (a * a)[:, None] * (np.pi * sigma2 / 2 * at - pair)
    lim = MAX_MOVE * jnp.sqrt(sigma2 / a)
    m = jnp.linalg.norm(d, axis=-1)
    d = d * jnp.minimum(1.0, lim / jnp.maximum(m, 1e-30))[:, None]
    return wrap(p + d)


def _kernel_levels(a, sigma2):
    """Pyramid level and blend of a dot's own kernel (variance sigma^2 / (4a) per axis)."""
    import jax.numpy as jnp
    l = jnp.clip(jnp.log2(sigma2 / (4 * a) / T0), 0.0, LEVELS - 1 - 1e-3)
    k = jnp.floor(l).astype(jnp.int32)
    return k, l - k


# ── Grid descent (rgbn) ───────────────────────────────────────────────────────

def _grid_pass(gi: int, stencil: int, sigma2: float):
    """Jitted N_PER_STEP steps on a (gi, gi, 2) grid (NaN = empty cell), a and levels frozen."""
    import jax
    import jax.numpy as jnp

    shifts = jnp.asarray(half_ball(stencil ** 2, 2))
    inv_s2 = 1.0 / sigma2

    def pair_sum(x, a):
        """sum_q (x_q - x_p) c^2 exp(-c r^2 / sigma^2), over the half stencil (action / reaction)."""
        def body(acc, s):
            xq, aq = jnp.roll(x, s, (0, 1)), jnp.roll(a, s, (0, 1))
            d = xq - x
            d = d - jnp.round(d)
            c = 2 * a * aq / (a + aq)
            r = jnp.sum(d * d, -1) * inv_s2 * c
            out = jnp.where((r < CUTOFF)[..., None], d * (c * c * jnp.exp(-jnp.minimum(r, CUTOFF)))[..., None], 0.0)
            return acc + out - jnp.roll(out, -s, (0, 1)), None
        return jax.lax.scan(body, jnp.zeros_like(x), shifts)[0]

    @jax.jit
    def run(x, a, k, f, grad):
        full = jnp.isfinite(x[..., 0]).reshape(-1)
        af, kf, ff = a.reshape(-1), k.reshape(-1), f.reshape(-1)

        def step(_, x):
            p = x.reshape(-1, 2)
            ps = jnp.where(full[:, None], p, 0.0)
            new = _move(ps, af, _attraction(grad, ps, kf, ff), pair_sum(x, a).reshape(-1, 2), sigma2)
            return jnp.where(full[:, None], new, jnp.nan).reshape(x.shape)

        return jax.lax.fori_loop(0, N_PER_STEP, step, x)

    return run


def _descend(x: np.ndarray, n_pass: int, density: Density, grad, amap, stencil: int, rng) -> np.ndarray:
    import gridpoints
    import jax.numpy as jnp

    n = len(x)
    gi = int(np.ceil(np.sqrt(n)))
    sigma2 = 1.0 / n
    run = _grid_pass(gi, stencil, sigma2)
    flat = np.full((gi * gi, 2), np.nan, np.float32)
    flat[:n] = x
    for _ in range(n_pass):
        empty = ~np.isfinite(flat).all(-1)
        flat = np.where(empty[:, None], rng.random(flat.shape), wrap(flat)).astype(np.float32)
        order = np.asarray(gridpoints.argsort(flat, (gi, gi), verbose=0, level=2))
        flat, empty = flat[order], empty[order]
        a = np.where(empty, 1.0, interpolate(amap, flat)).astype(np.float32)
        k, f = _kernel_levels(jnp.asarray(a), sigma2)
        flat[empty] = np.nan
        out = run(jnp.asarray(flat.reshape(gi, gi, 2)), jnp.asarray(a.reshape(gi, gi)),
                  k.reshape(gi, gi), f.reshape(gi, gi), grad)
        flat = np.array(out).reshape(-1, 2)
    return flat[np.isfinite(flat).all(-1)]


def adaptive_rgbn(image, N: int, n_iter: int = N_ITER, seed: int | None = None, invert: bool = True) -> np.ndarray:
    """
    N dots in [0, 1)^2 following an image (dark = dense if invert), recursive adaptive GBN.

    image: path, point set (K, 2), (512, 512) density indexed [x, y] (y up), or a `Density`.
    n_iter: passes (of N_PER_STEP steps) per level, the root level gets ROOT_N_ITER.
    """
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    density = _density(image, invert)
    grad = jnp.asarray(density.grad)
    x = None
    for level, m in enumerate(_level_sizes(max(N, 2))):
        amap = _amap(density, m)
        x = _draw(density, m, rng) if level == 0 else _clone(x, m, amap, rng)
        x = _descend(x, ROOT_N_ITER if level == 0 else n_iter, density, grad, amap, STENCIL, rng)
    return density.from_unit(x)


def adaptive_refine(image, x: np.ndarray, n_iter: int = 6, seed: int | None = None, invert: bool = True) -> np.ndarray:
    """n_iter more passes of rgbn on the dots x (N, 2), with the wider REFINE_STENCIL.
    x is given, and returned, in the frame of the target."""
    import jax.numpy as jnp

    density = _density(image, invert)
    x = density.to_unit(x).astype(np.float32)
    x = _descend(x, n_iter, density, jnp.asarray(density.grad), _amap(density, len(x)), REFINE_STENCIL,
                 np.random.default_rng(seed))
    return density.from_unit(x)


# ── All pairs (gbn) ───────────────────────────────────────────────────────────

def adaptive_gbn(image, N: int, n_iter: int = 2 * ROOT_N_ITER, seed: int | None = None, invert: bool = True,
                 x: np.ndarray | None = None) -> np.ndarray:
    """
    N dots in [0, 1)^2 following an image, adaptive GBN with all O(N^2) pairs (N up to ~10^4).

    n_iter: passes of N_PER_STEP steps. x: initial dots (N, 2) in the target's frame, default drawn
    from the density.
    """
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    density = _density(image, invert)
    grad, amap = jnp.asarray(density.grad), jnp.asarray(_amap(density, N))
    sigma2 = 1.0 / N
    x = _draw(density, N, rng) if x is None else density.to_unit(x).astype(np.float32)

    def step(_, p):
        a = interpolate(amap, p)
        k, f = _kernel_levels(a, sigma2)
        d = p[None, :, :] - p[:, None, :]                   # x_q - x_p, row p
        d = d - jnp.round(d)
        c = 2 * a[:, None] * a[None, :] / (a[:, None] + a[None, :])
        r = jnp.sum(d * d, -1) / sigma2 * c
        w = jnp.where(r < CUTOFF, c * c * jnp.exp(-jnp.minimum(r, CUTOFF)), 0.0)
        return _move(p, a, _attraction(grad, p, k, f), jnp.sum(d * w[..., None], 1), sigma2)

    run = jax.jit(lambda p: jax.lax.fori_loop(0, n_iter * N_PER_STEP, step, p))
    return density.from_unit(np.asarray(run(jnp.asarray(x))))
