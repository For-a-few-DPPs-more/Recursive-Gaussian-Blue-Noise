"""
Adaptive gaussian blue noise: dots following a density, e.g. image stippling (2D, jax).
Port of stipple (https://github.com/adecacqueray/stipple, src/rgbn.rs); its parameters are
the defaults of config.ADAPTIVE.

    x = adaptive_rgbn("zebra.jpg", N=20_000)
    x = adaptive_refine("zebra.jpg", x)        # optimise further

Energy (GBN, Ahmed, Ren, Wonka 2022). Every dot carries a gaussian whose variance follows the
local density, sigma^2 / a with a = density at the dot (mean 1), and the kernels must add up
to the image. Its gradient has two parts:
- a pair repulsion with the combined kernel of the two dots, c = 2 a_i a_j / (a_i + a_j);
- an attraction towards the image blurred by the dot's own kernel. Rather than blurring per
  dot, `Density` precomputes the gradient of the image blurred at every octave of variance
  (a pyramid), and a dot reads the two levels around its own kernel size.

Two solvers, as in the uniform case:
- adaptive_gbn    all pairs, O(N^2) per step: N up to ~10^4.
- adaptive_rgbn   coarse to fine: each level is cloned x3 into the next, then descended with
                  neighbours read off a grid sorted by gridpoints. The sorted grid stretches
                  with the density like the kernels do, so one stencil serves dark and pale areas.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import config
from .progress import Progress
from .targets import as_target, for_adaptive
from .torus import half_ball, wrap

CFG = config.ADAPTIVE


# ── Public samplers ───────────────────────────────────────────────────────────

def adaptive_rgbn(target, N: int, lr: float = 1.0, n_iter_scale: int = 6, x: np.ndarray | None = None,
                  seed: int | None = None, invert: bool = True, verbose: int = 1) -> np.ndarray:
    """
    N dots (N, 2) following a density, coarse to fine.

    target: image path (dark = dense if invert), 2D point set (K, 2), `Target`, (G, G) density
            grid indexed [x, y] (y up), or a `Density`. The dots are returned in its frame.
    lr: multiplier of config.ADAPTIVE.lr.
    n_iter_scale: passes per level = config.ADAPTIVE.iter_factor * n_iter_scale
                  (the coarsest level gets config.ADAPTIVE.root_n_iter).
    x: starting dots (N, 2) in the target's frame; the coarse-to-fine recursion is then skipped
       and they get root_n_iter passes.
    """
    rng = np.random.default_rng(seed)
    density = _as_density(target, invert)
    grad = jnp.asarray(density.grad)
    step = CFG.lr * lr

    if x is not None:
        x = density.to_unit(x).astype(np.float32)
        progress = Progress(f"[adaptive · N = {N}]", total=CFG.root_n_iter, verbose=verbose)
        x = _grid_descent(x, CFG.root_n_iter, grad, _local_density(density, N), CFG.stencil, step, rng, progress)
        progress.done()
        return density.from_unit(x)

    sizes = _level_sizes(max(N, 2))
    width = len(str(N))
    for level, n in enumerate(sizes):
        n_pass = CFG.root_n_iter if level == 0 else CFG.iter_factor * n_iter_scale
        progress = Progress(f"[adaptive level {len(sizes) - 1 - level} · N = {n:>{width}}]",
                            total=n_pass, verbose=verbose)
        amap = _local_density(density, n)
        x = _draw(density, n, rng) if level == 0 else _clone(x, n, amap, rng)
        x = _grid_descent(x, n_pass, grad, amap, CFG.stencil, step, rng, progress)
        progress.done()
    return density.from_unit(x)


def adaptive_refine(target, x: np.ndarray, n_iter: int = 6, lr: float = 1.0, seed: int | None = None,
                    invert: bool = True, verbose: int = 1) -> np.ndarray:
    """n_iter more passes on the dots x (N, 2), with the wider config.ADAPTIVE.refine_stencil.
    x is given, and returned, in the target's frame."""
    density = _as_density(target, invert)
    x = density.to_unit(x).astype(np.float32)
    progress = Progress(f"[adaptive refine · N = {len(x)}]", total=n_iter, verbose=verbose)
    x = _grid_descent(x, n_iter, jnp.asarray(density.grad), _local_density(density, len(x)),
                      CFG.refine_stencil, CFG.lr * lr, np.random.default_rng(seed), progress)
    progress.done()
    return density.from_unit(x)


def adaptive_gbn(target, N: int, lr: float = 1.0, n_iter_scale: int = 6, x: np.ndarray | None = None,
                 seed: int | None = None, invert: bool = True, verbose: int = 1) -> np.ndarray:
    """
    N dots (N, 2) following a density, all O(N^2) pairs at every step (N up to ~10^4).

    lr: multiplier of config.ADAPTIVE.lr.
    n_iter_scale: passes (of steps_per_iter steps) = config.ADAPTIVE.gbn_iter_factor * n_iter_scale.
    x: starting dots (N, 2) in the target's frame, drawn from the density by default.
    """
    rng = np.random.default_rng(seed)
    density = _as_density(target, invert)
    grad, amap = jnp.asarray(density.grad), jnp.asarray(_local_density(density, N))
    sigma2, step = 1.0 / N, CFG.lr * lr
    n_steps = CFG.gbn_iter_factor * n_iter_scale * CFG.steps_per_iter
    x = _draw(density, N, rng) if x is None else density.to_unit(x).astype(np.float32)

    def one_step(_, p):
        a = interpolate(amap, p)
        level, blend = _pyramid_level(a, sigma2, grad)
        d = p[None, :, :] - p[:, None, :]                   # x_q - x_p, row p
        d = d - jnp.round(d)
        c = 2 * a[:, None] * a[None, :] / (a[:, None] + a[None, :])
        r = jnp.sum(d * d, -1) / sigma2 * c
        w = jnp.where(r < CFG.cutoff, c * c * jnp.exp(-jnp.minimum(r, CFG.cutoff)), 0.0)
        return _move(p, a, _attraction(grad, p, level, blend), jnp.sum(d * w[..., None], 1), sigma2, step)

    progress = Progress(f"[adaptive gaussian · N = {N}]", verbose=verbose)
    run = jax.jit(lambda p: jax.lax.fori_loop(0, n_steps, one_step, p))
    x = np.asarray(run(jnp.asarray(x)))
    progress.done()
    return density.from_unit(x)


# ── Target side ───────────────────────────────────────────────────────────────

class Density:
    """
    The target density rho (mean 1) on a (G, G) grid indexed [x, y], G = config.ADAPTIVE.grid_size,
    and the pyramid grad[k] = gradient of rho * g_t, t = t0 2^k, t0 = (half a pixel)^2
    (g_t: unit gaussian of variance t per axis).

    target: a path, a 2D point set (read as a blurred histogram), a `Target`, or a non-negative
    (G, G) grid. The dots live in the unit square; `to_unit` / `from_unit` map them from / to
    the target's frame.
    """

    def __init__(self, target, invert: bool = True):
        G = CFG.grid_size
        if isinstance(target, np.ndarray) and target.shape == (G, G):
            self.target = None
            rho = target.astype(np.float64)
        else:
            self.target = as_target(target, 2)
            rho = for_adaptive(self.target, (G, G), invert)
        self.rho = (rho / rho.mean()).astype(np.float32)
        self.rho_hat = np.fft.fft2(self.rho.astype(np.float64))
        f = np.fft.fftfreq(G, 1.0 / G)
        self.fx, self.fy = np.meshgrid(f, f, indexing="ij")
        f2 = self.fx ** 2 + self.fy ** 2
        self.grad = np.empty((CFG.levels, G, G, 2), np.float32)
        for k in range(CFG.levels):
            g = np.exp(-2 * np.pi ** 2 * _t0(G) * 2 ** k * f2)
            z = np.fft.ifft2(self.rho_hat * 2j * np.pi * g * (self.fx + 1j * self.fy))  # re: d/dx, im: d/dy
            self.grad[k, ..., 0], self.grad[k, ..., 1] = z.real, z.imag

    def smooth(self, t: float) -> np.ndarray:
        """rho * g_t on the grid."""
        g = np.exp(-2 * np.pi ** 2 * t * (self.fx ** 2 + self.fy ** 2))
        return np.fft.ifft2(self.rho_hat * g).real.astype(np.float32)

    def to_unit(self, x) -> np.ndarray:
        return np.asarray(x) if self.target is None else self.target.to_unit(x)

    def from_unit(self, u) -> np.ndarray:
        return u if self.target is None else self.target.from_unit(u)


def _as_density(target, invert: bool) -> Density:
    return target if isinstance(target, Density) else Density(target, invert)


def _t0(G: int) -> float:
    """Finest kernel variance of the pyramid: half a pixel of standard deviation."""
    return 0.25 / G ** 2


def _local_density(density: Density, n: int) -> np.ndarray:
    """The map of a: the density seen at the scale of the mean spacing of n dots, at least a_min."""
    return np.maximum(density.smooth(0.25 / n), CFG.a_min)


# ── One descent step ──────────────────────────────────────────────────────────

def _pyramid_level(a, sigma2, grad):
    """Level k and blend f in [0, 1) of the pyramid matching each dot's kernel (variance
    sigma^2 / (4a) per axis)."""
    n_levels, G = grad.shape[0], grad.shape[1]
    level = jnp.clip(jnp.log2(sigma2 / (4 * a) / _t0(G)), 0.0, n_levels - 1 - 1e-3)
    k = jnp.floor(level).astype(jnp.int32)
    return k, level - k


def _attraction(grad, p, k, f):
    """Gradient of the image blurred by each dot's own kernel: levels k and k + 1 blended by f."""
    idx, w = bilinear(p, grad.shape[1])
    g0 = grad[k[None], idx[..., 0], idx[..., 1]]            # (4, n, 2)
    g1 = grad[k[None] + 1, idx[..., 0], idx[..., 1]]
    return (w[..., None] * (g0 * (1 - f)[None, :, None] + g1 * f[None, :, None])).sum(0)


def _move(p, a, attraction, repulsion, sigma2, step):
    """One step, preconditioned by a^-2 (every dot moves in units of its own spacing), each move
    capped at max_move local spacings."""
    d = step / (a * a)[:, None] * (np.pi * sigma2 / 2 * attraction - repulsion)
    lim = CFG.max_move * jnp.sqrt(sigma2 / a)
    m = jnp.linalg.norm(d, axis=-1)
    d = d * jnp.minimum(1.0, lim / jnp.maximum(m, 1e-30))[:, None]
    return wrap(p + d)


# ── Grid descent (rgbn) ───────────────────────────────────────────────────────

def _grid_descent(x: np.ndarray, n_pass: int, grad, amap, stencil: int, step: float, rng,
                  progress: Progress) -> np.ndarray:
    """n_pass passes: sort the dots on a grid (gridpoints), read their kernel sizes, then run
    steps_per_iter descent steps with these frozen."""
    import gridpoints

    n = len(x)
    gi = int(np.ceil(np.sqrt(n)))
    sigma2 = 1.0 / n
    run = _grid_steps(stencil, sigma2, step)
    flat = np.full((gi * gi, 2), np.nan, np.float32)        # (gi, gi) grid, NaN = empty cell
    flat[:n] = x
    for _ in range(n_pass):
        empty = ~np.isfinite(flat).all(-1)
        flat = np.where(empty[:, None], rng.random(flat.shape), wrap(flat)).astype(np.float32)
        order = np.asarray(gridpoints.argsort(flat, (gi, gi), verbose=0, level=2))
        flat, empty = flat[order], empty[order]
        a = np.where(empty, 1.0, interpolate(amap, flat)).astype(np.float32)
        k, f = _pyramid_level(jnp.asarray(a), sigma2, grad)
        flat[empty] = np.nan
        out = run(jnp.asarray(flat.reshape(gi, gi, 2)), jnp.asarray(a.reshape(gi, gi)),
                  k.reshape(gi, gi), f.reshape(gi, gi), grad)
        flat = np.array(out).reshape(-1, 2)
        progress.tick()
    return flat[np.isfinite(flat).all(-1)]


def _grid_steps(stencil: int, sigma2: float, step: float):
    """Jitted steps_per_iter steps on a (gi, gi, 2) grid of dots, kernel sizes a and pyramid
    levels frozen."""
    shifts = jnp.asarray(half_ball(stencil ** 2, 2))
    inv_s2, cutoff = 1.0 / sigma2, CFG.cutoff

    def repulsion(x, a):
        """sum_q (x_q - x_p) c^2 exp(-c r^2 / sigma^2) over the stencil, each pair once (action / reaction)."""
        def body(acc, s):
            xq, aq = jnp.roll(x, s, (0, 1)), jnp.roll(a, s, (0, 1))
            d = xq - x
            d = d - jnp.round(d)
            c = 2 * a * aq / (a + aq)
            r = jnp.sum(d * d, -1) * inv_s2 * c
            out = jnp.where((r < cutoff)[..., None], d * (c * c * jnp.exp(-jnp.minimum(r, cutoff)))[..., None], 0.0)
            return acc + out - jnp.roll(out, -s, (0, 1)), None
        return jax.lax.scan(body, jnp.zeros_like(x), shifts)[0]

    @jax.jit
    def run(x, a, k, f, grad):
        full = jnp.isfinite(x[..., 0]).reshape(-1)
        af, kf, ff = a.reshape(-1), k.reshape(-1), f.reshape(-1)

        def one_step(_, x):
            p = jnp.where(full[:, None], x.reshape(-1, 2), 0.0)
            new = _move(p, af, _attraction(grad, p, kf, ff), repulsion(x, a).reshape(-1, 2), sigma2, step)
            return jnp.where(full[:, None], new, jnp.nan).reshape(x.shape)

        return jax.lax.fori_loop(0, CFG.steps_per_iter, one_step, x)

    return run


# ── Initial dots ──────────────────────────────────────────────────────────────

def _level_sizes(n: int) -> list[int]:
    """Sizes of the levels, coarsest first, each about 3x the previous one."""
    sizes = [n]
    while sizes[-1] > CFG.root_max_n:
        m = sizes[-1]
        sizes.append(m // 3 + m % 3)
    return sizes[::-1]


def _draw(density: Density, m: int, rng) -> np.ndarray:
    """m dots drawn from the density, by rejection."""
    G = density.rho.shape[0]
    rho_max = density.rho.max()
    out = []
    while sum(len(o) for o in out) < m:
        p = rng.random((2 * m, 2))
        c = np.minimum((p * G).astype(np.int64), G - 1)
        out.append(p[density.rho[c[:, 0], c[:, 1]] > rng.random(2 * m) * rho_max])
    return np.concatenate(out)[:m].astype(np.float32)


def _clone(parents: np.ndarray, n: int, amap: np.ndarray, rng) -> np.ndarray:
    """n // 3 parents replaced by the 3 vertices of a small randomly rotated triangle, sized by
    the local spacing; the other parents are kept as they are. Returns n dots."""
    x = rng.permutation(parents)
    n_par = n // 3
    p = x[:n_par]
    scale = CFG.expansion_factor / np.sqrt(n * interpolate(amap, p))
    theta = rng.random(n_par) * 2 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    tri = np.array([(-np.sqrt(0.75), -0.5), (np.sqrt(0.75), -0.5), (0.0, 1.0)])
    off = np.stack([c[:, None] * tri[:, 0] - s[:, None] * tri[:, 1],
                    s[:, None] * tri[:, 0] + c[:, None] * tri[:, 1]], -1) * scale[:, None, None]
    children = (p[:, None, :] + off).reshape(-1, 2)
    return wrap(np.concatenate([x[n_par:], children]).astype(np.float32))


# ── Bilinear interpolation on a (G, G) grid ───────────────────────────────────

def bilinear(p, G: int):
    """Corner indices (4, n, 2) and weights (4, n) of points p (n, 2) of the unit torus on a
    (G, G) grid whose values sit at the cell centres (numpy or jax)."""
    xp = np if isinstance(p, np.ndarray) else jnp
    u = p * G - 0.5
    f = xp.floor(u)
    a, b = (u - f)[:, 0], (u - f)[:, 1]
    i0 = f.astype(xp.int32) % G
    i1 = (i0 + 1) % G
    x0, y0, x1, y1 = i0[:, 0], i0[:, 1], i1[:, 0], i1[:, 1]
    idx = xp.stack([xp.stack([x0, y0], -1), xp.stack([x0, y1], -1),
                    xp.stack([x1, y0], -1), xp.stack([x1, y1], -1)])
    w = xp.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b])
    return idx, w


def interpolate(grid, p):
    """Bilinear interpolation of a (G, G, ...) grid at points p (n, 2)."""
    idx, w = bilinear(p, grid.shape[0])
    vals = grid[idx[..., 0], idx[..., 1]]                   # (4, n, ...)
    return (w.reshape(w.shape + (1,) * (vals.ndim - 2)) * vals).sum(0)
