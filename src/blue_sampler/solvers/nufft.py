"""
Spectral blue noise: gradient descent of the structure-factor energy

    L(x) = 1/M sum_k w_k |f(k) - f_target(k)|^2,    f(k) = 1/N sum_j exp(i 2 pi k.x_j),

over the M = round(chi D (N-1)) lowest wave vectors (w_k ~ 1/|k|^2). chi is the fraction
of the degrees of freedom constrained: <= 0.3 fast, ~0.4 balanced, >= 0.5 may crystallise.

Three interchangeable models evaluate (L, grad L):
- "finufft": type-1 / type-2 NUFFTs, O(N log N), D <= 3, CPU (finufft) or GPU (cufinufft)
- "jax":     dense sums over the selected modes, O(N M), any D
- "keops":   same with KeOps reductions (no N x M matrix in memory), any D
The steps are RMS-normalised and their length adapted by a "bold driver" rule.
With newton_finition=True, the second half of the n_iter budget is spent on an L-BFGS
(scipy L-BFGS-B) descent started from the gradient-descent result.
"""
from __future__ import annotations

from math import pi

import numpy as np

from .. import config
from ..array_lib import get_backend, keops_available, wrap
from ..lattice import lowest_modes
from ..progress import Progress, log
from ..targets import as_target, for_nufft

CFG = config.NUFFT

# L-BFGS finish: history size, max evaluations per iteration
LBFGS_MEMORY = 30
LBFGS_MAXFUN_FACTOR = 10


class FinufftModel:
    """Spectra on a (G,)^D grid of modes, FFT ordering; convention exp(+2i pi k.x)."""
    name = "finufft"
    sign = 1

    def __init__(self, wv, N, D, precision, device):
        self.bk = bk = get_backend(device, precision)
        xp = bk.xp
        self.N, self.D, self.M = N, D, wv.M
        self.modes = wv.k
        G = 2 * (int(np.abs(wv.k).max()) + 1)                 # smallest even grid holding every mode
        self.shape = (G,) * D
        freqs = xp.asarray(np.fft.fftfreq(G, d=1.0 / G), dtype=bk.real)   # FFT ordering
        self.ks = xp.meshgrid(*[freqs] * D, indexing="ij")
        w = np.zeros(self.shape)
        w[tuple(wv.k.T.astype(np.int64) % G)] = wv.w                 # weights: non-zero on the M modes only
        self.w = xp.asarray(w, dtype=bk.real)
        kw = dict(n_trans=1, eps=CFG.eps[bk.precision], isign=1, dtype=bk.complex, modeord=1)
        self.type1 = bk.nufft.Plan(1, self.shape, **kw)
        self.type2 = bk.nufft.Plan(2, self.shape, **kw)
        self.label = f"finufft · {bk}"

    def asarray(self, x):
        return self.bk.xp.asarray(x, dtype=self.bk.real)

    def to_numpy(self, x):
        return self.bk.to_numpy(x)

    def _coords(self, x):
        return tuple(2 * pi * x[:, d] for d in range(self.D))

    def spectrum(self, points):
        points = self.asarray(points)
        self.type1.setpts(*self._coords(points))
        return self.type1.execute(self.bk.xp.ones(len(points), dtype=self.bk.complex)) / len(points)

    def from_modes(self, values):
        """Values on the M modes -> the (G,)^D grid (zero elsewhere: those modes have no weight)."""
        grid = np.zeros(self.shape, dtype=np.complex128)
        grid[tuple(self.modes.T.astype(np.int64) % self.shape[0])] = values
        return self.bk.xp.asarray(grid, dtype=self.bk.complex)

    def loss_and_grad(self, x, fk_target):
        xp = self.bk.xp
        diff = self.spectrum(x) - fk_target
        loss = float(xp.sum(xp.abs(diff) ** 2 * self.w) / self.M)
        self.type2.setpts(*self._coords(x))
        source = 2.0 * self.w * xp.conj(diff) * 2j * pi
        grad = [self.type2.execute((source * self.ks[d]).astype(self.bk.complex)).real for d in range(self.D)]
        return loss, xp.stack(grad, axis=1) / (self.N * self.M)


class JaxModel:
    """Spectra on the M modes; convention exp(-2i pi k.x)."""
    name = "jax"
    sign = -1

    def __init__(self, wv, N, D, precision, device=None):
        import jax
        import jax.numpy as jnp

        self.jnp = jnp
        self.real = jnp.float64 if precision == "float64" else jnp.float32
        cplx = jnp.complex128 if precision == "float64" else jnp.complex64
        self.cplx = cplx
        self.N, self.D, self.M = N, D, wv.M
        self.modes = wv.k
        k_max = int(np.abs(wv.k).max())
        k = jnp.asarray(wv.k, dtype=self.real)
        w = jnp.asarray(wv.w, dtype=self.real)
        k_pos = jnp.arange(1, k_max + 1, dtype=self.real)
        idx = jnp.clip(jnp.asarray(np.abs(wv.k), dtype=jnp.int32) - 1, 0, k_max - 1)
        sign = jnp.asarray(np.sign(wv.k), dtype=jnp.int8)
        minus_2i_pi = jnp.asarray(-2.0j * pi, dtype=cplx)
        self.k = k

        def modes(x):
            """e[m, n] = exp(-2i pi k_m.x_n), from the 1D factors exp(-2i pi j x_nd), j <= k_max."""
            factors_1d = jnp.exp(minus_2i_pi * x[:, :, None] * k_pos[None, None, :]).astype(cplx)
            f = jnp.take_along_axis(factors_1d[None], idx[:, None, :, None], axis=3)[..., 0]
            f = jnp.where(sign[:, None, :] < 0, jnp.conj(f), f)
            f = jnp.where(sign[:, None, :] == 0, jnp.ones((), dtype=cplx), f)
            return jnp.prod(f, axis=2)

        def loss_and_grad(x, fk_target):
            e = modes(x)
            diff = e.mean(axis=1) - fk_target
            loss = jnp.sum(w * jnp.abs(diff) ** 2) / self.M
            coeff = (2.0 / self.M) * w * jnp.conj(diff) * minus_2i_pi
            grad = jnp.real(k.T.astype(cplx) @ (coeff[:, None] * e)).T / N
            return loss, grad

        self._loss_and_grad = jax.jit(loss_and_grad)
        self._spectrum = jax.jit(lambda t: jax.vmap(lambda kk: jnp.exp(-2j * pi * (t @ kk)).mean())(k))
        self.label = f"jax · {jax.default_backend()}"

    def asarray(self, x):
        return self.jnp.asarray(x, dtype=self.real)

    def to_numpy(self, x):
        return np.asarray(x)

    def spectrum(self, points):
        return self._spectrum(self.asarray(points))

    def from_modes(self, values):
        return self.jnp.asarray(values, dtype=self.cplx)

    def loss_and_grad(self, x, fk_target):
        loss, grad = self._loss_and_grad(x, fk_target)
        return float(loss), grad


class KeopsModel:
    """Spectra on the M modes; convention exp(-2i pi k.x)."""
    name = "keops"
    sign = -1

    def __init__(self, wv, N, D, precision, device=None):
        from pykeops.numpy import LazyTensor

        self.LazyTensor = LazyTensor
        self.real = np.float64 if precision == "float64" else np.float32
        self.N, self.D, self.M = N, D, wv.M
        self.modes = wv.k
        self.k = np.ascontiguousarray(wv.k.astype(self.real))
        self.w = wv.w.astype(self.real)
        self.k_j = LazyTensor(self.k[None, :, :])
        self.label = "keops"

    def asarray(self, x):
        return np.ascontiguousarray(np.asarray(x, dtype=self.real))

    def to_numpy(self, x):
        return np.asarray(x)

    def _phase(self, x):
        # IMPORTANT: plain python float (a 0-d numpy array breaks LazyTensor.__rmul__)
        return float(2.0 * pi) * (self.LazyTensor(x[:, None, :]) * self.k_j).sum(-1)

    def spectrum(self, points):
        """exp(-2i pi k.x) averaged over the points."""
        points = self.asarray(points)
        phase = self._phase(points)
        re = np.asarray(phase.cos().sum_reduction(axis=0)).ravel()
        im = -np.asarray(phase.sin().sum_reduction(axis=0)).ravel()
        return (re + 1j * im) / len(points)

    def from_modes(self, values):
        return np.asarray(values, dtype=np.complex128)

    def loss_and_grad(self, x, fk_target):
        diff = self.spectrum(x) - fk_target
        loss = float(np.sum(self.w * np.abs(diff) ** 2) / self.M)
        pref = -2.0 * 2.0 * pi / (self.M * self.N)
        c_re = np.ascontiguousarray((pref * self.w * diff.real).astype(self.real))
        c_im = np.ascontiguousarray((pref * self.w * diff.imag).astype(self.real))
        phase = self._phase(x)
        scalar = (self.LazyTensor(c_re[None, :, None]) * phase.sin()
                  + self.LazyTensor(c_im[None, :, None]) * phase.cos())
        return loss, np.asarray((scalar * self.k_j).sum_reduction(axis=1)).astype(self.real)


MODELS = {"finufft": FinufftModel, "jax": JaxModel, "keops": KeopsModel}


def pick_model(N: int, D: int) -> str:
    if D <= CFG.max_dim:
        return "finufft"
    if N >= config.SAMPLE_POINTS.keops_min_n and keops_available():
        return "keops"
    return "jax"


def nufft(N: int = 10_000, D: int = 2, *, n_iter: int = 120, chi: float = 0.4, lr: float = 1.0,
          init=None, target=None, precision: str = "float32", device: str = "auto",
          seed: int | None = None, model: str = "auto", newton_finition: bool = False,
          verbose: int = 1) -> np.ndarray:
    """
    N points in [0, 1)^D minimising the low-frequency structure factor (see module docstring).

    init: starting points (N, D), uniform random otherwise.
    target: image path (D = 2), point set (K, D) or `targets.Target`, see `blue_sampler.targets`.
            None: uniform. The points are returned in the target's frame (see `Target.from_unit`).
    model: "finufft", "jax", "keops" or "auto". Ctrl-C stops early and returns the current points.
    newton_finition: spend the second half of n_iter on an L-BFGS descent (quasi-Newton finish).
    """
    model = pick_model(N, D) if model == "auto" else model
    target = None if target is None else as_target(target, D)
    if target is not None and init is not None:
        init = target.to_unit(init)                       # warm start given in the target's frame
    if model == "finufft" and D > CFG.max_dim:
        raise ValueError(f"the finufft model supports D <= {CFG.max_dim} (got {D})")
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(N, D)) if init is None else np.asarray(init).reshape(N, D)
    wv = lowest_modes(N, D, chi, rng)
    m = MODELS[model](wv, N, D, precision, device)

    fk_target = for_nufft(target, m)
    target_info = "uniform" if target is None else repr(target)

    log(f"[nufft · {m.label}] N={N:,} D={D} chi={chi:.3f} modes={wv.M:,} ({target_info}) — Ctrl-C stops early",
        verbose)
    if model != "finufft":
        log("  dense O(N M) model: fine for N <= 2k-4k on CPU, 10k-20k on GPU", verbose)

    x = m.asarray(x)
    n_lbfgs = n_iter // 2 if newton_finition else 0
    step = lr * CFG.initial_step * N ** (-1.0 / D)
    best = loss = float("inf")
    bar = Progress("[nufft]", total=n_iter, verbose=verbose)
    state = {}                                              # last L-BFGS iterate, kept for Ctrl-C
    try:
        for _ in range(n_iter - n_lbfgs):
            loss, grad = m.loss_and_grad(x, fk_target)
            x = wrap(x - grad / ((grad * grad).mean() + 1e-30) ** 0.5 * step)
            if loss < best:
                best, step = loss, step * CFG.step_increase
            else:
                step *= CFG.step_decrease
            bar.tick(f"loss {loss:.1e}")
        if n_lbfgs:
            x = _lbfgs(m, x, fk_target, n_lbfgs, loss, bar, state)
    except KeyboardInterrupt:
        log(f"\n[nufft] interrupted after {bar.count} iterations", verbose)
        x = state.get("x", x)
    bar.done()
    x = m.to_numpy(x)
    return x if target is None else target.from_unit(x)


def _lbfgs(m, x, fk_target, n_iter, loss0, bar, state):
    """L-BFGS-B (scipy) from x, the loss rescaled by its starting value."""
    from scipy.optimize import minimize

    N, D = x.shape
    scale = loss0 if np.isfinite(loss0) and loss0 > 0 else 1.0

    def fun(flat):
        loss, grad = m.loss_and_grad(wrap(m.asarray(flat.reshape(N, D))), fk_target)
        state["loss"] = loss
        return loss / scale, np.asarray(m.to_numpy(grad), dtype=np.float64).ravel() / scale

    def callback(xk):
        state["x"] = wrap(m.asarray(xk.reshape(N, D)))
        bar.tick(f"loss {state['loss']:.1e} (L-BFGS)")

    res = minimize(fun, np.asarray(m.to_numpy(x), dtype=np.float64).ravel(), jac=True,
                   method="L-BFGS-B", callback=callback,
                   options={"maxiter": n_iter, "maxcor": LBFGS_MEMORY,
                            "maxfun": LBFGS_MAXFUN_FACTOR * n_iter, "ftol": 0.0, "gtol": 0.0})
    return wrap(m.asarray(res.x.reshape(N, D)))