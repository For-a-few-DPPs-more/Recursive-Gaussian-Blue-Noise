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
"""
from __future__ import annotations

from math import pi

import numpy as np

from .. import config
from ..array_lib import get_backend, keops_available, wrap
from ..density import image_spectrum
from ..lattice import lowest_modes
from ..progress import Progress, log

CFG = config.NUFFT


class FinufftModel:
    name = "finufft"

    def __init__(self, wv, N, D, precision, device):
        self.bk = bk = get_backend(device, precision)
        xp = bk.xp
        self.N, self.D, self.M = N, D, wv.M
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

    def image_spectrum(self, path):
        return self.bk.xp.asarray(image_spectrum(path, self.shape), dtype=self.bk.complex)

    def loss_and_grad(self, x, fk_target):
        xp = self.bk.xp
        diff = self.spectrum(x) - fk_target
        loss = float(xp.sum(xp.abs(diff) ** 2 * self.w) / self.M)
        self.type2.setpts(*self._coords(x))
        source = 2.0 * self.w * xp.conj(diff) * 2j * pi
        grad = [self.type2.execute((source * self.ks[d]).astype(self.bk.complex)).real for d in range(self.D)]
        return loss, xp.stack(grad, axis=1) / (self.N * self.M)


class JaxModel:
    name = "jax"

    def __init__(self, wv, N, D, precision, device=None):
        import jax
        import jax.numpy as jnp

        self.jnp = jnp
        self.real = jnp.float64 if precision == "float64" else jnp.float32
        cplx = jnp.complex128 if precision == "float64" else jnp.complex64
        self.N, self.M = N, wv.M
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

    def loss_and_grad(self, x, fk_target):
        loss, grad = self._loss_and_grad(x, fk_target)
        return float(loss), grad


class KeopsModel:
    name = "keops"

    def __init__(self, wv, N, D, precision, device=None):
        from pykeops.numpy import LazyTensor

        self.LazyTensor = LazyTensor
        self.real = np.float64 if precision == "float64" else np.float32
        self.N, self.M = N, wv.M
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
          seed: int | None = None, model: str = "auto", verbose: int = 1) -> np.ndarray:
    """
    N points in [0, 1)^D minimising the low-frequency structure factor (see module docstring).

    init: starting points (N, D), uniform random otherwise.
    target: point cloud (K, D), or an image path ("finufft" model only). None: uniform.
    model: "finufft", "jax", "keops" or "auto". Ctrl-C stops early and returns the current points.
    """
    model = pick_model(N, D) if model == "auto" else model
    if model == "finufft" and D > CFG.max_dim:
        raise ValueError(f"the finufft model supports D <= {CFG.max_dim} (got {D})")
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(N, D)) if init is None else np.asarray(init).reshape(N, D)
    wv = lowest_modes(N, D, chi, rng)
    m = MODELS[model](wv, N, D, precision, device)

    if target is None:
        fk_target, target_info = 0.0, "uniform"
    elif isinstance(target, str):
        if model != "finufft":
            raise NotImplementedError(f"image targets are only supported by the finufft model (D <= {CFG.max_dim})")
        fk_target, target_info = m.image_spectrum(target), "image target"
    else:
        target = np.asarray(target)
        if target.ndim != 2 or target.shape[1] != D:
            raise ValueError(f"target must have shape (K, {D}), got {target.shape}")
        fk_target, target_info = m.spectrum(target), f"target = {len(target):,} points"

    log(f"[nufft · {m.label}] N={N:,} D={D} chi={chi:.3f} modes={wv.M:,} ({target_info}) — Ctrl-C stops early",
        verbose)
    if model != "finufft":
        log("  dense O(N M) model: fine for N <= 2k-4k on CPU, 10k-20k on GPU", verbose)

    x = m.asarray(x)
    step = lr * CFG.initial_step * N ** (-1.0 / D)
    best = float("inf")
    bar = Progress("[nufft]", total=n_iter, verbose=verbose)
    try:
        for _ in range(n_iter):
            loss, grad = m.loss_and_grad(x, fk_target)
            x = wrap(x - grad / ((grad * grad).mean() + 1e-30) ** 0.5 * step)
            if loss < best:
                best, step = loss, step * CFG.step_increase
            else:
                step *= CFG.step_decrease
            bar.tick(f"loss {loss:.1e}")
    except KeyboardInterrupt:
        log(f"\n[nufft] interrupted after {bar.count} iterations", verbose)
    bar.done()
    return m.to_numpy(x)
