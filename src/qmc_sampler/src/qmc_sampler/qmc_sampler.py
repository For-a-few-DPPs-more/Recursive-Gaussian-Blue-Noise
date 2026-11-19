"""
see https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft.py:
same but:
  - L1 ball or wd ball instead of only L2 ball
  - keops / torch backends (exact direct sums), finufft optional in D <= 3
  - quasi newton finition (optional)
"""

from __future__ import annotations

import time
from math import pi
from typing import Optional

import numpy as np

from qmc_sampler._backend import get_backend, nudft1, nudft2_real
from qmc_sampler.qmc_geom import CHI, get_wave_vectors

# ------------------------------------------------------------------
# Global defaults
# ------------------------------------------------------------------
EPSILON   = 1e-12
PRECISION = "float32"
NITER     = 150
LR        = 1.0

real_dtype    = np.float32 if PRECISION == "float32" else np.float64
complex_dtype = np.complex64 if PRECISION == "float32" else np.complex128
two_pi        = 2.0 * pi


# ------------------------------------------------------------------
# Wrapers (for conveniance)
# ------------------------------------------------------------------
def sample_l1(N, D, **kwargs):
    kwargs.setdefault("verbose", 0)
    return sample(N, D, "l1", **kwargs)

def sample_l2(N, D, **kwargs):
    kwargs.setdefault("verbose", 0)
    return sample(N, D, "l2", **kwargs)

def sample_wd(N, D, **kwargs):
    kwargs.setdefault("verbose", 0)
    return sample(N, D, "wd", **kwargs)

# ------------------------------------------------------------------
# Main optimisation pipeline (backend: see qmc_sampler.select_backend)
# ------------------------------------------------------------------
def sample(
    N,
    D,
    mode,
    Chi: float = CHI,
    target: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    n_iter: int = NITER,
    lr: float = LR,
    verbose: int = 1,
    warmstart: Optional[np.ndarray] = None,
    quasi_newton_finition: bool = True,
):
    """
    Generate a low-discrepancy point set on [0,1)^D by spectral optimisation.

    Parameters
    ----------
    N        : number of points
    D        : dimension
    mode     : "wd" (WD product weights), "l1" or "l2"
    Chi      : fraction of degrees of freedom kept as Fourier modes
    target   : optional target point cloud (for non-uniform densities)
    seed     : random seed
    n_iter   : number of gradient steps
    lr       : base learning-rate scale
    verbose  : print progress
    warmstart: optional warm start for the point cloud, eg a Sobol sequence

    Returns
    -------
    x : (N, D) array of points in [0,1)^D
    """
    rng = np.random.default_rng(seed)
    if warmstart is None and mode == "wd":
        warmstart = sample(N, D, "l1", Chi = 0.5, target = target, seed = seed, 
                           n_iter = n_iter, lr = lr, verbose = verbose, quasi_newton_finition = False)
    x = rng.uniform(size=(N, D)).astype(real_dtype) if warmstart is None else warmstart

    K, M = get_wave_vectors(N, D, mode=mode, chi=Chi, rng=rng)
    w = (1/(K*K + EPSILON).sum(axis = -1))
    norm = float(max(M, 1))

    backend = get_backend(D)
    K_int = np.asarray(K)
    K_real = K_int.astype(real_dtype)

    # optional target spectrum: f_target(k) = (1/Nt) sum_j exp(-2i pi k.y_j)
    if target is not None:
        tgt = np.asarray(target, dtype=real_dtype)
        fk_target = (nudft1(tgt, None, K_int, isign=-1, backend=backend)
                     / tgt.shape[0]).astype(complex_dtype)
    else:
        fk_target = np.zeros(M, dtype=complex_dtype)

    fk_tr = fk_target.real.astype(real_dtype)
    fk_ti = fk_target.imag.astype(real_dtype)

    def fourier_modes(pts):
        # f(k) = (1/N) sum_j exp(-2i pi k.x_j)
        f = nudft1(pts, None, K_int, isign=-1, backend=backend) / N
        return f.real.astype(real_dtype), f.imag.astype(real_dtype)

    def loss_and_grad(pts):
        fr, fi = fourier_modes(pts)
        dr = fr - fk_tr
        di = fi - fk_ti
        loss = float(np.sum(w * (dr ** 2 + di ** 2)) / norm)

        pref = 2.0 / norm * (-two_pi) / N
        cr = pref * w * dr
        ci = pref * w * di
        # grad_j = sum_k k (cr_k sin(2 pi k.x_j) + ci_k cos(2 pi k.x_j))
        #        = Re sum_k (ci_k - i cr_k) k exp(+2i pi k.x_j)
        a = (ci - 1j * cr)[:, None] * K_real
        grad = nudft2_real(pts, a, K_int, isign=1, backend=backend)
        return loss, np.asarray(grad).astype(real_dtype)

    adaptive = lr * 0.1 * N ** (-1.0 / D)
    prev_loss = np.inf

    if verbose:
        print(f"[{backend}-{mode}] N={N}  D={D}  modes={M}  Chi={Chi:.3f}")

    t0 = time.time()
    for it in range(n_iter):
        loss, grad = loss_and_grad(x)
        rms = float(np.sqrt(np.mean(grad * grad) + 1e-30))
        step = (grad / rms) * adaptive
        x_new = np.mod(x - step, 1.0).astype(real_dtype)

        # bold-driver: accept only if loss decreases
        if loss < prev_loss:
            x = x_new
            adaptive *= 1.05
            prev_loss = loss
        else:
            adaptive *= 0.8
            x = x_new   # still move (comment this line for strict rejection)

        if verbose and (it % 20 == 0 or it == n_iter - 1):
            print(f"  iter {it:4d} | loss {loss:.3e}")
        # ------------------------------------------------------------------
    # Finition L-BFGS (quasi-Newton) :
    # ------------------------------------------------------------------
    if quasi_newton_finition:
        from scipy.optimize import minimize

        loss0 = loss

        def fun(flat):
            pts = np.asarray(flat, dtype=real_dtype).reshape(N, D)
            l, g = loss_and_grad(pts)
            return l/loss0, g.ravel()/loss0

        t1 = time.time()
        res = minimize(
            fun,
            x.ravel(), 
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": n_iter, "maxcor": 30, "maxfun": 2 * n_iter, "ftol": 0, "gtol": 0, "maxfun": 10*n_iter},
        )

        x = np.mod(res.x.reshape(N, D), 1.0).astype(real_dtype)

        if verbose:
            print(f"[L-BFGS] {res.nit} it, {res.nfev} evals | "
                  f"loss {res.fun:.3e} | {time.time() - t1:.1f}s")

    if verbose:
        print(f"done in {time.time() - t0:.1f}s")
        cosmode, sinmode = fourier_modes(x)
        print(f"max fourier mode = {(cosmode**2 + sinmode**2).max()*N}")

    return np.asarray(x)
