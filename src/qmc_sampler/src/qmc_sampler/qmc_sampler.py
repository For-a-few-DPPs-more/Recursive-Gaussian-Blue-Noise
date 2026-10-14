"""
see https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft.py:
same but:
  - L1 ball or wd ball instead of L2 ball
  - keops backend instead of finufft
"""

from __future__ import annotations

import time
from math import pi
from typing import Optional

import numpy as np

from qmc_sampler.qmc_geom import CHI, get_wave_vectors


try:
    from pykeops.numpy import LazyTensor

    def to_numpy(x):
        return np.asarray(x)

except ImportError:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def LazyTensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

def sum0(x):
    return to_numpy(x.sum(0))

def sum1(x):
    return to_numpy(x.sum(1))

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
# Main KeOps optimisation pipeline
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
                           n_iter = n_iter, lr = lr, verbose = verbose)
    x = rng.uniform(size=(N, D)).astype(real_dtype) if warmstart is None else warmstart

    K, w, M = get_wave_vectors(N, D, mode=mode, chi=Chi, rng=rng)
    w = (1/(K*K + EPSILON).sum(axis = -1))
    k_coords = np.ascontiguousarray(K.astype(real_dtype))
    norm = float(max(M, 1))

    # optional target spectrum
    if target is not None:
        tx = LazyTensor(target[:, None, :])
        tk = LazyTensor(k_coords[None, :, :])
        phase = two_pi * (tx * tk).sum(-1)
        tr = np.asarray(sum0(phase.cos())).ravel()
        ti = -np.asarray(sum0(phase.sin())).ravel()
        fk_target = ((tr + 1j * ti) / target.shape[0]).astype(complex_dtype)
    else:
        fk_target = np.zeros(M, dtype=complex_dtype)

    fk_tr = np.ascontiguousarray(fk_target.real.astype(real_dtype))
    fk_ti = np.ascontiguousarray(fk_target.imag.astype(real_dtype))
    k_j   = LazyTensor(k_coords[None, :, :])

    def fourier_modes(pts):
        xi = LazyTensor(pts[:, None, :])
        phase = two_pi * (xi * k_j).sum(-1)
        fr = np.asarray(sum0(phase.cos())).ravel() / N
        fi = -np.asarray(sum0(phase.sin())).ravel() / N
        return fr.astype(real_dtype), fi.astype(real_dtype)

    def loss_and_grad(pts):
        fr, fi = fourier_modes(pts)
        dr = fr - fk_tr
        di = fi - fk_ti
        loss = float(np.sum(w * (dr ** 2 + di ** 2)) / norm)

        pref = 2.0 / norm * (-two_pi) / N
        cr = np.ascontiguousarray((pref * w * dr).astype(real_dtype))
        ci = np.ascontiguousarray((pref * w * di).astype(real_dtype))

        xi  = LazyTensor(pts[:, None, :])
        crj = LazyTensor(cr[None, :, None])
        cij = LazyTensor(ci[None, :, None])
        phase = two_pi * (xi * k_j).sum(-1, keepdim = True)
        scalar = crj * phase.sin() + cij * phase.cos()
        grad = sum1(scalar * k_j)
        return loss, np.asarray(grad).astype(real_dtype)

    adaptive = lr * 0.1 * N ** (-1.0 / D)
    prev_loss = np.inf

    if verbose:
        print(f"[keops-{mode}] N={N}  D={D}  modes={M}  Chi={Chi:.3f}")

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

    if verbose:
        print(f"done in {time.time() - t0:.1f}s")
        cosmode, sinmode = fourier_modes(x)
        print(f"max fourier mode = {(cosmode**2 + sinmode**2).max()*N}")

    return np.asarray(x)