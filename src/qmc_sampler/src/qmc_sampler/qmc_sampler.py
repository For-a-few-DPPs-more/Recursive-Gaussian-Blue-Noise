"""
see https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft.py:
same but:
  - L1 ball instead of L2 ball
  - keops backend instead of finufft
  - "wd" mode: gradient descent on the *exact* WD discrepancy (no Fourier
    truncation), warm-started from the L1 (chi=0.5) optimum

Common API: every mode provides a `loss_and_grad(points) -> (loss, grad)` and
`minimize` runs the same adaptive normalised-gradient / bold-driver descent.

There is no .prod() over the feature axis in KeOps, hence the products over
dimensions are written as exp(sum(log(.))).
"""

from __future__ import annotations

import time
from math import pi
from typing import Callable, Optional, Tuple

import numpy as np

from .qmc_geom import CHI, get_wave_vectors


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

# Max number of elements of one (block, N, D) temporary in the torch fallback
# (KeOps never materialises it, so it processes all rows at once).
_TORCH_MAX_ELEMS = 2e7


try:
    from pykeops.numpy import LazyTensor

    HAS_KEOPS = True

    def to_numpy(x):
        return np.asarray(x)

except ImportError:
    import torch

    HAS_KEOPS = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _torch_dtype = torch.float32 if PRECISION == "float32" else torch.float64

    def LazyTensor(x):
        return torch.as_tensor(x, dtype=_torch_dtype, device=device)

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

def sum0(x):
    return to_numpy(x.sum(0))

def sum1(x):
    return to_numpy(x.sum(1))


LossAndGrad = Callable[[np.ndarray], Tuple[float, np.ndarray]]


# ------------------------------------------------------------------
# Wrappers (for convenience)
# ------------------------------------------------------------------
def sample_l1(N, D):
    return sample(N, D, "l1", verbose=0, strict = False)

def sample_wd(N, D):
    return sample(N, D, "wd", verbose=0, strict = True)


# ------------------------------------------------------------------
# Common optimiser
# ------------------------------------------------------------------
def minimize(
    x: np.ndarray,
    loss_and_grad: LossAndGrad,
    n_iter: int = NITER,
    lr: float = LR,
    verbose: int = 1,
    strict: bool = False,
) -> np.ndarray:
    """
    Adaptive normalised-gradient descent on the torus (bold-driver step).

    loss_and_grad(x) must return (float loss, grad array with the shape of x).
    strict=False reproduces the historical behaviour (always move, only the
    step size reacts to a loss increase); strict=True returns to the best
    point seen so far (with a smaller step) when the loss does not decrease.
    """
    N, D = x.shape
    adaptive = lr * 0.1 * N ** (-1.0 / D)
    prev_loss = np.inf
    x_best = g_best = None

    t0 = time.time()
    for it in range(n_iter):
        loss, grad = loss_and_grad(x)
        improved = loss < prev_loss

        if strict and not improved and x_best is not None:
            # reject: go back to the best point with a smaller step
            adaptive *= 0.8
            x, grad = x_best, g_best

        rms = float(np.sqrt(np.mean(grad * grad) + 1e-30))
        step = (grad / rms) * adaptive
        x_new = np.mod(x - step, 1.0).astype(real_dtype)

        # bold-driver
        if improved:
            adaptive *= 1.05
            prev_loss = loss
            x_best, g_best = x, grad
        elif not strict:
            adaptive *= 0.8
        # historical behaviour: always move, only the step size adapts
        x = x_new

        if verbose and (it % 20 == 0 or it == n_iter - 1):
            print(f"  iter {it:4d} | loss {loss:.3e}")

    if verbose:
        print(f"done in {time.time() - t0:.1f}s")

    return np.asarray(x)


# ------------------------------------------------------------------
# Loss / gradient factory: truncated Fourier energy on an L1 ball
# ------------------------------------------------------------------
def make_l1_loss_and_grad(
    N: int,
    D: int,
    chi: float,
    rng: np.random.Generator,
    target: Optional[np.ndarray] = None,
):
    K, kl1, w, M = get_wave_vectors(N, D, mode="l1", chi=chi, rng=rng)
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
        phase = two_pi * (xi * k_j).sum(-1, keepdim=True)
        scalar = crj * phase.sin() + cij * phase.cos()
        grad = sum1(scalar * k_j)
        return loss, np.asarray(grad).astype(real_dtype)

    return loss_and_grad, M


# ------------------------------------------------------------------
# Loss / gradient factory: exact WD discrepancy
# ------------------------------------------------------------------
def make_wd_loss_and_grad(N: int, D: int, block: Optional[int] = None):
    """
    Exact squared wrap-around discrepancy and its gradient.

        WD^2(x) = -(4/3)^D + 1/N^2 sum_{i,j} P_ij,
        P_ij    = prod_d g(x_id - x_jd),     g(delta) = 3/2 - |delta| + delta^2.

    g >= 5/4 > 0, so P_ij = exp( sum_d log g_ijd ): a plain `.sum(-1)` is all
    that is needed (no product over the feature axis).

    Logarithmic derivative:  d log P_ij / d x_id = g'(delta_ijd) / g_ijd with
    g'(delta) = 2 delta - sign(delta)  (odd, = 0 on the diagonal i=j), hence

        dWD^2/dx_id = 2/N^2 * sum_j  P_ij * g'(delta_ijd) / g_ijd

    (the factor 2 comes from the symmetry P_ij = P_ji). Cost O(N^2 D), versus
    O(N^2 D^2) for "product of the other coordinates".

    The loss is accumulated as sum(P_ij - (4/3)^D), which keeps the row sums
    small and avoids most of the float32 cancellation against (4/3)^D.
    """
    const = (4.0 / 3.0) ** D
    inv_n2 = 1.0 / float(N) ** 2
    if block is None:
        block = N if HAS_KEOPS else max(1, int(_TORCH_MAX_ELEMS // (N * D)))
    block = min(block, N)

    def loss_and_grad(pts):
        xj = LazyTensor(pts[None, :, :])
        grad = np.empty((N, D), dtype=real_dtype)
        total = 0.0
        for i0 in range(0, N, block):
            i1 = min(i0 + block, N)
            xi = LazyTensor(pts[i0:i1, None, :])

            delta = xi - xj                              # (block, N, D)
            t = delta.abs()
            g = t * t - t + 1.5                          # 3/2 - |d| + d^2
            logP = g.log().sum(-1, keepdim=True)         # (block, N, 1)
            P = logP.exp()

            total += float(np.sum(sum1(P - const), dtype=np.float64))

            dg = 2.0 * delta - delta.sign()              # g'(delta)
            grad[i0:i1] = 2.0 * inv_n2 * sum1(P * dg / g)

        return total * inv_n2, grad

    return loss_and_grad


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def sample(
    N,
    D,
    mode,
    chi: float = CHI,
    target: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    n_iter: int = NITER,
    lr: float = LR,
    verbose: int = 1,
    warmstart: Optional[np.ndarray] = None,
    strict: bool = False,
    block: Optional[int] = None,
):
    """
    Generate a low-discrepancy point set on [0,1)^D by optimisation.

    Parameters
    ----------
    N        : number of points
    D        : dimension
    mode     : "l1"  -> truncated Fourier energy on a half L1 ball
               "wd"  -> exact WD discrepancy, warm-started from "l1"
                        (chi=0.5, same n_iter, same seed) unless `warmstart`
                        is given
    chi      : fraction of degrees of freedom kept as Fourier modes ("l1" only)
    target   : optional target point cloud for non-uniform densities ("l1" only)
    seed     : random seed
    n_iter   : number of gradient steps (for each stage)
    lr       : base learning-rate scale
    verbose  : print progress
    warmstart: optional warm start for the point cloud, eg a Sobol sequence
    strict   : reject steps that do not decrease the loss
    block    : rows per block in the "wd" kernel (only matters for torch)

    Returns
    -------
    x : (N, D) array of points in [0,1)^D
    """
    if mode not in ("l1", "wd"):
        raise ValueError(f"Unknown mode '{mode}'. Choose 'l1' or 'wd'.")
    if mode == "wd" and target is not None:
        raise ValueError("mode='wd' only supports the uniform measure (target=None).")

    rng = np.random.default_rng(seed)
    if warmstart is None and mode == "wd":
        warmstart = sample(N, D, "l1", chi=0.5, target=target, seed=seed,
                           n_iter=n_iter, lr=lr, verbose=verbose, strict=strict)
    x = rng.uniform(size=(N, D)).astype(real_dtype) if warmstart is None \
        else np.ascontiguousarray(warmstart, dtype=real_dtype)

    if mode == "l1":
        loss_and_grad, M = make_l1_loss_and_grad(N, D, chi, rng, target)
        if verbose:
            print(f"[keops-l1] N={N}  D={D}  modes={M}  chi={chi:.3f}")
    else:
        loss_and_grad = make_wd_loss_and_grad(N, D, block=block)
        if verbose:
            print(f"[keops-wd] N={N}  D={D}  exact WD^2")

    return minimize(x, loss_and_grad, n_iter=n_iter, lr=lr,
                    verbose=verbose, strict=strict)