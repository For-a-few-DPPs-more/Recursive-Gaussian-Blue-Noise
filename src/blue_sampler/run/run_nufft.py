"""
hyperuniform point cloud sampling through spectral 
optimisation. To spread the points uniformly
(possibly according to a given target distribution),
we compute a spectral lost, accelerated with finufft 
fast fourier transform, and perform gradient descent.
"""


import time
from math import pi
from typing import NamedTuple

import numpy as np
from ..gpu_setup import set_config  # builds the compute "config" (CPU/GPU, float32/64, NUFFT backend)


# =============================================================================
# Wave-vector selection (D = 1, 2, 3 only)
# =============================================================================
# We optimise point positions through a finite set of Fourier modes k ∈ ℤ^D.
# Because fk(-k) = conj(fk(k)), we keep only ONE representative per ±k pair
# ("half-space"). We pick exactly M = round(Chi * D * (N - 1)) such modes:
# the M with the smallest |k|² (ties broken at random).
# Modes are weighted by 1/|k|² in the loss (low frequencies matter most).


class WaveVectors(NamedTuple):
    """Container for the selected Fourier modes."""
    k: np.ndarray        # (M, D) int32 – one representative per ±k pair
    r2: np.ndarray       # (M,)   int64 – |k|²
    w: np.ndarray        # (M,)   float64 – loss weights in (0, 1], max = 1
    M: int               # number of independent modes
    chi_eff: float       # M / (D*(N-1)) ≈ Chi


def _half_space_mask(k):
    """
    Boolean mask: True where the first non-zero coordinate of k is > 0.
    Guarantees exactly one representative per ±k pair (origin is excluded).
    """
    pos = np.zeros(len(k), dtype=bool)
    decided = np.zeros(len(k), dtype=bool)
    for d in range(k.shape[1]):
        nz = ~decided & (k[:, d] != 0)
        pos |= nz & (k[:, d] > 0)
        decided |= nz
    return pos


def get_wave_vectors(N, D, Chi, rng=None):
    """
    Select the M lowest-frequency Fourier modes.

    Simple strategy
    ---------------
    1. Choose a covering cube [-R, R]^D large enough to contain at least M
       half-space modes (R estimated from the volume of the unit ball).
    2. Generate every integer wave-vector inside that cube.
    3. Keep only the half-space representatives (one k per ±k pair).
    4. Sort by ||k - ε|| where ε is a small random offset (breaks ties
       smoothly) and keep the first M.
    5. Assign weights w = 1/|k|² (normalised to max=1).
    """
    rng = np.random.default_rng() if rng is None else rng

    n_eff = N - 1                       # k=0 excluded → N-1 degrees of freedom
    M = int(round(Chi * D * n_eff))
    if M < 1:
        raise ValueError(f"Chi={Chi} gives no mode for N={N}, D={D}")

    # Volume of the unit ball → estimate a covering radius R
    vol = {1: 2.0, 2: pi, 3: 4.0 * pi / 3.0}[D]
    R = int(np.ceil((2.0 * M / vol) ** (1.0 / D))) + 2   # +2 safety margin

    # Generate the whole integer cube [-R … R]^D
    axis = np.arange(-R, R + 1, dtype=np.int32)
    grids = np.meshgrid(*([axis] * D), indexing="ij")
    k = np.stack([g.ravel() for g in grids], axis=1)

    # Half-space only (origin automatically dropped)
    keep = _half_space_mask(k)
    k = k[keep]

    # If the cube is still too small (rare), enlarge once more
    if len(k) < M:
        R = R + max(2, R // 5)
        axis = np.arange(-R, R + 1, dtype=np.int32)
        grids = np.meshgrid(*([axis] * D), indexing="ij")
        k = np.stack([g.ravel() for g in grids], axis=1)
        k = k[_half_space_mask(k)]

    # Sort by ||k - ε||² where ε is a tiny random offset → random tie-breaking
    eps = rng.uniform(-0.5, 0.5, size=k.shape)
    score = np.sum((k.astype(np.float64) - eps) ** 2, axis=1)
    order = np.argsort(score)[:M]
    k = k[order]

    r2 = np.sum(k.astype(np.int64) ** 2, axis=1)

    # Weights inversely proportional to |k|² (low frequencies prioritised)
    w = 1.0 / (r2.astype(np.float64) + 1e-3)
    w /= w.max()

    chi_eff = M / (D * n_eff)
    return WaveVectors(k=k, r2=r2, w=w, M=M, chi_eff=chi_eff)


# =============================================================================
# NUFFT optimisation
# =============================================================================

def _nufft_pipeline(N=10_000, D=2, lr=1.0, warmstart=None, Chi=0.4, target=None,
                    n_iter=120, precision="float32", device="auto", seed=None,
                    verbose=1):
    """
    NUFFT-based optimisation that produces low-discrepancy points in [0,1)^D.

    Principle
    ---------
    Minimise the Fourier energy
        L(x) = (1/M) Σ_k  w_k |f(k) − f_target(k)|²
    where f(k) = Σ_j exp(i 2π k·x_j) is the structure factor.
    Gradients are evaluated in O(N log N) via type-1 / type-2 NUFFTs.

    Parameters
    ----------
    N : int
        Number of points.
    D : int
        Dimension (1, 2 or 3 only).
    lr : float
        Scale of the initial step ≈ lr · 0.1 · N^(−1/D).
    warmstart : array (N, D) or None
        Starting configuration (otherwise uniform random).
    Chi : float
        Fraction of degrees of freedom constrained (number of retained modes).
        ≤ 0.3 → low frequencies only (fast)
        ≈ 0.4 → balanced (default)
        ≥ 0.5 → may start crystallising
    target : array or str, optional
        Target density (point cloud or image path). None → uniform.
    n_iter : int
        Maximum gradient-descent iterations.
    precision : {"float32", "float64"}
        float64 is slower but can reach ~10⁻²⁰.
    device : {"cpu", "cuda", "auto"}
    seed : int or None
    verbose : int
        >0 prints progress.

    Returns
    -------
    x : ndarray (N, D)
        Optimised positions in [0,1)^D.
    """
    if D not in (1, 2, 3):
        raise ValueError(f"Only D=1,2,3 supported (got {D})")

    cfg = set_config(device, precision, verbose)

    xp = cfg.xp
    real_dtype = cfg.real_dtype
    complex_dtype = cfg.complex_dtype
    device = cfg.device
    precision = cfg.precision
    to_numpy = cfg.to_numpy
    nufft_lib = cfg.nufft_lib

    eps = 1e-4 if precision == "float32" else 1e-8
    rng = np.random.default_rng(seed)

    # Initial positions
    if warmstart is not None:
        x = xp.asarray(warmstart, dtype=real_dtype).reshape(N, D)
    else:
        x = xp.asarray(rng.uniform(size=(N, D)), dtype=real_dtype)

    # Mode selection
    wv = get_wave_vectors(N, D, Chi, rng)
    M, chi_eff = wv.M, wv.chi_eff

    # Smallest even FFT grid that holds every selected mode
    K_max = int(np.abs(wv.k).max())
    G = 2 * (K_max + 1)
    n_modes = (G,) * D

    # Frequencies in FFT order: 0, 1, …, G/2-1, −G/2, …, −1
    freqs_np = np.concatenate([np.arange(0, G // 2), np.arange(-(G // 2), 0)])
    freqs = xp.asarray(freqs_np, dtype=real_dtype)
    if D == 1:
        ks = (freqs,)
    else:
        ks = xp.meshgrid(*[freqs] * D, indexing="ij")

    # Weight mask: non-zero only on the M retained modes
    w_np = np.zeros(n_modes, dtype=np.float64)
    w_np[tuple(wv.k[:, d].astype(np.int64) % G for d in range(D))] = wv.w
    w = xp.asarray(w_np, dtype=real_dtype)

    norm = float(M)

    # Reusable NUFFT plans (type-1: points → modes, type-2: modes → points)
    plan_kwargs = dict(
        n_trans=1,
        eps=eps,
        isign=1,
        dtype=complex_dtype,
        modeord=1,
    )
    plan1 = nufft_lib.Plan(1, n_modes, **plan_kwargs)
    plan2 = nufft_lib.Plan(2, n_modes, **plan_kwargs)

    c = xp.ones(N, dtype=complex_dtype)  # unit strengths

    # Target spectrum (computed once)
    if target is not None:
        if isinstance(target, str):          # image path
            target = im2spectrum(target, shape=n_modes, invert=True)
            fk_target = xp.asarray(target, dtype=complex_dtype)
            scale = 1.0 / N
        else:                                # point cloud
            target = xp.asarray(target, dtype=real_dtype)
            if target.ndim != 2 or target.shape[1] != D:
                raise ValueError(f"target must have shape (M, {D}), got {target.shape}")
            M_t = target.shape[0]
            c_target = xp.ones(M_t, dtype=complex_dtype)
            coords_t = tuple(2.0 * xp.pi * target[:, d] for d in range(D))
            plan1.setpts(*coords_t)
            fk_target = plan1.execute(c_target) / M_t
            scale = 1.0 / N
    else:
        fk_target = 0.0
        scale = 1.0

    def loss_and_grad(x):
        """Return (loss, gradient ∇_x L) for the current positions."""
        # NUFFT expects coordinates in [−π, π)
        coords = tuple(2.0 * xp.pi * x[:, d] for d in range(D))

        # Forward pass (type-1): f(k)
        plan1.setpts(*coords)
        fk = plan1.execute(c) * scale

        diff = fk - fk_target
        loss = float(xp.sum(xp.abs(diff) ** 2 * w) / norm)

        # Backward pass (type-2): gradient per coordinate
        grads = []
        for d in range(D):
            grad_source = (2.0 * w * xp.conj(diff) * (1j * 2.0 * xp.pi * ks[d])).astype(
                complex_dtype
            )
            plan2.setpts(*coords)
            g_d = plan2.execute(grad_source)
            grads.append(g_d.real * scale)

        grad = xp.stack(grads, axis=1) / norm
        return loss, grad

    # Initial step ≈ fraction of the mean inter-point distance
    delta = lr * 0.1 * N ** (-1.0 / D)

    if verbose:
        tgt_info = f"target={target.shape[0]} pts" if target is not None else "uniform"
        print(
            f"[nufft | {device}] "
            f"N={N}  D={D}   Chi={Chi:.3f} (eff={chi_eff:.5f})  "
            f"modes(+-k pairs)={M}  G={G}  "
            f"n_iter={n_iter}   ({tgt_info})"
        )
        print("For Early-stopping : Ctrl-C (Keyboard interrupt ⏹️)")

    t0 = time.time()
    adaptive = delta
    prev_loss = float("inf")

    try:
        for it in range(n_iter):
            loss, grad = loss_and_grad(x)

            # RMS normalisation → step length is controlled solely by `adaptive`
            rms = xp.sqrt(xp.mean(grad**2) + 1e-30)
            x_new = x - (grad / rms) * adaptive
            x_new = x_new - xp.floor(x_new)      # wrap back into [0,1)^D

            # Adaptive step ("bold driver")
            if loss < prev_loss:
                x = x_new
                adaptive *= 1.05
                prev_loss = loss
            else:
                adaptive *= 0.8
                x = x_new

            if verbose and (it % 20 == 0 or it == n_iter - 1):
                print(f"  iter {it:4d} | loss {loss:.1e}")

    except KeyboardInterrupt:
        if verbose:
            print(f"\n  [KeyboardInterrupt] stopped at iter {it} | loss {loss:.4f}")
            print(f"  elapsed : {time.time() - t0:.1f}s")

    if verbose:
        print(f"  done in {time.time() - t0:.1f}s")

    return to_numpy(x)


def im2spectrum(path, shape=(512, 512), invert=True):
    """
    Load an image and return its complex Fourier spectrum
    (DFT of a normalised density on `shape`).

    - Convert to greyscale and resize to `shape`.
    - Origin is placed at the bottom-left (mathematical convention).
    - If invert=True, dark pixels become high density.
    - Normalise so the density sums to 1.

    Returns
    -------
    spectrum : complex ndarray of shape `shape`
    """
    from PIL import Image

    img = Image.open(path).convert("L").resize(shape[::-1], Image.LANCZOS)
    rho = np.asarray(img, dtype=np.float64) / 255.0
    rho = (rho.T)[::-1]                 # origin at bottom-left
    if invert:
        rho = 1.0 - rho                 # dark → high density
    rho = rho / rho.sum()
    spectrum = np.fft.fftn(rho)
    return spectrum
