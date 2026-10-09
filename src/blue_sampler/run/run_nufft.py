import time
from math import pi
from typing import NamedTuple

import numpy as np
from ..gpu_setup import set_config   # builds the compute "config" (CPU/GPU, float32/64, NUFFT backend)


# =============================================================================
# Wave-vector selection (D = 1, 2, 3 only)
# =============================================================================
# We optimise the point positions through a finite set of Fourier modes k in Z^D.
# Because fk(-k) = conj(fk(k)), k and -k carry the same information, so we keep
# ONE representative per ±k pair ("half-space"). We pick EXACTLY
#       M = round(Chi * D * (N - 1))
# such representatives, namely the M with the smallest |k|^2 (ties in the last
# shell are broken at random). Modes are weighted by 1 / |k|^2 in the loss.

class WaveVectors(NamedTuple):
    k: np.ndarray        # (M, D) int32, one representative per ±k pair
    r2: np.ndarray       # (M,)   int64, |k|^2
    w: np.ndarray        # (M,)   float64, loss weights in (0, 1], max = 1
    M: int               # number of independent modes
    chi_eff: float       # M / (D * (N-1)); equals Chi up to rounding


def _ball_volume_coeff(D):
    """Volume of the unit ball in R^D: 2 (D=1), pi (D=2), 4*pi/3 (D=3)."""
    return {1: 2.0, 2: pi, 3: 4.0 * pi / 3.0}[D]


def _half_space_mask(k):
    """True where the FIRST nonzero coordinate of k is > 0 (origin -> False).
    Exactly one of k and -k satisfies this, so it picks one representative per ±k pair."""
    pos = np.zeros(len(k), dtype=bool)        # decided "positive"
    decided = np.zeros(len(k), dtype=bool)    # first nonzero coordinate already found
    for d in range(k.shape[1]):
        nz = ~decided & (k[:, d] != 0)
        pos |= nz & (k[:, d] > 0)
        decided |= nz
    return pos


def get_wave_vectors(N, D, Chi, rng=None):
    """Select M = round(Chi * D * (N-1)) wave vectors (one per ±k pair) with the smallest |k|^2.

    Strategy (cheap in D <= 3, no need for clever counting):
      1. Number of ±k pairs in a ball of radius R is about vol(B_D(R)) / 2,
         so R_est = (2 M / vol(B_D(1)))^(1/D) gives a starting radius.
      2. Enumerate the integer cube [-R, R]^D, keep the half-space points with
         |k|^2 <= R^2 (a true ball, so that no point inside it is missed).
      3. If fewer than M points, grow R and repeat.
      4. Sort by |k|^2 with random tie-breaking, keep the first M.
    """
    rng = np.random.default_rng() if rng is None else rng

    n_eff = N - 1                                   # origin (k=0) is excluded, hence N-1 dof
    M = int(round(Chi * D * n_eff))
    if M < 1:
        raise ValueError(f"Chi={Chi} gives no mode for N={N}, D={D}")

    R = int(np.ceil((2.0 * M / _ball_volume_coeff(D)) ** (1.0 / D))) + 1
    while True:
        axis = np.arange(-R, R + 1, dtype=np.int32)
        grids = np.meshgrid(*([axis] * D), indexing="ij")
        k = np.stack([g.ravel() for g in grids], axis=1)        # (cube size, D)
        r2 = np.sum(k.astype(np.int64) ** 2, axis=1)
        keep = (r2 <= R * R) & _half_space_mask(k)              # half-ball, origin excluded
        if keep.sum() >= M:
            break
        R += max(1, R // 10)                                    # not enough modes: enlarge

    k, r2 = k[keep], r2[keep]
    # primary key: |k|^2 ; secondary key: random number (random tie-breaking in the last shell)
    order = np.lexsort((rng.random(len(r2)), r2))[:M]
    k, r2 = k[order], r2[order]

    # Loss weights 1/(|k|^2 + 1e-3), normalised to max = 1:
    # low frequencies are penalised the most. 1e-3 only guards against division by zero.
    w = 1.0 / (r2.astype(np.float64) + 1e-3)
    w /= w.max()

    chi_eff = M / (D * n_eff)
    return WaveVectors(k=k, r2=r2, w=w, M=M, chi_eff=chi_eff)


# =============================================================================
# NUFFT optimisation
# =============================================================================

def _nufft_pipeline(N=10_000, D=2, lr=1.0, warmstart=None, Chi=0.4, target=None,
                    n_iter=120, precision="float32", device="auto", seed=None, 
                    verbose = 1):
    """
    NUFFT-based optimization to generate low-discrepancy points in [0,1)^D
    by minimizing a Fourier energy that penalizes low-frequency clustering.

    Big picture
    -----------
    For a point set x_1..x_N, the "structure factor" (collective coordinates) is
        f(k) = sum_j exp(i 2π k·x_j),   k in Z^D.
    A perfectly uniform (hyperuniform) configuration has f(k) ≈ 0 for small k != 0.
    We therefore minimise
        L(x) = (1/M) * sum_k  w_k * |f(k) - f_target(k)|^2
    over a chosen set of M wave vectors, by gradient descent on the point
    positions. Both f(k) and its gradient are evaluated in O(N log N) with
    non-uniform FFTs (NUFFT) instead of the naive O(N*M) sums.

    Parameters
    ----------
    N : int, default=10_000
        Number of points.
    D : int, default=2
        Dimension (only 1, 2 or 3 supported).
    lr : float, default=1.0
        Base learning-rate scale. Initial step size will be of order  lr * 0.1 * N^(-1/D),
        (i.e. a fraction of the mean inter-point distance).
    warmstart : array-like (N, D) or None, default=None
        Starting configuration. If None, points are drawn uniformly (seeded by `seed`).
    Chi : float, default=0.4
        Fraction of degrees of freedom constrained. The optimised wave vectors are
        EXACTLY M = round(Chi * D * (N-1)) independent modes (one per ±k pair), i.e. the
        M smallest |k|^2 of Z^D (ties in the last shell broken randomly).
        Thus:
            Chi <= 0.3 -> target only low frequencies, make the pipeline easier and faster
            Chi = 0.4 -> balanced, default setting
            Chi >= 0.5 -> target fluctuations at the point level, system starts crystallising
    target: np.ndarray, optional
        An other point cloud representing a target density. Default is uniform density.
        The bigger target the better, otherwise overfitting should be expected
        (A string is also accepted and is interpreted as an image path; see `im2spectrum`.)
    n_iter : int, default=120
        Maximum gradient-descent iterations.
    precision : {"float32", "float64"}, default="float32"
        Float64 will be slower, but allow to reach scattering intensity below 10^-20.
    device : str, default="auto"
        Compute device ("cpu", "cuda", or "auto").
    seed : int or None, default=None
        RNG seed (initial points when `warmstart` is None, and tie-breaking in the
        last shell of wave vectors).
    verbose : int, default=1
        If >0, prints progress.

    Returns
    -------
    x : ndarray, shape (N, D)
        Optimized points in [0,1)^D.
    """
    if D not in (1, 2, 3):
        raise ValueError(f"Only D=1,2,3 supported (got {D})")

    # `cfg` bundles everything that depends on the backend (numpy/cupy, float32/64, ...)
    cfg = set_config(device, precision, verbose)

    # ---------- read the configuration (GPU/CPU, Float/Double) ----------
    xp          = cfg.xp             # array module: numpy on CPU, cupy on GPU (same API)
    real_dtype  = cfg.real_dtype     # float32 or float64
    complex_dtype = cfg.complex_dtype  # complex64 or complex128 (matches real_dtype)
    device = cfg.device              # resolved device ("auto" -> "cpu" or "cuda")
    precision = cfg.precision
    to_numpy = cfg.to_numpy          # converts a xp array back to a plain numpy array
    nufft_lib = cfg.nufft_lib        # NUFFT backend (finufft on CPU / cufinufft on GPU); exposes `.Plan`

    # NUFFT relative tolerance: must be compatible with the floating-point precision
    eps = 1e-4 if precision == "float32" else 1e-8

    rng = np.random.default_rng(seed)

    # ---------- initial positions ----------
    if warmstart is not None:
        x = xp.asarray(warmstart, dtype=real_dtype).reshape(N, D)
    else:
        x = xp.asarray(rng.uniform(size=(N, D)), dtype=real_dtype)

    # ---------- modes: EXACTLY round(Chi*D*(N-1)) ±k pairs, half-space ----------
    # `wv` (see WaveVectors above) describes the selected Fourier modes:
    #   wv.k       : integer array (M, D), the wave vectors k in Z^D, one per ±k pair
    #   wv.w       : array (M,), weight of each mode in the loss (1/|k|^2, max = 1)
    #   wv.M       : number of selected modes
    #   wv.chi_eff : effective Chi actually obtained (M / (D*(N-1))), reported when verbose
    wv = get_wave_vectors(N, D, Chi, rng)
    M, chi_eff = wv.M, wv.chi_eff

    # Smallest even grid holding every selected mode in FFT order:
    # for even G the frequencies are -G/2 .. G/2-1, so +-K_max fit iff G/2 > K_max.
    K_max = int(np.abs(wv.k).max())      # largest |k_d| component among all selected modes
    G = 2 * (K_max + 1)                  # number of Fourier modes per axis in the NUFFT output
    n_modes = (G,) * D                   # shape of the (G, ..., G) Fourier grid, D times

    # Integer frequencies in FFT order (exact, no fftfreq * G rounding)
    # Order: 0, 1, ..., G/2-1, -G/2, ..., -1  (same as numpy.fft, matches modeord=1 below)
    freqs_np = np.concatenate([np.arange(0, G // 2), np.arange(-(G // 2), 0)])
    freqs = xp.asarray(freqs_np, dtype=real_dtype)
    if D == 1:
        ks = (freqs,)
    else:
        # ks[d][i1,...,iD] = value of the d-th component of k at grid cell (i1,...,iD)
        ks = xp.meshgrid(*[freqs] * D, indexing="ij")

    # Weight grid: nonzero ONLY on the M selected half-space modes.
    # (k mod G) is the FFT-order index of frequency k.
    # This acts as a mask: all other grid cells contribute nothing to loss or gradient.
    w_np = np.zeros(n_modes, dtype=np.float64)
    w_np[tuple(wv.k[:, d].astype(np.int64) % G for d in range(D))] = wv.w
    w = xp.asarray(w_np, dtype=real_dtype)

    # Mean over modes. fk(-k) = conj(fk(k)) => the half-space gives the same
    # loss as the full space, and the same gradient up to a factor 2
    # (cancelled by the RMS normalisation below).
    norm = float(M)

    # ---------- plans (reusable) ----------
    # A "plan" pre-computes everything that depends only on sizes/tolerance
    # (spreading kernel, FFT plan, ...). Only the point positions change per
    # iteration, via `setpts`. Reusing plans avoids re-planning at every step.
    plan_kwargs = dict(
        n_trans=1,              # one transform at a time (single vector of strengths)
        eps=eps,                # requested accuracy
        isign=1,                # exponent sign: +i  ->  exp(+i k·x)
        dtype=complex_dtype,
        modeord=1,              # output modes in FFT order (0..G/2-1, -G/2..-1), not centred
    )
    plan1 = nufft_lib.Plan(1, n_modes, **plan_kwargs)  # type-1 (forward): points -> Fourier modes
    plan2 = nufft_lib.Plan(2, n_modes, **plan_kwargs)  # type-2 (backward): Fourier modes -> points

    # Unit strengths: every point contributes exp(i 2π k·x_j) with weight 1
    c = xp.ones(N, dtype=complex_dtype)

    # ---------- target Fourier (computed once) ----------
    # We minimise |f(k) - f_target(k)|^2. For a uniform target, f_target(k)=0 for k != 0
    # (and the k=0 mode is never selected), hence fk_target = 0.
    if target is not None:
        if isinstance(target, str):  # path to an image
            # Target density given as an image: use its DFT directly as f_target
            target = im2spectrum(target, shape=n_modes, invert=True)
            fk_target = xp.asarray(target, dtype=complex_dtype)
            scale = 1.0 / N
        else:
            # Target density given as a (large) point cloud: compute its Fourier
            # coefficients with the same type-1 NUFFT.
            target = xp.asarray(target, dtype=real_dtype)
            if target.ndim != 2 or target.shape[1] != D:
                raise ValueError(f"target must have shape (M, {D}), got {target.shape}")
            M_t = target.shape[0]
            c_target = xp.ones(M_t, dtype=complex_dtype)
            coords_t = tuple(2.0 * xp.pi * target[:, d] for d in range(D))
            plan1.setpts(*coords_t)
            fk_target = plan1.execute(c_target) / M_t        # density normalisation
            scale = 1.0 / N                                  # so that both are densities
    else:
        fk_target = 0.0
        scale = 1.0                                          # classic |fk|^2 (uniform)

    def loss_and_grad(x):
        """Return (loss, gradient wrt x) for the current positions x of shape (N, D)."""
        # NUFFT expects coordinates in [-π, π) (periodic): rescale [0,1) -> [0, 2π)
        coords = tuple(2.0 * xp.pi * x[:, d] for d in range(D))

        # Forward pass: fk[k] = sum_j exp(i k·(2π x_j))  for every k on the grid (type-1 NUFFT)
        plan1.setpts(*coords)
        fk = plan1.execute(c) * scale

        # Residual to the target spectrum, then weighted mean squared modulus
        # (only the M selected modes have w != 0)
        diff = fk - fk_target
        loss = float(xp.sum(xp.abs(diff) ** 2 * w) / norm)

        # Backward pass: d loss / d x_{j,d}. Since d fk[k] / d x_{j,d} = i 2π k_d exp(i 2π k·x_j),
        # the gradient for point j is Re[ sum_k source_d[k] * exp(i 2π k·x_j) ],
        # i.e. a type-2 NUFFT (modes -> points) evaluated at the same points.
        grads = []
        for d in range(D):
            # gradient of |fk - fk_target|^2  ->  2 * conj(diff) * (i 2π k_d)
            grad_source = (2.0 * w * xp.conj(diff) * (1j * 2.0 * xp.pi * ks[d])).astype(
                complex_dtype
            )
            plan2.setpts(*coords)
            g_d = plan2.execute(grad_source)
            grads.append(g_d.real * scale)               # chain rule for the 1/N factor

        grad = xp.stack(grads, axis=1) / norm            # shape (N, D)
        return loss, grad

    # ---------- scheduled gradient descent ----------
    # Initial step length: a fraction (0.1 * lr) of the mean inter-point spacing N^(-1/D).
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
    adaptive = delta                 # current step length, adapted on the fly
    prev_loss = float("inf")         # best loss seen so far

    try:
        for it in range(n_iter):
            schedule = adaptive

            loss, grad = loss_and_grad(x)
            # Normalise the gradient by its RMS: the step LENGTH is then controlled
            # only by `schedule` (a distance in box units), not by the loss scale.
            rms = xp.sqrt(xp.mean(grad**2) + 1e-30)
            x_new = x - (grad / rms) * schedule
            x_new = x_new - xp.floor(x_new)      # wrap back into [0,1)^D (periodic box)

            # Adaptive step ("bold driver"): grow the step by 5% while the loss
            # decreases, shrink by 20% when it does not.
            # NB: in both branches the move is accepted (x = x_new); no rollback.
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
        # Ctrl-C: stop gracefully and still return the current positions
        if verbose:
            print(f"\n  [KeyboardInterrupt] stopped at iter {it} | loss {loss:.4f}")
            print(f"  elapsed : {time.time() - t0:.1f}s")

    if verbose:
        print(f"  done in {time.time() - t0:.1f}s")

    return to_numpy(x)


def im2spectrum(path, shape=(512, 512), invert=True):
    """
    Load any image and return its complex Fourier spectrum
    (DFT of a normalized density on `shape`).

    Returns
    -------
    spectrum : np.ndarray, complex, shape = shape
        FFT of the density (numpy complex128 by default).
    """
    from PIL import Image
    # Grayscale, resized to the Fourier grid size (PIL wants (width, height) = shape reversed)
    img = Image.open(path).convert("L").resize(shape[::-1], Image.LANCZOS)
    rho = np.asarray(img, dtype=np.float64) / 255.0
    # Transpose + flip rows so that axis 0 = x and axis 1 = y with the origin at the bottom-left
    # (image convention has y pointing down)
    rho = (rho.T)[::-1]
    if invert:
        rho = 1.0 - rho          # dark pixels -> high density (more points)
    rho = rho / rho.sum()        # normalise to a probability density (sums to 1)
    # FFT in the same frequency ordering that xp.fft.fftfreq uses
    spectrum = np.fft.fftn(rho)
    return spectrum
