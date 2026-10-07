import time
import numpy as np
from ..gpu_setup import set_config
from .run_nufft_helpers import get_wave_vectors


def _nufft_pipeline(N=10_000, D=2, lr=1.0, warmstart=None, Chi=0.4, target=None,
                    n_iter=120, precision="float32", device="auto", seed=None, 
                    verbose = 1):
    """
    NUFFT-based optimization to generate low-discrepancy points in [0,1)^D
    by minimizing a Fourier energy that penalizes low-frequency clustering.

    Parameters
    ----------
    N : int, default=10_000
        Number of points.
    D : int, default=2
        Dimension (only 1, 2 or 3 supported).
    lr : float, default=1.0
        Base learning-rate scale. Initial step size will be of order  lr * 0.1 * N^(-1/D),
    warmstart : array-like (N, D) or None, default=None
        Starting configuration. If None, points are drawn uniformly (seeded by `seed`).
    Chi : float, default=0.4
        Fraction of degrees of freedom constrained. The optimised wave vectors are
        EXACTLY M = round(Chi * D * N) independent modes (one per ±k pair), i.e. the
        M smallest |k|^2 of Z^D (ties in the last shell broken randomly).
        Thus:
            Chi <= 0.3 -> target only low frequencies, make the pipeline easier and faster
            Chi = 0.4 -> balanced, default setting
            Chi >= 0.5 -> target fluctuations at the point level, system starts crystallising
    target: np.ndarray, optional
        An other point cloud representing a target density. Default is uniform density.
        The bigger target the better, otherwise overfitting should be expected
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

    cfg = set_config(device, precision, verbose)

    # ---------- read the configuration (GPU/CPU, Float/Double) ----------
    xp          = cfg.xp
    real_dtype  = cfg.real_dtype
    complex_dtype = cfg.complex_dtype
    device = cfg.device
    precision = cfg.precision
    to_numpy = cfg.to_numpy
    nufft_lib = cfg.nufft_lib

    eps = 1e-4 if precision == "float32" else 1e-8

    rng = np.random.default_rng(seed)

    if warmstart is not None:
        x = xp.asarray(warmstart, dtype=real_dtype).reshape(N, D)
    else:
        x = xp.asarray(rng.uniform(size=(N, D)), dtype=real_dtype)

    # ---------- modes: EXACTLY round(Chi*D*N) ±k pairs, half-space ----------
    wv = get_wave_vectors(N, D, Chi, rng)
    M, chi_eff = wv.M, wv.chi_eff

    # Smallest even grid holding every selected mode in FFT order:
    # for even G the frequencies are -G/2 .. G/2-1, so +-K_max fit iff G/2 > K_max.
    K_max = int(np.abs(wv.k).max())
    G = 2 * (K_max + 1)
    n_modes = (G,) * D

    # Integer frequencies in FFT order (exact, no fftfreq * G rounding)
    freqs_np = np.concatenate([np.arange(0, G // 2), np.arange(-(G // 2), 0)])
    freqs = xp.asarray(freqs_np, dtype=real_dtype)
    if D == 1:
        ks = (freqs,)
    else:
        ks = xp.meshgrid(*[freqs] * D, indexing="ij")

    # Weight grid: nonzero ONLY on the M selected half-space modes.
    # (k mod G) is the FFT-order index of frequency k.
    w_np = np.zeros(n_modes, dtype=np.float64)
    w_np[tuple(wv.k[:, d].astype(np.int64) % G for d in range(D))] = wv.w
    w = xp.asarray(w_np, dtype=real_dtype)

    # Mean over modes. fk(-k) = conj(fk(k)) => the half-space gives the same
    # loss as the full space, and the same gradient up to a factor 2
    # (cancelled by the RMS normalisation below).
    norm = float(M)

    # ---------- plans (reusable) ----------
    plan_kwargs = dict(
        n_trans=1,
        eps=eps,
        isign=1,
        dtype=complex_dtype,
        modeord=1,
    )
    plan1 = nufft_lib.Plan(1, n_modes, **plan_kwargs)  # type-1 (forward)
    plan2 = nufft_lib.Plan(2, n_modes, **plan_kwargs)  # type-2 (backward)

    c = xp.ones(N, dtype=complex_dtype)

    # ---------- target Fourier (computed once) ----------
    if target is not None:
        if isinstance(target, str):  # path to an image
            target = im2spectrum(target, shape=n_modes, invert=True)
            fk_target = xp.asarray(target, dtype=complex_dtype)
            scale = 1.0 / N
        else:
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
        coords = tuple(2.0 * xp.pi * x[:, d] for d in range(D))

        plan1.setpts(*coords)
        fk = plan1.execute(c) * scale

        diff = fk - fk_target
        loss = float(xp.sum(xp.abs(diff) ** 2 * w) / norm)

        grads = []
        for d in range(D):
            # gradient of |fk - fk_target|^2  ->  2 * conj(diff) * (i 2π k_d)
            grad_source = (2.0 * w * xp.conj(diff) * (1j * 2.0 * xp.pi * ks[d])).astype(
                complex_dtype
            )
            plan2.setpts(*coords)
            g_d = plan2.execute(grad_source)
            grads.append(g_d.real * scale)               # chain rule for the 1/N factor

        grad = xp.stack(grads, axis=1) / norm
        return loss, grad

    # ---------- scheduled gradient descent ----------
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
            schedule = adaptive

            loss, grad = loss_and_grad(x)
            rms = xp.sqrt(xp.mean(grad**2) + 1e-30)
            x_new = x - (grad / rms) * schedule
            x_new = x_new - xp.floor(x_new)

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
    Load any image and return its complex Fourier spectrum
    (DFT of a normalized density on `shape`).

    Returns
    -------
    spectrum : np.ndarray, complex, shape = shape
        FFT of the density (numpy complex128 by default).
    """
    from PIL import Image
    img = Image.open(path).convert("L").resize(shape[::-1], Image.LANCZOS)
    rho = np.asarray(img, dtype=np.float64) / 255.0
    rho = (rho.T)[::-1]
    if invert:
        rho = 1.0 - rho
    rho = rho / rho.sum()
    # FFT in the same frequency ordering that xp.fft.fftfreq uses
    spectrum = np.fft.fftn(rho)
    return spectrum