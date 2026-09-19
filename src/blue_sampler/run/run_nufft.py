import time
import numpy as np
from ..gpu_setup import set_config


def _nufft_pipeline(N=10_000, D=2, lr=1.0, warmstart=None, Chi=0.4, target=None,
                    n_iter=120, precision="float32", device="auto", seed=None, verbose=1):
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
        Frequency cutoff: control the ratio of freedom degrees of the system we are optimising.
        The number of degrees of freedom is 2* Chi.
        Thus:
            Chi <= 0.3 -> target only low frequencies, make the pipeline easyer and faster
            CHi = 0.4 -> balanced, default setting
            Chi >= 0.5 -> target fluctuations at the point level, system start cristalising
    target: np.ndarray, optional
        An other point cloud representing a target density. Default is uniform density.
        The bigger target the better, otherwise overfitting should be expected
    n_iter : int, default=120
        Maximum gradient-descent iterations.
    precision : {"float32", "float64"}, default="float32"
        Float64 will be slower, but allow to reach scattering intensity bellow 10^-20.
    device : str, default="auto"
        Compute device ("cpu", "cuda", or "auto").
    seed : int or None, default=None
        RNG seed used only when `warmstart` is None.
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

    eps = 1e-5 if precision == "float32" else 1e-10
    resolution = 1.5
    # ---------- geometry ----------
    if D == 2:
        kfrac = float(np.sqrt(2 / np.pi) * 2 * Chi)
    elif D == 3:
        kfrac = float((2 / ((4 / 3) * np.pi)) ** (1 / 3) * 2 * Chi)
    else:
        kfrac = 2.0 * Chi

    if warmstart is not None:
        x = xp.asarray(warmstart, dtype=real_dtype).reshape(N, D)
    else:
        rng = np.random.default_rng(seed)
        x = xp.asarray(rng.uniform(size=(N, D)), dtype=real_dtype)

    G = int(np.ceil(N ** (1.0 / D)) * resolution)
    if G % 2:
        G += 1
    n_modes = (G,) * D

    # frequencies (FFT order)
    freqs = xp.fft.fftfreq(G).astype(real_dtype) * G
    if D == 1:
        ks = (freqs,)
    else:
        ks = xp.meshgrid(*[freqs] * D, indexing="ij")

    r2 = sum(k**2 for k in ks)
    rpow = (r2 + 1e-3) ** (-1.0)
    mask = (r2 > 0) & (r2 <= (G * min(kfrac / resolution, 1.0)) ** 2)
    w = xp.where(mask, rpow, 0.0).astype(real_dtype)
    norm = float(xp.maximum(mask.sum(), 1.0))
    w = w / w.max()

    # ---------- plans (réutilisables) ----------
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
        if isinstance(target, str): #path to an image
            target = im2spectrum(target, shape=n_modes, invert=True)
            fk_target = xp.asarray(target, dtype=complex_dtype)
            scale = 1.0 / N
        else:
            target = xp.asarray(target, dtype=real_dtype)
            if target.ndim != 2 or target.shape[1] != D:
                raise ValueError(f"target must have shape (M, {D}), got {target.shape}")
            M = target.shape[0]
            c_target = xp.ones(M, dtype=complex_dtype)
            coords_t = tuple(2.0 * xp.pi * target[:, d] for d in range(D))
            plan1.setpts(*coords_t)
            fk_target = plan1.execute(c_target) / M          # density normalisation
            scale = 1.0 / N                                  # so that both are densities
    else:
        fk_target = 0.0
        scale = 1.0                                      # classic |fk|^2 (uniform)

    def loss_and_grad(x: xp.ndarray):
        coords = tuple(2.0 * xp.pi * x[:, d] for d in range(D))

        plan1.setpts(*coords)
        fk = plan1.execute(c) * scale

        diff = fk - fk_target
        loss = float(xp.sum(xp.abs(diff) ** 2 * w) / norm)

        grads = []
        for d in range(D):
            # gradient of |fk - fk_target|^2  →  2 * conj(diff) * (i 2π k_d)
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
            f"[nufft | {device} | {precision}] "
            f"N={N}  D={D}  G={G}  kfrac={kfrac:.4f}  "
            f"n_iter={n_iter}  delta={delta:.4f}  ({tgt_info})"
        )
        print("For Early-stopping : Ctrl-C (Keyboard interrupt ⏹️)")

    t0 = time.time()
    adaptive = delta
    prev_loss = float("inf")

    try:
        for it in range(n_iter):
            envelope = 1.0 #float(np.exp(-5.0 * it / n_iter))
            schedule = adaptive * envelope

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