"""
fallback: replace run_nufft in dimension D >= 4 (were finufft is unavailable)
"""

import signal
import time
from math import gamma, pi

import numpy as np

try:
    from pykeops.numpy import LazyTensor

    _x = np.random.rand(10, 3).astype(np.float32)
    _y = np.random.rand(10, 3).astype(np.float32)

    _x_i = LazyTensor(_x[:, None, :])
    _y_j = LazyTensor(_y[None, :, :])

    _dist2 = ((_x_i - _y_j) ** 2).sum(-1)
    KEOPS_AVAILABLE = True
except Exception:
    KEOPS_AVAILABLE = False


def _nufft_pipeline_keops(
    N=10_000, D=2, lr=1.0, warmstart=None, Chi=0.4, target=None,
    n_iter=120, precision="float32", device="auto", seed=None, verbose=1,
):
    """KeOps bruteforce fallback for arbitrary dimension D.
    SLOW, with quadratic complexity in N.
    See blue_sampler.run.run_nufft for the fast D <= 3 version.
    """
    assert KEOPS_AVAILABLE, "keops unavailable"
    real_dtype = np.float32 if precision == "float32" else np.float64
    complex_dtype = np.complex64 if precision == "float32" else np.complex128

    # IMPORTANT: plain Python float (a 0-d numpy array breaks LazyTensor.__rmul__)
    two_pi = float(2.0 * pi)

    ball_ratio = (pi ** (D / 2) / gamma(D / 2 + 1)) / (2 ** D)
    kfrac = (2 * Chi * D / ball_ratio) ** (1.0 / D)

    if warmstart is not None:
        x = np.asarray(warmstart, dtype=real_dtype).reshape(N, D)
    else:
        rng = np.random.default_rng(seed)
        x = rng.uniform(size=(N, D)).astype(real_dtype)
    x = np.ascontiguousarray(x)

    G = int(np.ceil(N ** (1.0 / D)) * kfrac)
    if G % 2:
        G += 1

    freqs = (np.fft.fftfreq(G) * G).astype(real_dtype)
    ks_list = np.meshgrid(*([freqs] * D), indexing="ij")
    r2 = sum(k * k for k in ks_list)

    max_freq_sq = freqs.max() ** 2
    mask = (r2 > 0) & (r2 <= max_freq_sq)
    k_coords = np.ascontiguousarray(
        np.stack([k[mask] for k in ks_list], axis=-1).astype(real_dtype)
    )

    rpow = (r2[mask] + 1e-3) ** -1.0
    w = (rpow / rpow.max()).astype(real_dtype)

    M = k_coords.shape[0]
    norm = float(max(M, 1))
    K_max = G // 2

    # Convention: fk(x) = (1/N) * sum_j exp(-2i*pi*k.x_j)
    if target is not None:
        if isinstance(target, str):
            raise NotImplementedError(
                "image target is not supported in the pure-KeOps path"
            )

        target = np.ascontiguousarray(np.asarray(target, dtype=real_dtype))
        if target.ndim != 2 or target.shape[1] != D:
            raise ValueError(
                f"target must have shape (M, {D}), got {target.shape}"
            )

        target_x = LazyTensor(target[:, None, :])
        target_k = LazyTensor(k_coords[None, :, :])

        target_phase = two_pi * (target_x * target_k).sum(-1)

        target_real = np.asarray(target_phase.cos().sum_reduction(axis=0)).ravel()
        target_imag = -np.asarray(target_phase.sin().sum_reduction(axis=0)).ravel()

        fk_target = (
            (target_real + 1j * target_imag) / target.shape[0]
        ).astype(complex_dtype)
    else:
        fk_target = np.zeros(M, dtype=complex_dtype)

    fk_target_real = np.ascontiguousarray(fk_target.real.astype(real_dtype))
    fk_target_imag = np.ascontiguousarray(fk_target.imag.astype(real_dtype))

    k_j = LazyTensor(k_coords[None, :, :])

    def _fourier_modes(x):
        x_i = LazyTensor(x[:, None, :])
        phase = two_pi * (x_i * k_j).sum(-1)

        fk_real = np.asarray(phase.cos().sum_reduction(axis=0)).ravel() / N
        fk_imag = -np.asarray(phase.sin().sum_reduction(axis=0)).ravel() / N

        return fk_real.astype(real_dtype), fk_imag.astype(real_dtype)

    def _loss_and_grad(x):
        fk_real, fk_imag = _fourier_modes(x)

        diff_real = fk_real - fk_target_real
        diff_imag = fk_imag - fk_target_imag

        loss = float(np.sum(w * (diff_real ** 2 + diff_imag ** 2)) / norm)

        # d fk / d x_j carries a 1/N factor (fk is normalised by N)
        pref = 2.0 / norm * (-two_pi) / N
        coeff_real = np.ascontiguousarray((pref * w * diff_real).astype(real_dtype))
        coeff_imag = np.ascontiguousarray((pref * w * diff_imag).astype(real_dtype))

        x_i = LazyTensor(x[:, None, :])
        coeff_real_j = LazyTensor(coeff_real[None, :, None])
        coeff_imag_j = LazyTensor(coeff_imag[None, :, None])

        phase = two_pi * (x_i * k_j).sum(-1)

        scalar = coeff_real_j * phase.sin() + coeff_imag_j * phase.cos()

        grad = (scalar * k_j).sum_reduction(axis=1)

        return loss, np.asarray(grad).astype(real_dtype)

    delta = lr * 0.1 * N ** (-1.0 / D)
    adaptive = delta
    prev_loss = np.inf
    last_loss = prev_loss

    stop_requested = False

    def _sigint_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        if verbose:
            print("\n[KeyboardInterrupt]")

    try:
        old_handler = signal.signal(signal.SIGINT, _sigint_handler)
    except ValueError:  # not in the main thread
        old_handler = None

    if verbose:
        tgt_info = f"target={target.shape[0]} pts" if target is not None else "uniform"
        print(
            f"nufft unavailable ({D}D curently unsuported by finufft)...\n"
            f"...keops-radical fallback (slow, O(N^2) instead of NlogN)\n"
            f"should be ok for N <= 2k/4k on CPU, or 10k/20K on GPU\n"
            f"[keops-rad] "
            f"N={N} D={D} Chi={Chi:.2f} modes={M} K_max={K_max} "
            f"n_iter={n_iter} ({tgt_info})"
        )

    t0 = time.time()

    try:
        for it in range(n_iter):
            if stop_requested:
                break

            loss, grad = _loss_and_grad(x)

            rms = float(np.sqrt(np.mean(grad * grad) + 1e-30))

            x = np.ascontiguousarray(
                np.mod(x - (grad / rms) * adaptive, 1.0).astype(real_dtype)
            )

            improved = loss < prev_loss

            if improved:
                adaptive *= 1.05
                prev_loss = loss
            else:
                adaptive *= 0.8

            last_loss = loss

            if verbose:
                print(f"iter {it:4d} | loss {float(last_loss):.1e}")

    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)

    if verbose:
        status = " (keyboard interrupted)" if stop_requested else ""
        print(f"done in {time.time() - t0:.1f}s{status}")

    return np.asarray(x)