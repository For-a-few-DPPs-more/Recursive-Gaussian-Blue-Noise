"""
fallback: replace run_nufft in dimension D >= 4 (where finufft is unavailable)

Wave vectors: see run_nufft_helpers.py (exactly M = round(Chi*D*N) ±k pairs).
"""

import signal
import time
from math import pi

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap

from .run_nufft_helpers import get_wave_vectors


def _nufft_pipeline_jax(
    N=10_000, D=2, lr=1.0, warmstart=None, Chi=0.4, target=None,
    n_iter=120, precision="float32", device="auto", seed=None, verbose=1,
):
    """JAX brute-force fallback for arbitrary dimension D.
    SLOW, with quadratic complexity in N.
    See blue_sampler.run.run_nufft for the fast D <= 3 version.

    The modes are exactly M = round(Chi*D*N) ±k pairs (half-space).
    """
    real_dtype = jnp.float32 if precision == "float32" else jnp.float64
    complex_dtype = jnp.complex64 if precision == "float32" else jnp.complex128
    two_pi = jnp.asarray(2.0 * pi, dtype=real_dtype)
    neg_two_pi_i = jnp.asarray(-2.0j * pi, dtype=complex_dtype)

    rng = np.random.default_rng(seed)

    if warmstart is not None:
        x0 = jnp.asarray(warmstart, dtype=real_dtype).reshape(N, D)
    else:
        x0 = jnp.asarray(rng.uniform(size=(N, D)), dtype=real_dtype)

    # --- modes: EXACTLY round(Chi*D*N) ±k pairs, half-space ---
    wv = get_wave_vectors(N, D, Chi, rng)
    M, chi_eff = wv.M, wv.chi_eff
    r2max = float(wv.r2.max())

    k_coords = jnp.asarray(wv.k, dtype=real_dtype)
    w = jnp.asarray(wv.w, dtype=real_dtype)

    # Mean over modes. fk(-k) = conj(fk(k)) => the half-space gives the same
    # loss as the full space, and the same gradient up to a factor 2
    # (cancelled by the RMS normalisation in the update step).
    norm = float(max(M, 1))

    # Largest |k_d| actually present (exact: integer-valued coordinates)
    K_max = int(np.abs(wv.k).max())
    k_pos = jnp.arange(1, K_max + 1, dtype=real_dtype)

    k_abs = jnp.asarray(np.abs(wv.k), dtype=jnp.int32)
    k_sign = jnp.asarray(np.sign(wv.k), dtype=jnp.int8)

    # Convention: fk(x) = (1/N) * sum_j exp(-2i*pi*k.x_j)  (mean over points),
    # and the same normalisation for the target.
    if target is not None:
        if isinstance(target, str):
            raise NotImplementedError("image target is not supported in the pure-JAX path")
        target = jnp.asarray(target, dtype=real_dtype)
        if target.ndim != 2 or target.shape[1] != D:
            raise ValueError(f"target must have shape (M, {D}), got {target.shape}")

        def _one_mode_target(k):
            return jnp.exp(-1j * two_pi * (target @ k)).mean()

        fk_target = vmap(_one_mode_target)(k_coords)
    else:
        fk_target = jnp.zeros(M, dtype=complex_dtype)

    def _build_buf(x):
        # buf[n, d, k-1] = exp(-2i*pi * k * x[n, d]),  k = 1..K_max
        phase = x[:, :, None] * k_pos[None, None, :]
        return jnp.exp(neg_two_pi_i * phase).astype(complex_dtype)

    def _modes_from_buf(buf):
        # e[m, n] = prod_d exp(-2i*pi * k[m, d] * x[n, d]) from the 1D buffer
        idx = jnp.clip(k_abs - 1, 0, K_max - 1)
        factors = jnp.take_along_axis(
            buf[None, :, :, :], idx[:, None, :, None], axis=3
        )[..., 0]
        factors = jnp.where(k_sign[:, None, :] < 0, jnp.conj(factors), factors)
        factors = jnp.where(
            k_sign[:, None, :] == 0,
            jnp.ones((), dtype=complex_dtype),
            factors,
        )
        return jnp.prod(factors, axis=2)

    def loss_and_grad(x):
        buf = _build_buf(x)
        e_mn = _modes_from_buf(buf)
        fk = e_mn.mean(axis=1)
        diff = fk - fk_target
        loss = jnp.sum(w * jnp.abs(diff) ** 2) / norm
        # The 1/N from d fk / d x_j is omitted: it is cancelled by the RMS
        # normalisation of the gradient in body_fun.
        coeff = (2.0 / norm) * w * jnp.conj(diff) * neg_two_pi_i
        weighted_e = coeff[:, None] * e_mn
        grad = jnp.real(k_coords.T.astype(complex_dtype) @ weighted_e).T
        return loss, grad

    loss_and_grad = jit(loss_and_grad)

    def body_fun(i, state):
        x, adaptive, prev_loss, _ = state
        loss, grad = loss_and_grad(x)
        rms = jnp.sqrt(jnp.mean(grad * grad) + 1e-30)
        x_new = jnp.mod(x - (grad / rms) * adaptive, 1.0)
        improved = loss < prev_loss
        adaptive_new = jnp.where(improved, adaptive * 1.05, adaptive * 0.8)
        prev_loss_new = jnp.where(improved, loss, prev_loss)
        return x_new, adaptive_new, prev_loss_new, loss

    delta = lr * 0.1 * N ** (-1.0 / D)
    x = x0
    adaptive = jnp.asarray(delta, dtype=real_dtype)
    prev_loss = jnp.asarray(jnp.inf, dtype=real_dtype)
    last_loss = prev_loss

    chunk = 20
    n_chunks = (n_iter + chunk - 1) // chunk
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
            f"nufft unavailable ({D}D currently unsupported by finufft)...\n"
            f"...jax-radical fallback (slow, O(N^2) instead of NlogN)\n"
            f"should be ok for N <= 2k/4k on CPU, or 10k/20K on GPU\n"
            f"[jax-rad | {jax.default_backend()}] "
            f"N={N} D={D} Chi={Chi:.3f} (eff={chi_eff:.5f}) modes(+-k pairs)={M} "
            f"|k|^2_max={r2max:.0f} K_max={K_max} n_iter={n_iter} ({tgt_info})"
        )

    t0 = time.time()

    try:
        for c in range(n_chunks):
            if stop_requested:
                break

            start = c * chunk
            n_steps = min(chunk, n_iter - start)
            x, adaptive, prev_loss, last_loss = lax.fori_loop(
                0, n_steps, body_fun, (x, adaptive, prev_loss, last_loss)
            )

            if verbose:
                it = start + n_steps - 1
                print(f"iter {it:4d} | loss {float(last_loss):.1e}")
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)

    if verbose:
        status = " (keyboard interrupted)" if stop_requested else ""
        print(f"done in {time.time() - t0:.1f}s{status}")

    return np.asarray(x)