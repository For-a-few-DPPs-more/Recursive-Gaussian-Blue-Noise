"""
fallback: replace run_nufft in dimension D >= 4 (were finufft is unavailable)
"""


import signal
import time
from math import gamma, pi

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap


def _nufft_pipeline_jax(
    N=10_000, D=2, lr=1.0, warmstart=None, Chi=0.4, target=None,
    n_iter=120, precision="float32", device="auto", seed=None, verbose=1,
):
    """JAX bruteforce fallback for arbitrary dimension D.
    SLOW, with quadratic complexity in N.
    See blue_sampler.run.run_nufft for the fast D <= 3 version.
    """
    real_dtype = jnp.float32 if precision == "float32" else jnp.float64
    complex_dtype = jnp.complex64 if precision == "float32" else jnp.complex128
    two_pi = jnp.asarray(2.0 * pi, dtype=real_dtype)
    neg_two_pi_i = jnp.asarray(-2.0j * pi, dtype=complex_dtype)

    ball_ratio = (pi ** (D / 2) / gamma(D / 2 + 1)) / (2 ** D)
    kfrac = (2 * Chi * D / ball_ratio) ** (1.0 / D)

    if warmstart is not None:
        x0 = jnp.asarray(warmstart, dtype=real_dtype).reshape(N, D)
    else:
        rng = np.random.default_rng(seed)
        x0 = jnp.asarray(rng.uniform(size=(N, D)), dtype=real_dtype)

    G = int(np.ceil(N ** (1.0 / D)) * kfrac)
    if G % 2:
        G += 1

    freqs = jnp.fft.fftfreq(G).astype(real_dtype) * G
    ks_list = jnp.meshgrid(*([freqs] * D), indexing="ij")
    r2 = sum(k * k for k in ks_list)

    max_freq_sq = freqs.max() ** 2
    mask = (r2 > 0) & (r2 <= max_freq_sq)
    k_coords = jnp.stack([k[mask] for k in ks_list], axis=-1)

    rpow = (r2[mask] + 1e-3) ** -1.0
    w = rpow / rpow.max()

    M = k_coords.shape[0]
    norm = float(max(M, 1))
    K_max = G // 2
    k_pos = jnp.arange(1, K_max + 1, dtype=real_dtype)

    k_abs = jnp.abs(k_coords).astype(jnp.int32)
    k_sign = jnp.sign(k_coords).astype(jnp.int8)
    dim_idx = jnp.arange(D)[:, None]

    if target is not None:
        if isinstance(target, str):
            raise NotImplementedError("image target is not supported in the pure-JAX path")
        target = jnp.asarray(target, dtype=real_dtype)
        if target.ndim != 2 or target.shape[1] != D:
            raise ValueError(f"target must have shape (M, {D}), got {target.shape}")

        def _one_mode_target(k):
            return jnp.exp(-1j * two_pi * (target @ k)).mean()

        fk_target = vmap(_one_mode_target)(k_coords)
        scale = 1.0 / N
    else:
        fk_target = jnp.zeros(M, dtype=complex_dtype)
        scale = 1.0

    def _build_buf(x):
        phase = x[:, :, None] * k_pos[None, None, :]
        return jnp.exp(neg_two_pi_i * phase).astype(complex_dtype)

    def _modes_from_buf(buf):
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
        fk = e_mn.mean(axis=1) * scale
        diff = fk - fk_target
        loss = jnp.sum(w * jnp.abs(diff) ** 2) / norm
        coeff = (2.0 / norm) * w * jnp.conj(diff) * neg_two_pi_i * scale
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

    old_handler = signal.signal(signal.SIGINT, _sigint_handler)

    if verbose:
        tgt_info = f"target={target.shape[0]} pts" if target is not None else "uniform"
        print(
            f"nufft unavailable ({D}D curently unsuported by finufft)...\n" 
            f"...jax-radical fallback (slow, O(N^2) instead of NlogN)\n"
            f"should be ok for N <= 2k/4k on CPU, or 10k/20K on GPU"
            f"[jax-rad | {jax.default_backend()}] "
            f"N={N} D={D} Chi={Chi:.2f} modes={M} K_max={K_max} "
            f"n_iter={n_iter} ({tgt_info})"
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
        signal.signal(signal.SIGINT, old_handler)

    if verbose:
        status = " (keyboard interrupted)" if stop_requested else ""
        print(f"done in {time.time() - t0:.1f}s{status}")

    return np.asarray(x)
