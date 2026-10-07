"""
fallback: replace run_nufft in dimension D >= 4 (where finufft is unavailable)

The number of modes is FIXED EXACTLY: M = round(Chi * D * N) pairs ±k
(a single representative per pair, since fk(-k) = conj(fk(k)) gives the same loss).
The radius (as an integer |k|^2) is the minimal excess radius containing at least
M pairs, computed by counting (theta series), without building anything.
"""

import signal
import time
from math import gamma, pi

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap


# ----------------------------------------------------------------------------
# Integer lattice: exact counting and half-space construction
# (identical to the KeOps fallback; can be imported from there instead)
# ----------------------------------------------------------------------------

def _isqrt(x):
    """Exact floor(sqrt(x)) for integers (scalar or array)."""
    x = np.asarray(x, dtype=np.int64)
    z = np.floor(np.sqrt(x.astype(np.float64))).astype(np.int64)
    z -= (z * z > x)
    z += ((z + 1) * (z + 1) <= x)
    return z


def _min_r2_for_pairs(D, M_target, n_guess):
    """Smallest integer n* such that #{k in Z^D, 0 < |k|^2 <= n*} / 2 >= M_target.

    r_D(n) = #{k : |k|^2 = n} is obtained by convolving the 1D theta series
    D times. No point is ever constructed.
    """
    n_max = max(int(n_guess), 4)
    while True:
        r = np.zeros(n_max + 1, dtype=np.int64)
        r[0] = 1
        for _ in range(D):
            new = np.zeros_like(r)
            s = 0
            while s * s <= n_max:
                new[s * s:] += (1 if s == 0 else 2) * r[: n_max + 1 - s * s]
                s += 1
            r = new
        pairs = (np.cumsum(r) - 1) // 2          # ±k pairs, origin excluded
        idx = int(np.searchsorted(pairs, M_target, side="left"))
        if idx <= n_max:
            return idx
        n_max *= 2                               # search window too small: double it


def _cylinder_z(counts, lo):
    """For each cylinder i, the integers lo[i], lo[i]+1, ..., lo[i]+counts[i]-1,
    concatenated (int32, no meshgrid)."""
    total = int(counts.sum())
    starts = (np.cumsum(counts) - counts).astype(np.int32)
    z = np.arange(total, dtype=np.int32)
    z -= np.repeat(starts, counts)
    z += np.repeat(np.asarray(lo, dtype=np.int32), counts)
    return z


def _full_ball_int(D, r2max):
    """All k in Z^D with |k|^2 <= r2max (exact integer bound), int32."""
    if D == 1:
        K = int(_isqrt(r2max))
        return np.arange(-K, K + 1, dtype=np.int32)[:, None]
    prev = _full_ball_int(D - 1, r2max)
    r2 = np.sum(prev.astype(np.int64) ** 2, axis=1)
    m = _isqrt(r2max - r2)
    counts = 2 * m + 1
    base = np.repeat(prev, counts, axis=0)
    z = _cylinder_z(counts, -m)
    return np.concatenate([base, z[:, None]], axis=1)


def _half_ball_int(D, r2max):
    """Half-space {last nonzero coordinate > 0}, origin excluded.
    One representative per ±k pair, without ever building the full D-dim ball."""
    if D == 1:
        K = int(_isqrt(r2max))
        return np.arange(1, K + 1, dtype=np.int32)[:, None]

    # last coordinate z > 0, the first (D-1) coordinates arbitrary
    full_prev = _full_ball_int(D - 1, r2max)
    r2 = np.sum(full_prev.astype(np.int64) ** 2, axis=1)
    m = _isqrt(r2max - r2)                          # z in 1..m
    base = np.repeat(full_prev, m, axis=0)
    z = _cylinder_z(m, np.ones_like(m))
    part1 = np.concatenate([base, z[:, None]], axis=1)
    del base, z, full_prev

    # last coordinate z = 0, the first (D-1) coordinates in the half-space
    half_prev = _half_ball_int(D - 1, r2max)
    part2 = np.concatenate(
        [half_prev, np.zeros((len(half_prev), 1), dtype=np.int32)], axis=1
    )
    return np.concatenate([part1, part2], axis=0)


def exact_chi_modes(N, D, Chi, real_dtype=np.float32, rng=None, torquato=False):
    """Return (k_coords (M, D), r2 (M,)) with M = round(Chi*D*N) EXACTLY.

    One representative per ±k pair (origin excluded). The M smallest |k|^2 are
    kept; ties in the last (partially filled) shell are broken randomly.

    torquato=True: M = round(Chi * D * (N-1)) instead of round(Chi * D * N).
    """
    rng = np.random.default_rng() if rng is None else rng
    n_eff = (N - 1) if torquato else N
    M_target = int(round(Chi * D * n_eff))

    # Starting bound for the search (volume-based), ONLY used to size the
    # counting array; the final radius is the exact minimal excess radius.
    ball_ratio = (pi ** (D / 2) / gamma(D / 2 + 1)) / (2 ** D)
    kfrac = (2 * Chi * D / ball_ratio) ** (1.0 / D)
    n_guess = 1.5 * (N ** (1.0 / D) * kfrac / 2) ** 2 + 10

    r2star = _min_r2_for_pairs(D, M_target, n_guess)

    K = _half_ball_int(D, r2star)
    r2 = np.sum(K.astype(np.int64) ** 2, axis=1)
    assert len(K) >= M_target

    # sort by |k|^2, random tie-breaking within the last shell
    order = np.lexsort((rng.random(len(r2)), r2))[:M_target]
    K, r2 = K[order], r2[order]

    return K.astype(real_dtype), r2.astype(np.float64)


# ----------------------------------------------------------------------------
# JAX pipeline
# ----------------------------------------------------------------------------

def _nufft_pipeline_jax(
    N=10_000, D=2, lr=1.0, warmstart=None, Chi=0.4, target=None,
    n_iter=120, precision="float32", device="auto", seed=None, verbose=1,
    torquato=False,
):
    """JAX brute-force fallback for arbitrary dimension D.
    SLOW, with quadratic complexity in N.
    See blue_sampler.run.run_nufft for the fast D <= 3 version.

    The modes are exactly M = round(Chi*D*N) ±k pairs (half-space).
    """
    np_real = np.float32 if precision == "float32" else np.float64
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
    k_np, r2_np = exact_chi_modes(N, D, Chi, np_real, rng, torquato=torquato)

    M = k_np.shape[0]                           # number of independent modes
    n_eff = (N - 1) if torquato else N
    assert M == int(round(Chi * D * n_eff)), (M, Chi, D, N)
    chi_eff = M / (D * n_eff)
    assert abs(chi_eff - Chi) <= 0.5 / (D * n_eff) + 1e-12, (chi_eff, Chi)

    k_coords = jnp.asarray(np.ascontiguousarray(k_np), dtype=real_dtype)
    r2 = jnp.asarray(r2_np, dtype=real_dtype)

    rpow = (r2 + 1e-3) ** -1.0
    w = rpow / rpow.max()

    # Mean over modes. fk(-k) = conj(fk(k)) => the half-space gives the same
    # loss as the full space, and the same gradient up to a factor 2
    # (cancelled by the RMS normalisation in the update step).
    norm = float(max(M, 1))
    r2max = float(r2_np.max())

    # Largest |k_d| actually present (exact: integer-valued coordinates)
    K_max = int(np.abs(k_np).max())
    k_pos = jnp.arange(1, K_max + 1, dtype=real_dtype)

    k_abs = jnp.abs(k_coords).astype(jnp.int32)
    k_sign = jnp.sign(k_coords).astype(jnp.int8)

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