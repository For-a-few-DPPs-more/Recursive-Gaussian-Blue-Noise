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


# ----------------------------------------------------------------------------
# Integer lattice: exact counting and half-space construction
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
# KeOps pipeline
# ----------------------------------------------------------------------------

def _nufft_pipeline_keops(
    N=10_000, D=2, lr=1.0, warmstart=None, Chi=0.4, target=None,
    n_iter=120, precision="float32", device="auto", seed=None, verbose=1,
    torquato=False,
):
    """KeOps brute-force fallback for arbitrary dimension D.
    SLOW, with quadratic complexity in N.
    See blue_sampler.run.run_nufft for the fast D <= 3 version.

    The modes are exactly M = round(Chi*D*N) ±k pairs (half-space).
    """
    assert KEOPS_AVAILABLE, "keops unavailable"
    real_dtype = np.float32 if precision == "float32" else np.float64
    complex_dtype = np.complex64 if precision == "float32" else np.complex128

    # IMPORTANT: plain Python float (a 0-d numpy array breaks LazyTensor.__rmul__)
    two_pi = float(2.0 * pi)

    rng = np.random.default_rng(seed)

    if warmstart is not None:
        x = np.asarray(warmstart, dtype=real_dtype).reshape(N, D)
    else:
        x = rng.uniform(size=(N, D)).astype(real_dtype)
    x = np.ascontiguousarray(x)

    # --- modes: EXACTLY round(Chi*D*N) ±k pairs, half-space ---
    k_coords, r2 = exact_chi_modes(N, D, Chi, real_dtype, rng, torquato=torquato)
    k_coords = np.ascontiguousarray(k_coords)

    M = k_coords.shape[0]                       # number of independent modes
    n_eff = (N - 1) if torquato else N
    assert M == int(round(Chi * D * n_eff)), (M, Chi, D, N)
    chi_eff = M / (D * n_eff)
    assert abs(chi_eff - Chi) <= 0.5 / (D * n_eff) + 1e-12, (chi_eff, Chi)

    rpow = (r2 + 1e-3) ** -1.0
    w = (rpow / rpow.max()).astype(real_dtype)

    # Mean over modes. fk(-k) = conj(fk(k)) => the half-space gives the same
    # loss as the full space, and the same gradient up to a factor 2
    # (cancelled by the RMS normalisation below).
    norm = float(max(M, 1))
    r2max = float(r2.max())

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
            f"nufft unavailable ({D}D currently unsupported by finufft)...\n"
            f"...keops-radical fallback (slow, O(N^2) instead of NlogN)\n"
            f"should be ok for N <= 2k/4k on CPU, or 10k/20K on GPU\n"
            f"[keops-rad] "
            f"N={N} D={D} Chi={Chi:.3f} (eff={chi_eff:.5f}) modes(+-k pairs)={M} "
            f"|k|^2_max={r2max:.0f} n_iter={n_iter} ({tgt_info})"
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