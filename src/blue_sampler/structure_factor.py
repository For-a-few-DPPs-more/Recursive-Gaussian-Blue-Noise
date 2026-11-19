"""
Structure factor estimation.

Two modes, selected by ``trust_boundary``
-----------------------------------------
trust_boundary=True (default, reference)
    The boundary of [0, 1)^D is trusted: plain scattering intensity
    S(n) = |sum_j exp(2i pi n.x_j)|^2 / N on the integer wave vectors of [0, 1)^D.
    Exact for point sets that "self compensate" for boundary effects,
    e.g. because they were sampled with a periodic procedure on the unit torus.

trust_boundary=False
    The boundary is not trusted. Tests the *bulk* structure of the point set,
    independently of how its boundary was handled by the sampler (torus, hard
    cube, exotic domain):

    1. Truncation: keep only the points of the inner cube [m, 1-m]^D, a window the sampler never "saw" as a boundary, and rescale it to [0, 1)^D.
    2. Tapering: weight each point by a separable Kaiser-Bessel taper
       h(x) = prod_d I0(beta sqrt(1 - (2 x_d - 1)^2)) / norm, a closed-form
       approximation of the optimal (Slepian / DPSS) taper. Its spectral leakage
       is ~exp(-2 beta) instead of ~1/q^2 for a hard cut.
    3. Debiasing with the *promised* density rho = N (points per unit volume):
       F(n) = sum_j h(x_j) e^{2i pi n.x_j} - rho |W| H(n),   S(n) = |F(n)|^2 / (rho |W|),
       where H is the analytic Fourier transform of the taper.

    Trade-off: the spectral window of the taper has a half-width of about
    beta / pi wave-number units of the truncated window. Estimates at
    |n| < beta/pi + 1 (see ``tapered_kmin``) are dominated by this
    resolution limit and should not be interpreted.
    Rule of thumb: leakage <= 1e-10 needs beta ~ 15 and a margin of ~5 modes.
    By default beta is chosen by ``auto_beta`` (15 when there are enough points per axis, smaller in high dimension).

Wave vectors: exhaustive grid with NUFFT for D <= 3, lattice + random sampling
with brute-force sums for D >= 4 (numpy or cupy, depending on ``device``).
"""

from __future__ import annotations


import numpy as np

from . import config
from .array_lib import get_backend
from .lattice import integer_grid, isotropic_wave_vectors

CFG = config.STRUCTURE_FACTOR

__all__ = [
    "structure_factor",
    "structure_factor_and_average",
    "tapered_kmin",
    "auto_beta",
]

DEFAULT_MARGIN = CFG.margin
DEFAULT_BETA = None          # None -> auto_beta()


# ---------------------------------------------------------------------------
# Kaiser-Bessel taper on [0, 1]
# ---------------------------------------------------------------------------

def _kaiser_norm(beta: float) -> float:
    """sqrt(int_0^1 I0(beta sqrt(1 - (2x - 1)^2))^2 dx), by Gauss-Legendre quadrature."""
    u, w = np.polynomial.legendre.leggauss(400)          # u in [-1, 1], x = (u + 1) / 2
    return float(np.sqrt(0.5 * np.sum(w * np.i0(beta * np.sqrt(1.0 - u ** 2)) ** 2)))


def _kaiser_weights(pts: np.ndarray, beta: float) -> np.ndarray:
    """Separable taper h(x_j) (normalised: int_[0,1]^D h^2 = 1), shape (N,)."""
    u = 2.0 * np.clip(pts, 0.0, 1.0) - 1.0
    w = np.i0(beta * np.sqrt(1.0 - u ** 2)) / _kaiser_norm(beta)
    return np.prod(w, axis=1)


def _kaiser_ft(kint: np.ndarray, beta: float) -> np.ndarray:
    """H(n) = int_[0,1]^D h(x) exp(2i pi n.x) dx, analytic, shape (M,).

    Uses int_{-1}^{1} I0(beta sqrt(1-u^2)) e^{i w u} du = 2 sinh(sqrt(beta^2 - w^2)) / sqrt(beta^2 - w^2).
    """
    norm = _kaiser_norm(beta)
    H = np.ones(len(kint), dtype=complex)
    for d in range(kint.shape[1]):
        n = kint[:, d].astype(float)
        r = np.sqrt((beta ** 2 - (np.pi * n) ** 2).astype(complex))
        safe = np.abs(r) > 1e-12
        g = np.ones_like(r)                                   # limit sinh(r)/r -> 1
        g[safe] = np.sinh(r[safe]) / r[safe]
        H *= np.real(g) * np.cos(np.pi * n) / norm           # (-1)^n from the shift x -> u
    return H


def auto_beta(n_points: int, D: int, margin: float = DEFAULT_MARGIN) -> float:
    """Kaiser parameter used when beta=None.

    The spectral window of the taper spans ~beta/pi modes, while the truncated window
    only holds n_axis = (rho|W|)^(1/D) modes per axis up to the inter-particle scale.
    We take beta = min(15, pi * n_axis / 4): leakage ~1e-13 whenever there are enough
    points (typically D <= 3), otherwise the main lobe is capped at a quarter of the
    useful range and the leakage floor rises to ~exp(-2 beta) (curse of dimension:
    certifying a low bulk S(k) in high D requires a huge number of points).
    """
    n_axis = (n_points * (1.0 - 2.0 * margin) ** D) ** (1.0 / D)
    return float(min(CFG.max_beta, np.pi * n_axis / 4.0))


def tapered_kmin(n_points: int, D: int, margin: float = DEFAULT_MARGIN,
                 beta: float | None = DEFAULT_BETA) -> float:
    """Smallest trustworthy normalised wave number (k = 1 <-> |n| = (rho|W|)^(1/D))
    when trust_boundary=False (truncated + tapered estimate)."""
    if beta is None:
        beta = auto_beta(n_points, D, margin)
    n_exp = n_points * (1.0 - 2.0 * margin) ** D
    return (beta / np.pi + 1.0) / n_exp ** (1.0 / D)


# ---------------------------------------------------------------------------
# Fourier sums F(n) = sum_j c_j exp(2i pi n.x_j)
# ---------------------------------------------------------------------------

def _grid(n_modes: int, D: int) -> np.ndarray:
    """Wave vectors of a centred cube of n_modes^D modes, in finufft's (modeord=0) order."""
    return integer_grid(np.arange(-(n_modes // 2), n_modes - (n_modes // 2)), D)


def _sums_nufft(pts, c, n_modes, bk, eps):
    xp, nufft = bk.xp, bk.nufft
    N, D = pts.shape
    x = 2.0 * xp.pi * xp.asarray(pts, dtype=bk.real).T
    c = xp.asarray(c, dtype=bk.complex)
    shape = (n_modes,) * D
    if D == 1:
        fk = nufft.nufft1d1(x[0].copy(), c, n_modes, eps=eps, isign=1)
    elif D == 2:
        fk = nufft.nufft2d1(x[0].copy(), x[1].copy(), c, shape, eps=eps, isign=1)
    else:
        fk = xp.zeros(shape, dtype=bk.complex)
        for s in range(0, N, CFG.nufft_chunk):
            e = min(s + CFG.nufft_chunk, N)
            fk += nufft.nufft3d1(x[0, s:e].copy(), x[1, s:e].copy(), x[2, s:e].copy(),
                                 c[s:e].copy(), shape, eps=eps, isign=1)
    return _grid(n_modes, D), bk.to_numpy(fk.ravel())


def _sums_xp(pts, c, kint, bk):
    """Brute force on the chosen device (numpy or cupy), chunked over points."""
    xp = bk.xp
    kv = xp.asarray(2.0 * np.pi * kint, dtype=bk.real)
    p = xp.asarray(pts, dtype=bk.real)
    cc = xp.asarray(c, dtype=bk.complex)
    M = kv.shape[0]
    chunk = max(1024, int(CFG.bruteforce_chunk_elems / max(M, 1)))
    F = xp.zeros(M, dtype=bk.complex)
    for s in range(0, len(p), chunk):
        e = min(s + chunk, len(p))
        F += cc[s:e] @ xp.exp(1j * (p[s:e] @ kv.T))
    return bk.to_numpy(F)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _structure_factor_core(points, resolution, device, precision, trust_boundary,
                           margin, beta, backend):
    """Return (kint, S, n_exp): wave vectors of the analysed window, S, expected count rho|W|."""
    pts = np.asarray(points, dtype=float).reshape(-1, np.asarray(points).shape[-1])
    N, D = pts.shape
    taper = not trust_boundary

    if taper:
        side = 1.0 - 2.0 * margin
        if not 0.0 < side <= 1.0:
            raise ValueError("margin must be in [0, 0.5)")
        keep = np.all((pts >= margin) & (pts < 1.0 - margin), axis=1)
        pts = (pts[keep] - margin) / side
        n_exp = N * side ** D                      # promised density x volume of the window
        if beta is None:
            beta = auto_beta(N, D, margin)
        weights = _kaiser_weights(pts, beta)
    else:
        n_exp = float(N)
        weights = np.ones(len(pts))

    kunit = float(n_exp ** (1.0 / D))
    kmax = CFG.k_max * kunit
    bk = get_backend(device, precision)
    eps = CFG.eps[bk.precision]

    if backend == "auto":
        backend = "nufft" if D <= 3 else "bruteforce"

    if backend == "nufft":
        if D > 3:
            raise ValueError("NUFFT backend only supports D <= 3")
        kint, F = _sums_nufft(pts, weights, int(np.ceil(kmax)) + 1, bk, eps)
    else:
        if D <= 3:
            kint = _grid(int(np.ceil(kmax)) + 1, D)
        else:
            k_low = max(int((resolution * CFG.low_mode_fraction) ** (1.0 / D)), 1)
            if kmax <= k_low:
                kmax = k_low + 1
            kint = isotropic_wave_vectors(k_low, kmax, D, int(resolution * (1.0 - CFG.low_mode_fraction)))
        if backend == "bruteforce":
            F = _sums_xp(pts, weights, kint, bk)
        else:
            raise ValueError(f"unknown backend {backend!r}")

    if taper:
        F = F - n_exp * _kaiser_ft(kint, beta)
    S = np.abs(F) ** 2 / n_exp

    isnt0 = ~np.all(kint == 0, axis=1)
    kint, S = kint[isnt0], S[isnt0]
    order = np.argsort((kint ** 2).sum(axis=1), kind="stable")
    return kint[order], S[order], n_exp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def structure_factor(
    points: np.ndarray,
    resolution: int = 2000,
    device="auto",
    precision="float32",
    trust_boundary: bool = True,
    margin: float = DEFAULT_MARGIN,
    beta: float | None = DEFAULT_BETA,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate S(k) via scattering intensity.

    Parameters
    ----------
    points         : (N, D) array of point coordinates in [0, 1)^D.
    resolution     : number of sampled wave vectors (only used when D >= 4).
    device         : "auto" | "cpu" | "gpu".
    precision      : "float32" | "float64". Use float64 to resolve S below ~1e-6.
    trust_boundary : True (default): plain estimate on the whole cube.
                     False: truncate to [margin, 1 - margin]^D, Kaiser-taper, debias
                     with the promised density (see module docstring).
    margin         : width removed on each side (trust_boundary=False only).
    beta           : Kaiser parameter (trust_boundary=False only). Larger -> less
                     leakage (~exp(-2 beta)) but coarser resolution (~beta/pi modes).
                     None (default) -> auto_beta(N, D, margin).
    backend        : "auto" | "nufft" | "bruteforce".

    Returns
    -------
    kint : (M, D) int array of wave vectors of the analysed window (k = 2 pi kint),
           sorted by norm. With trust_boundary=False they refer to the rescaled
           truncated window.
    S    : (M,) array.

    Note
    ----
    Points must lie in the unit hypercube [0, 1)^D. This is not checked.
    """
    kint, S, _ = _structure_factor_core(points, resolution, device, precision,
                                        trust_boundary, margin, beta, backend)
    return kint, S


def structure_factor_and_average(
    points,
    resolution: int = 20000,
    min_val: float = 1e-20,
    precision="float32",
    trust_boundary: bool = True,
    margin: float = DEFAULT_MARGIN,
    beta: float | None = DEFAULT_BETA,
    backend: str = "auto",
):
    """
    Compute the structure factor of a point set and return raw and radially-averaged curves.

    ``trust_boundary`` behave as in
    ``structure_factor``.

    Returns
    -------
    kraw, Sraw     : normalised wave numbers and raw S values (subsampled for display).
    kgroup, Sgroup : radially averaged then log-k smoothed curve.

    Wave numbers are normalised so that k = 1 corresponds to |n| = (rho |W|)^(1/D),
    i.e. to the inter-particle distance, in both modes.
    Group averages are taken on raw values (which can be slightly negative after
    debiasing) and clipped to ``min_val`` afterwards.
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    D = pts.shape[1]
    kint, Sraw, n_exp = _structure_factor_core(pts, resolution // CFG.resolution_divisor, "auto", precision,
                                               trust_boundary, margin, beta, backend)
    kunit = n_exp ** (1.0 / D)
    k2_ = (kint ** 2).sum(axis=1)

    if D in CFG.radial_bins:
        groupbin, groupstart = CFG.radial_bins[D]
        mask = k2_ >= groupstart
        k2_[mask] = groupbin * (k2_[mask] // groupbin)

    kraw = np.sqrt(k2_) / kunit
    k2group_, inverse = np.unique(k2_, return_inverse=True)
    counts = np.bincount(inverse)
    Sgroup = (np.bincount(inverse, weights=Sraw) / counts).clip(min=min_val)
    kgroup = np.sqrt(k2group_) / kunit
    Sraw = Sraw.clip(min=min_val)

    if len(kraw) >= resolution:
        p = kraw ** (-D)
        p *= resolution / p.sum()
        keep = np.random.random(len(kraw)) < p
        kraw, Sraw = kraw[keep], Sraw[keep]

    logk = np.log(kgroup)
    logk_u = np.linspace(logk[0], logk[-1], 1000)
    S_u = np.interp(logk_u, logk, Sgroup)
    sigma_pixels = (logk[-1] - logk[0]) * CFG.smoothing / (logk_u[1] - logk_u[0])
    Sgroup = np.interp(logk, logk_u, gaussian_filter1d(S_u, sigma_pixels, truncate=4.0)).clip(min=min_val)
    Sraw = Sraw.clip(min=Sgroup.min())
    return kraw, Sraw, kgroup, Sgroup


def gaussian_filter1d(x, sigma, truncate=4.0):
    """Gaussian smoothing of a 1D array using a truncated kernel."""
    radius = int(np.ceil(truncate * sigma))
    if radius == 0:
        return x.copy()
    kernel_x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= kernel.sum()
    xp = np.pad(x, radius, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(xp, 2 * radius + 1)
    return windows @ kernel
