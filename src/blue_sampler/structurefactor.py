from __future__ import annotations
import numpy as np
from .math import sample_wave_vectors
from .gpu_setup import set_config

def structure_factor(
    points: np.ndarray,
    resolution: int = 2000,
    device="auto",
    precision="float32",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the radial structure factor S(kint) via scattering intensity.

    Parameters
    ----------
    points     : (N, D) array of point coordinates in [0, 1)^D.
    resolution : number of sampled wave vectors used to estimate sf
                 (only used when D >= 4; ignored for D <= 3 where a full
                 Fourier grid is computed via FINUFFT).
    device     : "auto" | "cpu" | "cuda" 
    precision  : "float32" | "float64"

    Returns
    -------
    kint : (M, D) int array — integer wave numbers
    S    : (M,) float array — S(k) values.

    Note
    ---
    The sampling domain of the points have to be the unit hypercube [0, 1)**D for correct estimation.  
    This will not be checked.
    """
    eps = 1e-5 if precision == "float32" else 1e-8
    cfg = set_config(device, precision, verbose=0)
    xp = cfg.xp
    real_dtype = cfg.real_dtype
    complex_dtype = cfg.complex_dtype
    to_numpy = cfg.to_numpy
    nufft_lib = cfg.nufft_lib

    # --- always work in the chosen backend from the start ---
    pts = xp.asarray(points, dtype=real_dtype).reshape(-1, points.shape[-1])
    N, D = pts.shape
    kunit = float(N ** (1.0 / D))
    kmax = 2.0 * kunit

    if D <= 3:
        n_modes = int(xp.ceil(kmax)) + 1
        x = 2.0 * xp.pi * pts.T          # (D, N)
        c = xp.ones(N, dtype=complex_dtype)
        n = xp.arange(-(n_modes // 2), n_modes - (n_modes // 2))

        if D == 1:
            kint = to_numpy(n[:, None])
            fk = to_numpy(nufft_lib.nufft1d1(x[0].copy(), c, n_modes, eps=eps, isign=1))
        elif D == 2:
            fk = nufft_lib.nufft2d1(
                x[0].copy(), x[1].copy(), c, (n_modes, n_modes),
                eps=eps, isign=1
            )
            nx, ny = xp.meshgrid(n, n, indexing="ij")
            kint = to_numpy(xp.stack([nx.ravel(), ny.ravel()], axis=1))
            fk = to_numpy(fk.ravel())
        else:  # D == 3
            max_chunk = 400_000
            fk = xp.zeros((n_modes, n_modes, n_modes), dtype=complex_dtype)
            for start in range(0, N, max_chunk):
                stop = min(start + max_chunk, N)
                fk += nufft_lib.nufft3d1(
                    x[0, start:stop].copy(),
                    x[1, start:stop].copy(),
                    x[2, start:stop].copy(),
                    c[start:stop].copy(),
                    (n_modes, n_modes, n_modes),
                    eps=eps,
                    isign=1,
                )
            nx, ny, nz = xp.meshgrid(n, n, n, indexing="ij")
            kint = to_numpy(xp.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1))
            fk = to_numpy(fk.ravel())

        Sk = np.abs(fk) ** 2 / N
        knorm = np.linalg.norm(kint, axis=1) / kunit

    else:
        # Monte-Carlo path – stay fully on the chosen device
        kmed = max(int((resolution / 4.0) ** (1.0 / D)), 1)
        if kmax <= kmed:
            kmax = kmed + 1
        n_high = int(resolution * 3.0 / 4.0)

        kint_np = sample_wave_vectors(kmed, kmax, D, n_high)  # returns NumPy
        kvecs = xp.asarray(2.0 * np.pi * kint_np, dtype=real_dtype)
        M = kvecs.shape[0]

        chunk_size = max(1024, int(8_000_000 / max(M, 1)))
        rho = xp.zeros(M, dtype=complex_dtype)

        for start in range(0, N, chunk_size):
            stop = min(start + chunk_size, N)
            phase = pts[start:stop] @ kvecs.T          # (chunk, M)
            rho += xp.sum(xp.exp(1j * phase), axis=0)

        Sk = to_numpy(xp.abs(rho) ** 2 / N)
        knorm = np.sqrt(np.sum(kint_np ** 2, axis=1)) / kunit
        kint = kint_np

    # --- final common path: always NumPy ---
    isnt0 = ~np.all(kint == 0, axis=1)
    kint, Sk, knorm = kint[isnt0], Sk[isnt0], knorm[isnt0]
    sort_idx = np.argsort(knorm)
    return kint[sort_idx], Sk[sort_idx]


def structure_factor_and_average(points, resolution: int = 20000, min_val: float = 1e-20, precision = "float32"):
    """
    Compute the structure factor of a point set and return raw and radially-averaged curves.
    Parameters
    ----------
    points : array-like
        Point coordinates, shape (N, D) or any shape with D as the last axis.
    resolution : int, optional
        Approximate number of sampled wave vectors in the final output. 
    min_val : float, optional
        Lower clip to avoid issues when taking logs. Default: 1e-20.
    Returns
    -------
    kraw : ndarray
        Normalised wave numbers for the raw scatter data.
    Sraw : ndarray
        Raw structure-factor values at each k.
    kgroup : ndarray
        Normalised wave numbers for the radially-averaged data.
    Sgroup : ndarray
        Smoothed radial average of the structure factor at each kgroup.
    
    Warning
    -------
    1. We insist that the sampling domain of the points have to be the
        unit hypercube [0, 1)**D for correct estimation.  This will NOT
        be checked.
    2. kraw and kgroup wave vector are expressed in normalised format,
        where k = 1 correspond to the limit frequency associated to
        the inter-particle distance delta = N**(-1/D) 
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    N, D = pts.shape
    kint, Sraw = structure_factor(pts, resolution=resolution // 10, precision = precision)
    kunit = N ** (1 / D)
    k2_ = (kint ** 2).sum(axis=1)

    bin_params = {1: (10, 50), 2: (5, 25), 3: (2, 6)}
    if D in bin_params:
        groupbin, groupstart = bin_params[D]
        mask = k2_ >= groupstart
        k2_[mask] = groupbin * (k2_[mask] // groupbin)

    kraw = np.sqrt(k2_) / kunit
    Sraw = Sraw.clip(min=min_val)

    k2group_, inverse = np.unique(k2_, return_inverse=True)
    counts = np.bincount(inverse)
    Sgroup = np.bincount(inverse, weights=Sraw) / counts
    kgroup = np.sqrt(k2group_) / kunit

    if len(kraw) >= resolution:
        target = resolution
        p = kraw ** (-D)
        p *= target / p.sum()
        keep = np.random.random(len(kraw)) < p
        kraw = kraw[keep]
        Sraw = Sraw[keep]

    logk = np.log(kgroup)
    logS = np.log(Sgroup)
    logk_uniform = np.linspace(logk[0], logk[-1], 1000)
    logS_uniform = np.interp(logk_uniform, logk, logS)
    sigma = (logk[-1] - logk[0]) * 0.01
    dx = logk_uniform[1] - logk_uniform[0]
    sigma_pixels = sigma / dx
    logS_smooth_uniform = gaussian_filter1d(logS_uniform, sigma_pixels, truncate=4.0)
    logS_smooth = np.interp(logk, logk_uniform, logS_smooth_uniform)
    Sgroup = np.exp(logS_smooth)
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