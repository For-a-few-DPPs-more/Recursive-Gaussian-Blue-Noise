"""
Low-level mathematical helpers
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import finufft


# ──────────────────────────────────────────────────────────────────────────────
# Lattice helpers
# ──────────────────────────────────────────────────────────────────────────────

def drop_symmetric(directions: np.ndarray) -> np.ndarray:
    """
    Keep only one representative from each direction pair {v, -v}.

    The canonical representative is the one whose *first non-zero component*
    is positive.

    Parameters
    ----------
    directions : (M, D) int array

    Returns
    -------
    (K, D) int array  with K ≤ M // 2 + 1
    """
    first_nz_idx = (directions != 0).argmax(axis=1)
    first_nz_val = directions[np.arange(len(directions)), first_nz_idx]
    return directions[first_nz_val > 0]


def integers_in_half_ball(radius: float, D: int) -> np.ndarray:
    """
    Return all non-zero integer lattice vectors inside a sphere of *radius*,
    keeping only one vector per direction pair.

    Parameters
    ----------
    radius : float
    D : int

    Returns
    -------
    (M, D) int32 array
    """
    if radius <= 0.9:
        return np.zeros((0, D), dtype=np.int32)
    if radius <= 1.9:
        return np.eye(D, dtype=np.int32)

    r   = np.arange(-np.ceil(radius), np.ceil(radius) + 1)
    pts = np.stack(np.meshgrid(*(r,) * D, indexing="ij"), axis=-1).reshape(-1, D)
    d2  = np.sum(pts ** 2, axis=-1)
    return drop_symmetric(pts[(d2 > 0) & (d2 <= radius ** 2)])


def simplex(D: int) -> np.ndarray:
    """
    Vertices of a regular simplex centred at the origin in R^D.

    Returns
    -------
    (D+1, D) float64 array
    """
    if D == 1:
        return np.array([-1.0, 1.0])[:, None]
    null = np.zeros((D, 1))
    tip  = np.zeros((1, D))
    tip[0, -1] = 1.0
    base = np.hstack((simplex(D - 1), null))
    return np.vstack((np.sqrt(1.0 - (1.0 / D) ** 2) * base - tip / D, tip))


def grid_shape(N: int, D: int) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    """
    Smallest D-hypercube grid that contains at least *N* points.

    Returns
    -------
    IJK   : shape tuple  e.g. (32, 32) for D=2
    total : total number of grid slots  (I^D)
    axes  : tuple(range(D))
    """
    I    = int(np.ceil(N ** (1.0 / D)))
    IJK  = (I,) * D
    return IJK, I ** D, tuple(range(D))


# ──────────────────────────────────────────────────────────────────────────────
# Torus geometry  (JAX)
# ──────────────────────────────────────────────────────────────────────────────

def torus_wrap(x: jnp.ndarray) -> jnp.ndarray:
    """Wrap coordinates into [0, 1)^D."""
    return x - jnp.floor(x)


def torus_delta(delta: jnp.ndarray) -> jnp.ndarray:
    """Shortest signed displacement on the unit torus."""
    return delta - jnp.round(delta)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient / status helpers  (JAX)
# ──────────────────────────────────────────────────────────────────────────────

def clean_grad(x: jnp.ndarray) -> jnp.ndarray:
    """Replace NaN gradient contributions (fictive points) with 0."""
    return jnp.nan_to_num(x, nan=0.0)

# ──────────────────────────────────────────────────────────────────────────────
# Wave-vector preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_wave_vectors(
    Ks: np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build JAX arrays for the spectral gradient.

    Parameters
    ----------
    Ks : (M, D) integer wave-vector matrix

    Returns
    -------
    K_w : complex array of shape (M, D)  — phase multipliers
    K_  : complex array of shape (M, D)  — normalised duals
    """
    D = Ks.shape[-1]
    K  = (2.0 * jnp.pi * Ks * 1j)
    Kn = (jnp.abs(K) ** D).sum(axis=-1, keepdims=True)
    return K, -K / Kn


# ──────────────────────────────────────────────────────────────────────────────
# Grid initialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def prepare_points(
    x: np.ndarray | None,
    N_asked: int,
    IJK: tuple[int, ...],
    D: int,
) -> jnp.ndarray:
    """
    Pad *N_asked* real points to fill the I^D grid.

    Fictive slots receive a NaN coordinate so gradients ignore them.

    Parameters
    ----------
    x       : (N_asked, D) array or *None* (random initialisation).
    N_asked : number of real points.
    IJK     : grid shape tuple.
    D       : spatial dimension.

    Returns
    -------
    jnp.ndarray of shape (*IJK, D)
    """
    if x is None:
        x = np.random.rand(N_asked, D)
    else:
        x = np.asarray(x).reshape(N_asked, D)

    total             = int(np.prod(IJK))
    xfull             = np.random.rand(total, D)
    xfull[:N_asked] = x
    xfull[N_asked:]  = np.nan   # status = NaN → fictive
    return jnp.array(xfull.reshape(*IJK, D))

def random_rotations(x, batch_size, Dout, Din):
    Q, _      = np.linalg.qr(np.random.randn(batch_size, Dout, Din))
    offsets   = np.einsum(
        "nij,kj->nki", Q, x
    )
    return offsets

def sample_wave_vectors(kmed: int, kmax: int, D: int, n_high: int) -> np.ndarray:
    # ─────────────────────────────
    # LOW k : exhaustive lattice
    # ─────────────────────────────

    low = integers_in_half_ball(kmed, D)

    # ─────────────────────────────
    # HIGH k : isotropic sampling
    # ─────────────────────────────

    dirs = np.random.normal(size=(n_high, D))
    norm_dirs = np.linalg.norm(dirs, axis=1, keepdims=True)
    
    dirs = dirs / norm_dirs

    r = np.random.uniform(kmed, kmax, size=(len(dirs), 1))
    high = np.rint(r * dirs).astype(int)

    # remove zeros + duplicates 
    vecs = np.concatenate([low, high], axis=0)
    vecs = vecs[np.any(vecs != 0, axis=1)]
    
    return np.unique(vecs, axis=0)

# ──────────────────────────────────────────────────────────────────────────────
# Structure factor
# ──────────────────────────────────────────────────────────────────────────────
def structure_factor(
    points: np.ndarray,
    resolution: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the radial structure factor S(kint) via scattering intensity.

    Parameters
    ----------
    points     : (N, D) array of point coordinates in [0, 1)^D.
    resolution : number of sampled wave vectors used to estimate sf
                 (only used when D >= 4; ignored for D <= 3 where a full
                 Fourier grid is computed via FINUFFT).

    Returns
    -------
    kint : (M, D) int array — integer wave numbers.
    S : (M,) float array — S(k) values.

    Note
    ----
    - When D <= 3 the structure factor is evaluated exactly (up to the
      requested FINUFFT tolerance) on a Cartesian grid of integer modes
      that covers frequencies up to approximately 2 * N^{1/D}.  All
      modes are returned (sorted by magnitude).
    - When D >= 4 a Monte-Carlo sampling of wave-vectors is performed.
      Beyond a radius fixed to capture 1/4 of the "resolution" budget,
      ALL allowed wavevectors are sampled to get maximal precision on
      low frequencies.  The remaining budget is shared evenly across
      all pertinent frequency scales.

    Warning
    -------
    1. We insist that the sampling domain of the points have to be the
       unit hypercube [0, 1)**D for correct estimation.  This will NOT
       be checked.
    2. The integer wave numbers e.g kint = [nx, ny, ...] are returned, and not the wave 
    vectors, which are given as k = 2pi/L x kint .
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1]).astype(np.float64)
    N, D = pts.shape

    kunit = N ** (1.0 / D)
    kmax = 2 * kunit

    if D <= 3:
        # ------------------------------------------------------------------
        # Exact evaluation on a full integer-mode grid via FINUFFT (type-1)
        # ------------------------------------------------------------------
        # Number of modes per dimension so that |n|_∞ ≈ kmax
        n_modes = 2 * int(np.ceil(kmax)) + 1          # odd → centred on 0

        # Non-uniform points in [0, 2π)
        x = 2.0 * np.pi * pts.T                       # shape (D, N)
        c = np.ones(N, dtype=np.complex128)

        if D == 1:
            fk = finufft.nufft1d1(x[0].copy(), c, n_modes, eps=1e-8, isign=1)
            # mode indices
            n = np.arange(-(n_modes // 2), n_modes - (n_modes // 2))
            kint = n[:, None]
        elif D == 2:
            fk = finufft.nufft2d1(x[0].copy(), x[1].copy(), c, (n_modes, n_modes),
                                 eps=1e-8, isign=1)
            n = np.arange(-(n_modes // 2), n_modes - (n_modes // 2))
            nx, ny = np.meshgrid(n, n, indexing="ij")
            kint = np.stack([nx.ravel(), ny.ravel()], axis=1)
            fk = fk.ravel()
        else:  # D == 3
            fk = finufft.nufft3d1(x[0].copy(), x[1].copy(), x[2].copy(), c,
                                 (n_modes, n_modes, n_modes),
                                 eps=1e-8, isign=1)
            n = np.arange(-(n_modes // 2), n_modes - (n_modes // 2))
            nx, ny, nz = np.meshgrid(n, n, n, indexing="ij")
            kint = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
            fk = fk.ravel()

        Sk = np.abs(fk) ** 2 / N
        knorm = np.linalg.norm(kint, axis=1) / kunit

    else:
        # ------------------------------------------------------------------
        # Classical Monte-Carlo sampling of wave-vectors (D >= 4)
        # ------------------------------------------------------------------
        kmed = max(int((resolution / 4.0) ** (1.0 / D)), 1)

        # Edge case: handle kmed potentially larger or equal to kmax
        if kmax <= kmed:
            kmax = kmed + 1

        n_high = int(resolution * 3.0 / 4.0)
        kint = sample_wave_vectors(kmed, kmax, D, n_high)

        kvecs = jnp.array(2.0 * np.pi * kint)
        pts_j = jnp.array(pts)

        def Sk_one(k: jnp.ndarray) -> jnp.ndarray:
            rho = jnp.sum(jnp.exp(1j * (pts_j @ k)), axis=0)
            return jnp.abs(rho) ** 2 / N

        Sk = np.asarray(jax.lax.map(Sk_one, kvecs))
        knorm = np.sqrt(np.sum(kint**2, axis=1)) / kunit

    isnt0 = ~(kint == 0).all(axis = 1)
    kint, Sk, knorm = kint[isnt0], Sk[isnt0], knorm[isnt0]

    # Sort by wave-vector magnitude for convenience
    sort_idx = np.argsort(knorm)
    return kint[sort_idx], Sk[sort_idx]

def structure_factor_and_average(points, resolution: int = 20000, min_val: float = 1e-20):
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

    # --- Compute structure factor and normalise wave numbers ---
    kint, Sraw = structure_factor(pts, resolution=resolution//10)
    kunit = N ** (1 / D)

    k2_ = (kint ** 2).sum(axis=1)

    # Bin distant wave vectors to reduce clutter on the plot
    bin_params = {1: (10, 50), 2: (5, 25), 3: (2, 6)}
    if D in bin_params:
        groupbin, groupstart = bin_params[D]
        mask = k2_ >= groupstart
        k2_[mask] = groupbin * (k2_[mask] // groupbin)

    kraw = np.sqrt(k2_) / kunit
    Sraw = Sraw.clip(min=min_val)
    
    # --- Radial average: group by k² bin ---
    k2group_, inverse = np.unique(k2_, return_inverse=True)
    counts = np.bincount(inverse)
    Sgroup = np.bincount(inverse, weights=Sraw) / counts
    kgroup = np.sqrt(k2group_) / kunit

    if len(kraw) >= resolution:
        target = resolution
        p = kraw**(-D)
        p *= target / p.sum()
        keep = np.random.random(len(kraw)) <  p
        kraw = kraw[keep]
        Sraw = Sraw[keep]

    logk = np.log(kgroup)
    logS = np.log(Sgroup)

    # Regular grid in log(k)
    logk_uniform = np.linspace(logk[0], logk[-1], 1000)
    logS_uniform = np.interp(logk_uniform, logk, logS)

    # Gaussian width in log(k)
    sigma = (logk[-1] - logk[0]) * 0.01
    dx = logk_uniform[1] - logk_uniform[0]
    sigma_pixels = sigma / dx

    # Truncated Gaussian convolution
    logS_smooth_uniform = gaussian_filter1d(
        logS_uniform,
        sigma_pixels,
        truncate=4.0,
    )

    # Back to original k positions
    logS_smooth = np.interp(
        logk,
        logk_uniform,
        logS_smooth_uniform,
    )

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

    # Pad at the boundaries by reflection
    xp = np.pad(x, radius, mode="reflect")

    # Sliding windows: shape (len(x), 2 * radius + 1)
    windows = np.lib.stride_tricks.sliding_window_view(
        xp, 2 * radius + 1
    )

    return windows @ kernel