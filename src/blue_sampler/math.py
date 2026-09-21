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