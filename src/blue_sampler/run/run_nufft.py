from __future__ import annotations

import itertools
import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from ..math import kdtree_order

# ---------------------------------------------------------------------------
# Helpers (équivalents NumPy des fonctions de ..math)
# ---------------------------------------------------------------------------

def torus_delta(x: NDArray) -> NDArray:
    """Distance minimale sur le tore [0, 1)^D (version vectorisée)."""
    return x - np.round(x)


def torus_wrap(x: NDArray) -> NDArray:
    """Ramène les coordonnées dans [0, 1)."""
    return np.mod(x, 1.0)


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------

def get_shifts(radius: int, dims: int) -> NDArray:
    ranges = [range(-radius, radius + 1)] * dims
    shifts = [
        s for s in itertools.product(*ranges)
        if sum(v * v for v in s) <= radius * radius
    ]
    return np.asarray(shifts, dtype=np.int32)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def _nufft_pipeline(
    N: int,
    D: int,
    lr: float = 1.0,
    kfrac: float = 1.0,
    warmstart: Optional[NDArray] = None,
    verbose: int = 1,
    n_iter: int = 200,
    regrid_every: int = 30,
) -> NDArray:
    """
    Generalised NUFFT-style point set optimisation (version pure NumPy).
    """
    # Taille de grille (puissance de 2)
    G = 2 ** int(np.log2(N ** (1.0 / D)) + 1e-6)

    if D == 2:
        r = 5
    elif D == 3:
        r = 4
    else:  # D >= 4
        r = 3

    shifts = get_shifts(r, D)
    cell_size = G ** D
    assert N % cell_size == 0, (
        f"N={N} must be divisible by G**D with G power of 2 "
        f"(got G={G}, cell_size={cell_size})"
    )
    n_cells = N // cell_size

    log = (lambda msg: print(msg)) if verbose >= 1 else (lambda msg: None)

    K_RADIUS = max(1, int(G * kfrac))
    SIGMA2 = (1.0 / G) ** 2
    base_delta = 0.01 / G * lr
    AXES = tuple(range(D))

    # Grille spatiale
    g_axis = np.linspace(0.0, 1.0, G, endpoint=False)
    GRID = np.stack(
        np.meshgrid(*([g_axis] * D), indexing="ij"),
        axis=-1,
    )[..., None, :]  # shape (G,)*D + (1, D)

    # Poids fréquentiels
    freqs = np.fft.fftfreq(G) * G
    freq_grids = np.meshgrid(*([freqs] * D), indexing="ij")
    freq_r2 = sum(f ** 2 for f in freq_grids)

    MASK = (freq_r2 > K_RADIUS ** 2) | (freq_r2 <= 0.01)
    K_WEIGHT = np.where(MASK, 0.0, 1.0 / freq_r2)
    K_WEIGHT = K_WEIGHT[..., None]  # broadcast sur les cellules

    # ------------------------------------------------------------------
    # Loss + gradient (manuel)
    # ------------------------------------------------------------------
    def _loss_and_grad(x_grid: NDArray, shifts: NDArray):
        # x_grid : (G,)*D + (n_cells, D)

        # Forward : densité
        dens = np.zeros((G,) * D + (n_cells,), dtype=x_grid.dtype)
        for shift in shifts:
            rolled = np.roll(x_grid, shift, axis=AXES)
            delta = torus_delta(GRID - rolled)
            dens_contrib = np.exp(-np.sum(delta ** 2, axis=-1) / SIGMA2)
            dens += dens_contrib

        # FFT + loss
        F = np.fft.fftn(dens, axes=AXES)
        loss = np.sum((np.abs(F) ** 3) * K_WEIGHT)

        # Gradient dans le domaine fréquentiel
        grad_F = K_WEIGHT * F * np.abs(F)
        grad_dens = np.fft.ifftn(grad_F, axes=AXES).real

        # Backward
        grad_x = np.zeros_like(x_grid)
        for shift in shifts:
            rolled = np.roll(x_grid, shift, axis=AXES)
            delta = torus_delta(GRID - rolled)
            dist2 = np.sum(delta ** 2, axis=-1, keepdims=True)
            gpix = grad_dens[..., None] * np.exp(-dist2 / SIGMA2) * delta
            grad_x += np.roll(gpix, -np.asarray(shift), axis=AXES)

        return loss, grad_x

    # ------------------------------------------------------------------
    # Un chunk d'optimisation (remplace le scan jitté)
    # ------------------------------------------------------------------
    def _run_chunk(x_init: NDArray, n_steps: int, adap: float, prev_loss: float):
        x = x_init.copy()
        for _ in range(n_steps):
            loss, g = _loss_and_grad(x, shifts)
            rms = np.sqrt(np.mean(g ** 2) + 1e-30)
            x = torus_wrap(x - (g / rms) * adap)

            improved = loss < prev_loss
            adap = adap * 1.01 if improved else adap * 0.9
            prev_loss = loss

        return x, adap, prev_loss

    def _loss(x: NDArray) -> float:
        return float(_loss_and_grad(x, shifts)[0])

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    if warmstart is not None:
        if warmstart.shape != (N, D):
            raise ValueError(f"warmstart must be ({N}, {D}), got {warmstart.shape}")
        x_np = warmstart.astype(np.float64)
    else:
        x_np = np.random.rand(N, D).astype(np.float64)

    def to_grid(pts: NDArray) -> NDArray:
        return kdtree_order(pts, G=G)

    log(f"[nufft] {D}D  | {N} pts | {n_iter} iters")

    t0 = time.time()
    x_grid = to_grid(x_np)
    log(f"kdtree built (elapsed {time.time() - t0:.2f}s)")
    loss0 = _loss(x_grid)
    log(f"loss 0: {loss0:.2e}")

    adap = float(base_delta)
    prev_loss = np.inf

    # Premier chunk
    x_grid, adap, prev_loss = _run_chunk(x_grid, regrid_every, adap, prev_loss)
    flat = x_grid.reshape(-1, D)

    # Boucle principale
    t0 = time.time()
    adap = float(base_delta)
    prev_loss = np.inf

    for start in range(0, n_iter, regrid_every):
        log(f"loss {start + regrid_every}: {_loss(x_grid):.2e}")
        x_grid = to_grid(flat)

        x_grid, adap, prev_loss = _run_chunk(
            x_grid, regrid_every, adap, prev_loss
        )
        flat = x_grid.reshape(-1, D)

    x_grid = to_grid(flat)
    loss2 = _loss(x_grid)

    log(
        f"done in {time.time() - t0:.2f}s | "
        f"loss {loss0:.2e} → {loss2:.2e}"
    )

    return x_grid.reshape(-1, D)