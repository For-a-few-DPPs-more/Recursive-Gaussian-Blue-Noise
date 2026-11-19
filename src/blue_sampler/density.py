"""
Images as target densities (nufft, cstit; also read by `blue_sampler.adaptive`).

- `load_image`: image as a density on a grid.
- `image_to_points` / `image_spectrum`: image as atoms (cstit, nufft) or Fourier coefficients (nufft).

Images are read as densities indexed [x, y], with y pointing up (mathematical
convention) and dark pixels mapped to high density when ``invert=True``.
"""
from __future__ import annotations

import numpy as np

from . import config

CFG = config.DENSITY


# ── Images ────────────────────────────────────────────────────────────────────

def load_image(path: str, shape=CFG.image_shape, invert: bool = True) -> np.ndarray:
    """Any image (colour or grey, any size) resampled to a density of `shape` summing to 1."""
    from PIL import Image
    img = Image.open(path).convert("L").resize(shape[::-1], Image.LANCZOS)
    rho = np.asarray(img, dtype=np.float64) / 255.0
    rho = rho.T[:, ::-1]                  # [row, col] (y down) -> [x, y] (y up)
    if invert:
        rho = 1.0 - rho
    return rho / rho.sum()


def image_spectrum(path: str, shape, invert: bool = True) -> np.ndarray:
    """Fourier coefficients sum_n rho[n] exp(+2i pi k.n / G) on `shape`, FFT ordering
    (the finufft convention isign=+1, modeord=1)."""
    rho = load_image(path, shape, invert)
    return np.fft.ifftn(rho) * rho.size


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic gaussian blur by FFT (2D)."""
    size = int(6 * sigma + 1)
    t = np.arange(-size // 2 + 1, size // 2 + 1)
    X, Y = np.meshgrid(t, t)
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    pad = np.zeros_like(img)
    pad[:size, :size] = kernel / kernel.sum()
    pad = np.roll(pad, (-size // 2, -size // 2), axis=(0, 1))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)))


def image_to_points(path: str, n: int, oversample: int = 1) -> np.ndarray:
    """oversample * n i.i.d. atoms (2D) distributed according to the image."""
    shape = CFG.image_shape
    density = _blur(load_image(path, shape), CFG.image_blur_sigma).clip(min=0).ravel()
    axis = np.linspace(0.0, 1.0, shape[0], endpoint=False)
    pixels = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    idx = np.random.choice(len(density), size=oversample * n, replace=True, p=density / density.sum())
    return pixels[idx] + CFG.image_jitter * np.random.randn(oversample * n, 2)
