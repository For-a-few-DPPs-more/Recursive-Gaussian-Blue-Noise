"""
Images as densities.

Images are read as densities indexed [x, y], with y pointing up (mathematical
convention), pixel (i, j) at position (i, j) / S, and dark pixels mapped to high
density when ``invert=True``.
"""
from __future__ import annotations

import numpy as np

from .. import config

CFG = config.TARGETS


def load_image(path: str, shape=CFG.image_shape, invert: bool = True) -> np.ndarray:
    """Any image (colour or grey, any size) resampled to a density of `shape` summing to 1."""
    from PIL import Image
    img = Image.open(path).convert("L").resize(shape[::-1], Image.LANCZOS)
    rho = np.asarray(img, dtype=np.float64) / 255.0
    rho = rho.T[:, ::-1]                  # [row, col] (y down) -> [x, y] (y up)
    if invert:
        rho = 1.0 - rho
    return rho / rho.sum()


def image_fourier(path: str, k: np.ndarray, sign: int = 1, invert: bool = True) -> np.ndarray:
    """
    Fourier coefficients sum_n rho[n] exp(sign 2i pi k.n / S) of the image at the integer modes k (M, 2).

    The image is loaded at a resolution S >= 4 (max |k| + 1), so the modes are read from its FFT
    without resampling to a coarser grid.
    """
    k = np.asarray(k, dtype=np.int64).reshape(-1, 2)
    k_max = int(np.abs(k).max()) if k.size else 0
    S = max(CFG.image_shape[0], 4 * (k_max + 1))
    rho = load_image(path, (S, S), invert)
    full = np.fft.ifftn(rho) * rho.size                   # sum_n rho[n] exp(+2i pi k.n / S)
    out = full[k[:, 0] % S, k[:, 1] % S]
    return out if sign > 0 else np.conj(out)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic gaussian blur of a 2D grid (sigma in pixels), by FFT."""
    fx = np.fft.fftfreq(img.shape[0])[:, None]
    fy = np.fft.fftfreq(img.shape[1])[None, :]
    g = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * g))


def sample_image(path: str, n: int, invert: bool = True, rng=None) -> np.ndarray:
    """n i.i.d. atoms (n, 2) distributed according to the image (slightly blurred, jittered)."""
    rng = np.random.default_rng() if rng is None else rng
    shape = CFG.image_shape
    density = blur(load_image(path, shape, invert), CFG.image_blur_sigma).clip(min=0).ravel()
    axis = np.linspace(0.0, 1.0, shape[0], endpoint=False)
    pixels = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    idx = rng.choice(len(density), size=n, replace=True, p=density / density.sum())
    return (pixels[idx] + CFG.image_jitter * rng.standard_normal((n, 2))) % 1.0
