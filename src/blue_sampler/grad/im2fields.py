"""
Turn an image into a density usable as `target` in fields.py
(make_multi_scales_field_fun(im2field(path), ftype="density"))
"""
from __future__ import annotations
import numpy as np

def im2field(path, shape=(512, 512), invert=True):
    """Load any image (color, grayscale, any size/ratio) and resample it
    into a normalized density on `shape`. Pixels close to white get high
    density unless `invert=True` (use that for dark strokes on white)."""
    from PIL import Image
    img = Image.open(path).convert("L").resize(shape[::-1], Image.LANCZOS)
    rho = np.asarray(img, dtype=np.float32) / 255.0
    rho = (rho.T)[:, ::-1]
    if invert:
        rho = 1.0 - rho
    return rho / rho.sum()

def gaussian_kernel(size, sigma):
    x = np.arange(-size//2+1, size//2+1)
    X, Y = np.meshgrid(x, x)
    K = np.exp(-(X**2 + Y**2)/(2*sigma**2))
    return K / K.sum()

def fft_blur(img, sigma):
    size = int(6*sigma+1)
    kernel = gaussian_kernel(size, sigma)

    pad = np.zeros_like(img)
    h, w = kernel.shape
    pad[:h, :w] = kernel
    pad = np.roll(pad, -h//2, axis=0)
    pad = np.roll(pad, -w//2, axis=1)

    return np.real(np.fft.ifft2(
        np.fft.fft2(img) * np.fft.fft2(pad)
    ))

def _im2targ(path, N, oversample=3):
    density = im2field(path, shape=(512, 512), invert=True)
    density = fft_blur(density, sigma=1)

    density = density.clip(min = 0)

    g_axis = np.linspace(0.0, 1.0, 512, endpoint=False)
    grid = np.stack(np.meshgrid(*([g_axis] * 2), indexing="ij"), axis=-1).reshape(-1, 2)
    prob = density.ravel() / density.max()

    idx_candidates = np.random.choice(
        len(prob),
        size=oversample * N,
        replace=True,
        p=prob / prob.sum()         
    )
    candidates = grid[idx_candidates] + 1/2000 * np.random.randn(oversample * N, 2)

    
    
    return candidates[:N]