"""
Target side of adaptive Gaussian blue noise: the image seen through Gaussians of every size.

`grad[k]` is the gradient of (rho * g_t), t = T0 * 2^k (g_t: unit-mass Gaussian of variance t
per axis), so a dot reads its attraction between two levels of this pyramid, at its own
kernel size. Grids are indexed [x, y] (y up), samples at the cell centres.
"""
from __future__ import annotations

import numpy as np

from ..targets import as_target, for_adaptive

G = 512
T0 = 0.25 / G ** 2          # smallest kernel variance: half a pixel of standard deviation
LEVELS = 14                 # one octave of variance each: up to a standard deviation of ~45 pixels


class Density:
    def __init__(self, image, invert: bool = True):
        """image: path, point set (K, 2) (pseudo image, see `targets.points_to_density`), `targets.Target`,
        or a non-negative (G, G) density grid indexed [x, y].
        The dots live in the unit square; `from_unit` / `to_unit` map them to / from the target's frame."""
        if isinstance(image, np.ndarray) and image.shape == (G, G):
            self.target = None
            rho = image.astype(np.float64)
        else:
            self.target = as_target(image, 2)
            rho = for_adaptive(self.target, (G, G), invert)
        self.rho = (rho / rho.mean()).astype(np.float32)            # mean 1
        self.rho_hat = np.fft.fft2(self.rho.astype(np.float64))
        f = np.fft.fftfreq(G, 1.0 / G)
        self.fx, self.fy = np.meshgrid(f, f, indexing="ij")
        f2 = self.fx ** 2 + self.fy ** 2
        grad = np.empty((LEVELS, G, G, 2), np.float32)
        for k in range(LEVELS):
            g = np.exp(-2 * np.pi ** 2 * T0 * 2 ** k * f2)
            z = np.fft.ifft2(self.rho_hat * 2j * np.pi * g * (self.fx + 1j * self.fy))  # re: d/dx, im: d/dy
            grad[k, ..., 0], grad[k, ..., 1] = z.real, z.imag
        self.grad = grad

    def to_unit(self, x) -> np.ndarray:
        return np.asarray(x) if self.target is None else self.target.to_unit(x)

    def from_unit(self, u) -> np.ndarray:
        return u if self.target is None else self.target.from_unit(u)

    def smooth(self, t: float) -> np.ndarray:
        """rho * g_t on the grid (mean 1)."""
        g = np.exp(-2 * np.pi ** 2 * t * (self.fx ** 2 + self.fy ** 2))
        return np.fft.ifft2(self.rho_hat * g).real.astype(np.float32)


def bilinear(p):
    """Corner indices (4, n, 2) and weights (4, n) of points p (n, 2) of the unit torus (jax or numpy)."""
    xp = np if isinstance(p, np.ndarray) else __import__("jax.numpy", fromlist=["numpy"])
    u = p * G - 0.5
    f = xp.floor(u)
    a, b = (u - f)[:, 0], (u - f)[:, 1]
    i0 = f.astype(xp.int32) % G
    i1 = (i0 + 1) % G
    x0, y0, x1, y1 = i0[:, 0], i0[:, 1], i1[:, 0], i1[:, 1]
    idx = xp.stack([xp.stack([x0, y0], -1), xp.stack([x0, y1], -1),
                    xp.stack([x1, y0], -1), xp.stack([x1, y1], -1)])
    w = xp.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b])
    return idx, w


def interpolate(grid, p):
    """Bilinear interpolation of a (G, G, ...) grid at points p (n, 2)."""
    idx, w = bilinear(p)
    vals = grid[idx[..., 0], idx[..., 1]]                    # (4, n, ...)
    return (w.reshape(w.shape + (1,) * (vals.ndim - 2)) * vals).sum(0)
