"""
Non-uniform target densities: an image, or a point set read as samples of the density.

    t = as_target("zebra.jpg")          # image (2D), dark pixels = dense
    t = as_target(points)               # point set (K, D), any bounding box

`as_target` is called once, by the public entry points (`sample_points`, `cstit`, the
spectral solver, the adaptive samplers, the extras). Everything below it receives a validated `Target`.

The solvers work in the unit cube. A point set lying outside of it is shrunk into it (same
factor on every axis, centred, with a margin since the cube is a torus); `Target.from_unit`
maps the solver output back, so the samples come out in the frame of the target.

Each solver family then needs the target in its own form:

    solver      image                           point set
    spectral    i.i.d. points drawn from it     the points themselves (qmc_sampler)
    cstit       i.i.d. atoms drawn from it      the points themselves
    adaptive    density grid                    blurred histogram
"""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np

from . import config

CFG = config.TARGETS


# ── Target ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class Target:
    """
    kind    "image" (then `path` is set) or "points" (then `points` is set, in the unit cube).
    center, scale
            unit = (x - center) * scale + 1/2 maps the user's frame to the unit cube
            (the identity for images and for point sets already in [0, 1]^D).
    """
    kind: str
    path: str | None = None
    points: np.ndarray | None = None
    center: np.ndarray | float = 0.5
    scale: float = 1.0

    @property
    def D(self) -> int:
        return 2 if self.kind == "image" else self.points.shape[1]

    @property
    def moved(self) -> bool:
        """True if the point set had to be moved into the unit cube."""
        return self.scale != 1.0 or bool(np.any(np.asarray(self.center) != 0.5))

    def to_unit(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.center) * self.scale + 0.5 if self.moved else x

    def from_unit(self, u) -> np.ndarray:
        if not self.moved:
            return u
        u = np.asarray(u)
        return ((u - 0.5) / self.scale + self.center).astype(u.dtype, copy=False)

    def __repr__(self) -> str:
        if self.kind == "image":
            return f"Target(image {self.path!r})"
        moved = f", rescaled x{self.scale:.3g}" if self.moved else ""
        return f"Target({len(self.points):,} points, D={self.D}{moved})"


def as_target(target, D: int | None = None) -> Target:
    """
    target: an image path, a point set (K, D) or a `Target` (returned as is).
    D: if given, the dimension the target must have (images are 2D).
    """
    if isinstance(target, Target):
        t = target
    elif isinstance(target, (str, PathLike)):
        if not Path(target).is_file():
            raise FileNotFoundError(f"target image not found: {str(target)!r}")
        t = Target("image", path=str(target))
    else:
        t = _points_target(target)

    if D is not None and t.D != D:
        what = "an image target is 2D" if t.kind == "image" else f"the target points are {t.D}D"
        raise ValueError(f"{what}, but D={D} was requested")
    return t


def _points_target(points) -> Target:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 2:
        raise ValueError(f"target points must have shape (K, D) with K >= 2, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("target points contain NaN or inf")
    if np.ptp(points, axis=0).max() == 0:
        raise ValueError("target points are all identical")

    lo, hi = points.min(0), points.max(0)
    tol = 1e-5
    if lo.min() >= -tol and hi.max() <= 1 + tol:                 # already in the unit cube
        return Target("points", points=points % 1.0)

    extent = float((hi - lo).max())
    center = (lo + hi) / 2
    scale = (1 - 2 * CFG.points_margin) / extent
    return Target("points", points=(points - center) * scale + 0.5, center=center, scale=scale)


# ── Images ────────────────────────────────────────────────────────────────────
# Read as densities indexed [x, y] with y pointing up, pixel (i, j) at (i, j) / S.

def load_image(path: str, shape: tuple[int, int] | None = None, invert: bool = True) -> np.ndarray:
    """Any image resampled to a density grid of `shape` summing to 1. invert: dark = dense."""
    from PIL import Image

    shape = CFG.image_shape if shape is None else shape
    img = Image.open(path).convert("L").resize(shape[::-1], Image.LANCZOS)
    rho = np.asarray(img, dtype=np.float64) / 255.0
    rho = rho.T[:, ::-1]                                          # [row, col] -> [x, y]
    if invert:
        rho = 1.0 - rho
    return rho / rho.sum()


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic gaussian blur of a 2D grid, sigma in pixels."""
    fx = np.fft.fftfreq(img.shape[0])[:, None]
    fy = np.fft.fftfreq(img.shape[1])[None, :]
    g = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * g))


def sample_image(path: str, n: int, invert: bool = True, rng=None) -> np.ndarray:
    """n i.i.d. points (n, 2) drawn from the image (slightly blurred, then jittered)."""
    rng = np.random.default_rng() if rng is None else rng
    shape = CFG.image_shape
    density = blur(load_image(path, shape, invert), CFG.image_blur_sigma).clip(min=0).ravel()
    axis = np.linspace(0.0, 1.0, shape[0], endpoint=False)
    pixels = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    idx = rng.choice(len(density), size=n, replace=True, p=density / density.sum())
    return (pixels[idx] + CFG.image_jitter * rng.standard_normal((n, 2))) % 1.0


# ── Point sets ────────────────────────────────────────────────────────────────

def points_to_density(points: np.ndarray, shape: tuple[int, int], sigma: float | None = None) -> np.ndarray:
    """
    Pseudo image of 2D points in the unit square: histogram on a `shape` grid indexed [x, y],
    periodic gaussian blur, normalised to sum 1.
    sigma: blur in pixels. By default about one mean spacing of the points: enough to hide the
    points themselves, not their density variations.
    """
    if sigma is None:
        spacing = shape[0] / np.sqrt(len(points))
        sigma = max(CFG.points_blur_min, CFG.points_blur_spacing * spacing)
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=shape, range=[[0, 1], [0, 1]])
    rho = blur(hist, sigma).clip(min=0)
    return rho / rho.sum()


# ── The target as each solver family reads it ─────────────────────────────────

def for_cstit(target: Target | None, N: int, n_cells: int, n_atoms: int | None = None, rng=None):
    """
    Atoms (K, D) in the unit cube, to be split into `n_cells` equal cells; None if uniform.
    image: n_atoms i.i.d. atoms (default cstit_oversample * N, rounded up to a multiple of
    n_cells). point set: the points, at least N of them and a multiple of n_cells.
    """
    if target is None:
        return None
    if target.kind == "image":
        K = CFG.cstit_oversample * N if n_atoms is None else n_atoms
        K += (-K) % n_cells
        return sample_image(target.path, K, rng=rng)

    K = len(target.points)
    if K < N:
        raise ValueError(f"need at least N = {N:,} target points (comfortably more), got {K:,}")
    if K % n_cells:
        raise ValueError(f"the number of target points must be a multiple of the number of cells "
                         f"({n_cells}), got {K:,}: keep the first {K - K % n_cells:,}")
    return target.points


def for_adaptive(target: Target, shape: tuple[int, int], invert: bool = True) -> np.ndarray:
    """Density grid of `shape`, indexed [x, y], summing to 1 (2D targets only)."""
    if target.kind == "image":
        return load_image(target.path, shape, invert)
    return points_to_density(target.points, shape)
