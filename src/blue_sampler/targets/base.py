"""
Classification, validation and normalisation of a target: image path or point set.

Point sets outside [0, 1]^D are mapped into the unit cube by a uniform scaling (same factor on
every axis, so isotropy is preserved), centred, with a margin `config.TARGETS.points_margin`
(the cube is a torus). Solvers work in the unit cube and map their output back with
`Target.from_unit`, so the samples live in the frame of the target.
"""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np

from .. import config
from .points import check_points

CFG = config.TARGETS


@dataclass(frozen=True, eq=False)
class Target:
    """
    A validated target. kind: "image" (path, 2D) or "points" ((K, D) array, in the unit cube).

    unit = (x - center) * scale + 1/2: frame of the user -> unit cube (identity for images and
    for point sets already in [0, 1]^D).
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
    def normalised(self) -> bool:
        """True if the target was moved into the unit cube (to_unit is not the identity)."""
        return self.scale != 1.0 or bool(np.any(np.asarray(self.center) != 0.5))

    def to_unit(self, x) -> np.ndarray:
        """Points of the user's frame -> unit cube."""
        x = np.asarray(x, dtype=np.float64)
        return x if not self.normalised else (x - self.center) * self.scale + 0.5

    def from_unit(self, u) -> np.ndarray:
        """Points of the unit cube (solver output) -> user's frame."""
        if not self.normalised:
            return u
        u = np.asarray(u)
        return ((u - 0.5) / self.scale + self.center).astype(u.dtype, copy=False)

    def __repr__(self) -> str:
        if self.kind == "image":
            return f"Target(image {self.path!r})"
        frame = f", rescaled x{self.scale:.3g}" if self.normalised else ""
        return f"Target({len(self.points):,} points, D={self.D}{frame})"


def _normalise(points: np.ndarray, tol: float = 1e-5):
    """(unit points, center, scale): identity if the points are in [0, 1]^D, else uniform scaling."""
    lo, hi = points.min(0), points.max(0)
    if lo.min() >= -tol and hi.max() <= 1 + tol:
        return points % 1.0, 0.5, 1.0
    extent = float((hi - lo).max())
    center = (lo + hi) / 2
    scale = (1 - 2 * CFG.points_margin) / extent
    return (points - center) * scale + 0.5, center, scale


def as_target(target, D: int | None = None) -> Target:
    """
    Target from an image path, a point set (K, D) (any bounding box, see module docstring) or a
    `Target` (returned as is).

    D: dimension of the sampling, checked against the target (images are 2D).
    """
    if isinstance(target, Target):
        t = target
    elif isinstance(target, (str, PathLike)):
        path = str(target)
        if not Path(path).is_file():
            raise FileNotFoundError(f"target image not found: {path!r}")
        t = Target("image", path=path)
    else:
        unit, center, scale = _normalise(check_points(target))
        t = Target("points", points=unit, center=center, scale=scale)
    if D is not None and t.D != D:
        what = "an image target is 2D" if t.kind == "image" else f"the target points are {t.D}D"
        raise ValueError(f"{what}, but D={D} was requested")
    return t
