"""Compute backend selection: numpy + finufft on CPU, cupy + cufinufft on GPU."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

import numpy as np


@dataclass(frozen=True)
class Backend:
    device: str             # "cpu" | "gpu"
    precision: str          # "float32" | "float64"
    xp: ModuleType          # numpy or cupy
    nufft: ModuleType       # finufft or cufinufft

    @property
    def real(self):
        return self.xp.float64 if self.precision == "float64" else self.xp.float32

    @property
    def complex(self):
        return self.xp.complex128 if self.precision == "float64" else self.xp.complex64

    def to_numpy(self, a) -> np.ndarray:
        return self.xp.asnumpy(a) if self.device == "gpu" else np.asarray(a)

    def __str__(self) -> str:
        return f"{self.device} | {self.precision}"


def get_backend(device: str = "auto", precision: str = "float32") -> Backend:
    """'auto' and 'gpu' try cupy + cufinufft and fall back to numpy + finufft."""
    precision = "float64" if precision in ("float64", "double") else "float32"
    if device.lower() in ("auto", "gpu", "cuda"):
        try:
            import cupy
            import cufinufft
            return Backend("gpu", precision, cupy, cufinufft)
        except Exception:
            pass
    import finufft
    return Backend("cpu", precision, np, finufft)


def gpu_array_module():
    """(xp, to_xp, to_numpy): cupy if available, numpy otherwise."""
    try:
        import cupy as cp
        return cp, cp.asarray, cp.asnumpy
    except Exception:
        return np, np.asarray, np.asarray


@lru_cache(maxsize=None)
def _probe_keops():
    """Compile and run a tiny KeOps reduction once; returns the error, or None if it works."""
    try:
        from pykeops.numpy import LazyTensor
        x = np.random.rand(10, 3).astype(np.float32)
        ((LazyTensor(x[:, None, :]) - LazyTensor(x[None, :, :])) ** 2).sum(-1).sum_reduction(axis=1)
        return None
    except Exception as exc:  # noqa: BLE001 (any failure means "unusable")
        return exc


def keops_available() -> bool:
    """True if pykeops is installed *and* able to compile. Checked lazily, once."""
    return _probe_keops() is None


def keops_error():
    return _probe_keops()
