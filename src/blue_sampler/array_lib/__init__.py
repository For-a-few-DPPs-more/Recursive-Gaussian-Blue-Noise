"""
Backend-agnostic array primitives.

backend   numpy/cupy + finufft/cufinufft selection, KeOps detection
ops       elementwise math on numpy, cupy, jax and KeOps LazyTensors
torus     periodic geometry on the unit torus [0, 1)^D
"""
from .backend import Backend, get_backend, keops_available, keops_error, gpu_array_module
from .ops import Ops, NUMPY, JAX, KEOPS
from .torus import wrap, min_image

__all__ = [
    "Backend", "get_backend", "keops_available", "keops_error", "gpu_array_module",
    "Ops", "NUMPY", "JAX", "KEOPS", "wrap", "min_image",
]
