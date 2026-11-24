"""
Optional acceleration: cupy arrays, KeOps reductions (JAX picks its own device).
Everything falls back to numpy on the CPU. `check_gpu` reports what is usable.
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache

import numpy as np


def array_module():
    """(xp, to_xp, to_numpy) for cupy if it imports, else numpy."""
    try:
        import cupy as cp
        return cp, cp.asarray, cp.asnumpy
    except Exception:
        return np, np.asarray, np.asarray


@lru_cache(maxsize=None)
def keops_error():
    """None if pykeops compiles and runs a tiny reduction (tried once), else the exception."""
    try:
        from pykeops.numpy import LazyTensor
        x = np.random.rand(10, 3).astype(np.float32)
        ((LazyTensor(x[:, None, :]) - LazyTensor(x[None, :, :])) ** 2).sum(-1).sum_reduction(axis=1)
        return None
    except Exception as exc:  # noqa: BLE001 (any failure means "unusable")
        return exc


def keops_available() -> bool:
    return keops_error() is None


# ── Diagnostic ────────────────────────────────────────────────────────────────

def check_gpu() -> None:
    """Print which parts of the optional GPU stack work: JAX, CuPy, KeOps."""
    print("GPU support diagnostic\n----------------------")
    if not _nvidia_gpu():
        print("No NVIDIA GPU detected: CPU-only execution.")
        return
    print("NVIDIA GPU detected.")

    def jax_check():
        import jax
        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        if not gpus:
            raise RuntimeError("JAX is installed without GPU backend")
        x = jax.numpy.ones((1000, 1000))
        jax.numpy.dot(x, x).block_until_ready()
        return f"GPU {gpus[0]}"

    def cupy_check():
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return "OK"

    def keops_check():
        if keops_error() is not None:
            raise RuntimeError(keops_error())
        return "OK"

    checks = [
        ("JAX", jax_check, "Install JAX with GPU support."),
        ("CuPy", cupy_check, "Install a CuPy version matching your CUDA version."),
        ("KeOps", keops_check, "Optional: pykeops only speeds up sampling, it is not required."),
    ]
    ok = True
    for name, check, hint in checks:
        try:
            print(f"{name}: {check()}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: FAILED ({exc})\n  {hint}")
            ok = ok and name == "KeOps"
    if ok:
        print("All right, config is operational!")


def _nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and bool(out.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False
