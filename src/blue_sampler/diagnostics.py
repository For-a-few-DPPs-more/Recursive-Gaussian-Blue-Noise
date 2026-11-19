"""Check the optional GPU stack: NVIDIA driver, JAX, CuPy, cuFINUFFT, KeOps."""
from __future__ import annotations

import shutil
import subprocess


def _nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and bool(out.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


def _check_jax():
    import jax
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        raise RuntimeError("JAX is installed without GPU backend: install the GPU-enabled JAX")
    x = jax.numpy.ones((1000, 1000))
    jax.numpy.dot(x, x).block_until_ready()
    return f"GPU {gpus[0]}"


def _check_cupy():
    import cupy
    cupy.cuda.runtime.getDeviceCount()
    return "OK"


def _check_cufinufft():
    import cufinufft  # noqa: F401
    return "installed"


def _check_keops():
    from .array_lib import keops_error
    if keops_error() is not None:
        raise RuntimeError(keops_error())
    return "OK"


CHECKS = [
    ("JAX", _check_jax, "Install JAX with GPU support."),
    ("CuPy", _check_cupy, "Install a CuPy version matching your CUDA version."),
    ("cuFINUFFT", _check_cufinufft, "Install cuFINUFFT with conda (its CUDA components may need compiling)."),
    ("KeOps", _check_keops, "Optional: pykeops only speeds up sampling, it is not required."),
]


def check_gpu() -> None:
    """Print a diagnostic of the GPU dependencies."""
    print("GPU support diagnostic\n----------------------")
    if not _nvidia_gpu():
        print("No NVIDIA GPU detected: CPU-only execution.")
        return
    print("NVIDIA GPU detected.")
    ok = True
    for name, check, hint in CHECKS:
        try:
            print(f"{name}: {check()}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: FAILED ({exc})\n  {hint}")
            if name != "KeOps":
                ok = False
    if ok:
        print("All right, config is operational!")
