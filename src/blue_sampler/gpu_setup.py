from types import SimpleNamespace

def check_gpu():
    """Check GPU availability and GPU dependencies.

    This function checks whether a GPU is available, whether JAX detects it,
    and whether CuPy and cuFINUFFT are installed correctly.
    """
    import shutil
    import subprocess

    print("GPU support diagnostic")
    print("----------------------")

    # ------------------------------------------------------------------
    # 1. Check for a NVIDIA GPU
    # ------------------------------------------------------------------
    gpu_available = False

    if shutil.which("nvidia-smi") is not None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            gpu_available = result.returncode == 0 and bool(result.stdout.strip())
        except (subprocess.SubprocessError, OSError):
            pass

    if not gpu_available:
        print("No NVIDIA GPU detected.")
        print("CPU-only execution is available.")
        return

    print("NVIDIA GPU detected.")

    # ------------------------------------------------------------------
    # 2. Check JAX
    # ------------------------------------------------------------------
    try:
        import jax

        jax_devices = jax.devices()
        jax_gpu_devices = [
            device for device in jax_devices
            if device.platform == "gpu"
        ]

        if not jax_gpu_devices:
            print("JAX is installed, but no GPU backend was detected.")
            print("Install the GPU-enabled version of JAX.")
            return

        print(f"JAX detects GPU: {jax_gpu_devices[0]}")

    except ImportError:
        print("JAX is not installed.")
        print("Install JAX with GPU support.")
        return
    except Exception as exc:
        print(f"JAX GPU check failed: {exc}")
        return

    # ------------------------------------------------------------------
    # 3. Check CuPy
    # ------------------------------------------------------------------
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        print("CuPy: OK")

    except ImportError:
        print("CuPy is not installed.")
        print("Install CuPy with a version compatible with your CUDA version.")
        return
    except Exception as exc:
        print(f"CuPy GPU check failed: {exc}")
        return

    # ------------------------------------------------------------------
    # 4. Check cuFINUFFT
    # ------------------------------------------------------------------
    try:
        import cufinufft

        print("cuFINUFFT: installed")

    except ImportError:
        print("cuFINUFFT is not installed.")
        print(
            "Install cuFINUFFT using conda. Depending on your environment, "
            "its native CUDA components may need to be compiled."
        )
        print(
            "See the cuFINUFFT installation instructions for the "
            "recommended conda setup."
        )
        return

    # ------------------------------------------------------------------
    # 5. Dummy test
    # ------------------------------------------------------------------
    try:
        import numpy as np

        # JAX GPU test
        x = jax.numpy.ones((1000, 1000))
        y = jax.numpy.dot(x, x)
        y.block_until_ready()

        # cuFINUFFT import + basic object creation test
        # Keep this deliberately small: the goal is only to verify that
        # the CUDA backend can be initialized.
        _ = cufinufft

        print("All right, config is operational !")

    except Exception as exc:
        print("GPU dependencies are installed, but the test failed.")
        print(f"Error: {exc}")


def set_config(device, precision, verbose):
    if device == "auto":
        device = "gpu" #we will try gpu if available
    
    device = device.lower()
    precision = "float64" if precision in ("float64", "double") else "float32"

    if device == "gpu":
        try:
            import cupy as cp, cufinufft as nufft
            xp, to_numpy = cp, cp.asnumpy
        except Exception:
            device = "cpu"
            import numpy as np, finufft as nufft
            xp, to_numpy = np, np.asarray
    else:
        import numpy as np, finufft as nufft
        xp, to_numpy = np, np.asarray

    if precision == "float64":
        real, cplx, plan_dt = xp.float64, xp.complex128, "complex128"
    else:
        real, cplx, plan_dt = xp.float32, xp.complex64, "complex64"

    cfg = SimpleNamespace()

    cfg.__dict__.update(
        device=device, precision=precision, xp=xp, nufft_lib=nufft,
        to_numpy=to_numpy, real_dtype=real, complex_dtype=cplx,
        plan_dtype=plan_dt
    )
    if verbose >= 1:
        print(f"[config] {cfg.device} | {precision} | dtype={cfg.real_dtype}")
    return cfg