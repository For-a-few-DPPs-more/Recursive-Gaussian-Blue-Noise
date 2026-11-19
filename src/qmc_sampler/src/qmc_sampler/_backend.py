"""
Backends for the Fourier sums used by the samplers and by the structure factor.

Two primitives, for points x (N, D) in [0, 1)^D and integer wave vectors k (M, D):

    nudft1(x, c, k, isign)       F_m = sum_j c_j     exp(isign * 2i pi k_m . x_j)     -> (M,) complex
    nudft2_real(x, a, k, isign)  g_j = Re sum_m a_mp exp(isign * 2i pi k_m . x_j)     -> (N, P) real

Available backends
------------------
"keops"   : exact direct sums (O(N M)), pykeops LazyTensor reductions (GPU if available).
"torch"   : exact direct sums (O(N M)), dense torch tensors by chunks (cuda if available).
"finufft" : non-uniform FFT (O(N log N + M)), D <= 3 only. Approximate, with relative
            tolerance FINUFFT_EPS[precision] (1e-4 in float32, 1e-8 in float64).

The backend is chosen per dimension:

    select_backend(dimensions={2, 3}, backend="finufft")

Default (dimensions never selected): "keops" if available, else "torch", else "finufft".
finufft is never the default when keops or torch is available, so that, unless asked
otherwise, all Fourier sums are exact up to machine precision.
"""
from __future__ import annotations

import importlib
from typing import Iterable, Optional, Union

import numpy as np

__all__ = [
    "BACKENDS",
    "FINUFFT_EPS",
    "FINUFFT_MAX_DIM",
    "available_backends",
    "get_backend",
    "select_backend",
    "nudft1",
    "nudft2_real",
]

BACKENDS = ("keops", "torch", "finufft")      # also the default priority order
FINUFFT_EPS = {"float32": 1e-4, "float64": 1e-8}
FINUFFT_MAX_DIM = 3
TORCH_CHUNK_ELEMS = 8_000_000                 # torch: (points x wave vectors) per chunk

_MODULES = {"keops": "pykeops.numpy", "torch": "torch", "finufft": "finufft"}
_PIP = {"keops": "pykeops", "torch": "torch", "finufft": "finufft"}
_cache: dict = {}                             # backend -> imported module, or the ImportError
_selected: dict = {}                          # dimension -> backend


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _load(backend: str):
    """Import (once) and return the module of a backend; raise a clean ImportError otherwise."""
    if backend not in _cache:
        try:
            _cache[backend] = importlib.import_module(_MODULES[backend])
        except Exception as err:              # ImportError, or a broken install
            _cache[backend] = err
    mod = _cache[backend]
    if isinstance(mod, Exception):
        raise ImportError(
            f"qmc_sampler: backend {backend!r} is not available "
            f"(import of {_MODULES[backend]!r} failed: {mod}). "
            f"Install it with `pip install {_PIP[backend]}`."
        ) from None
    return mod


def _is_available(backend: str) -> bool:
    try:
        _load(backend)
        return True
    except ImportError:
        return False


def _check_dim(D) -> int:
    if isinstance(D, bool) or not isinstance(D, (int, np.integer)) or D < 1:
        raise ValueError(f"qmc_sampler: dimensions must be positive integers, got {D!r}")
    return int(D)


def available_backends(D: Optional[int] = None) -> list:
    """Backends that can be imported (and, if D is given, that support dimension D)."""
    out = [b for b in BACKENDS if _is_available(b)]
    if D is not None and _check_dim(D) > FINUFFT_MAX_DIM:
        out = [b for b in out if b != "finufft"]
    return out


def select_backend(dimensions: Union[int, Iterable[int]], backend: Optional[str]) -> None:
    """
    Select the backend used for the given dimension(s) only.

    Parameters
    ----------
    dimensions : int or iterable of ints, e.g. 2 or {2, 3}.
    backend    : "keops" | "torch" | "finufft", or None to restore the default
                 for these dimensions.

    Raises
    ------
    ValueError  : unknown backend, invalid dimension, or finufft with a dimension > 3.
    ImportError : the backend cannot be imported.

    Nothing is changed if an error is raised.

    Example
    -------
    >>> import qmc_sampler
    >>> qmc_sampler.select_backend(dimensions={2, 3}, backend="finufft")
    """
    dims = [dimensions] if isinstance(dimensions, (int, np.integer)) else list(dimensions)
    if not dims:
        raise ValueError("qmc_sampler: `dimensions` is empty")
    dims = [_check_dim(D) for D in dims]

    if backend is None:
        for D in dims:
            _selected.pop(D, None)
        return
    if backend not in BACKENDS:
        raise ValueError(f"qmc_sampler: unknown backend {backend!r}, expected one of {BACKENDS}")
    if backend == "finufft":
        too_big = sorted(D for D in dims if D > FINUFFT_MAX_DIM)
        if too_big:
            raise ValueError(
                f"qmc_sampler: the finufft backend only supports dimensions <= "
                f"{FINUFFT_MAX_DIM}, got {too_big}"
            )
    _load(backend)
    for D in dims:
        _selected[D] = backend


def get_backend(D: int) -> str:
    """Backend used in dimension D (selected one, or default)."""
    D = _check_dim(D)
    if D in _selected:
        return _selected[D]
    for b in BACKENDS:
        if (b != "finufft" or D <= FINUFFT_MAX_DIM) and _is_available(b):
            return b
    raise ImportError(
        f"qmc_sampler: no backend available in dimension {D}. Install pykeops "
        f"(`pip install pykeops`) or torch (`pip install torch`)"
        + (", or finufft (`pip install finufft`)." if D <= FINUFFT_MAX_DIM else
           " (finufft only supports D <= 3).")
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _precision(x) -> str:
    return "float64" if np.asarray(x).dtype == np.float64 else "float32"


def _real(precision):
    return np.float64 if precision == "float64" else np.float32


def _complex(precision):
    return np.complex128 if precision == "float64" else np.complex64


# ---------------------------------------------------------------------------
# KeOps
# ---------------------------------------------------------------------------

def _keops_phase(x, k, real):
    LazyTensor = _load("keops").LazyTensor
    xi = LazyTensor(np.ascontiguousarray(x[:, None, :], dtype=real))
    kj = LazyTensor(np.ascontiguousarray(2.0 * np.pi * k[None, :, :], dtype=real))
    return LazyTensor, (xi * kj).sum(-1, keepdim=True)            # (N, M, 1)


def _keops_nudft1(x, c, k, isign, real):
    LazyTensor, phase = _keops_phase(x, k, real)
    cos, sin = phase.cos(), phase.sin()
    # F = sum (cr + i ci)(cos + i isign sin)
    #   = sum [cr cos - isign ci sin] + i [isign cr sin + ci cos]
    if c is None:
        return np.asarray(cos.sum(0)).ravel() + 1j * isign * np.asarray(sin.sum(0)).ravel()
    cr = LazyTensor(np.ascontiguousarray(np.real(c).reshape(-1, 1, 1), dtype=real))
    re = np.asarray((cr * cos).sum(0)).ravel().astype(np.float64)
    im = isign * np.asarray((cr * sin).sum(0)).ravel().astype(np.float64)
    if np.iscomplexobj(c) and np.any(np.imag(c)):
        ci = LazyTensor(np.ascontiguousarray(np.imag(c).reshape(-1, 1, 1), dtype=real))
        re -= isign * np.asarray((ci * sin).sum(0)).ravel()
        im += np.asarray((ci * cos).sum(0)).ravel()
    return re + 1j * im


def _keops_nudft2_real(x, a, k, isign, real):
    LazyTensor, phase = _keops_phase(x, k, real)
    ar = LazyTensor(np.ascontiguousarray(np.real(a)[None, :, :], dtype=real))   # (1, M, P)
    ai = LazyTensor(np.ascontiguousarray(np.imag(a)[None, :, :], dtype=real))
    # Re[(ar + i ai)(cos + i isign sin)] = ar cos - isign ai sin
    return np.asarray((ar * phase.cos() - isign * (ai * phase.sin())).sum(1))


# ---------------------------------------------------------------------------
# torch
# ---------------------------------------------------------------------------

def _torch_device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_nudft1(x, c, k, isign, real):
    torch = _load("torch")
    dev, dt = _torch_device(torch), getattr(torch, np.dtype(real).name)
    kt = torch.as_tensor(2.0 * np.pi * k, dtype=dt, device=dev)
    N, M = len(x), len(k)
    c = np.ones(N, dtype=real) if c is None else np.asarray(c)
    cr = torch.as_tensor(np.real(c), dtype=dt, device=dev)
    ci = torch.as_tensor(np.imag(c), dtype=dt, device=dev) if np.iscomplexobj(c) else None
    chunk = max(1, TORCH_CHUNK_ELEMS // max(M, 1))
    re = torch.zeros(M, dtype=dt, device=dev)
    im = torch.zeros(M, dtype=dt, device=dev)
    for s in range(0, N, chunk):
        phase = torch.as_tensor(x[s:s + chunk], dtype=dt, device=dev) @ kt.T     # (n, M)
        cos, sin = phase.cos(), phase.sin()
        re += cr[s:s + chunk] @ cos
        im += isign * (cr[s:s + chunk] @ sin)
        if ci is not None:                    # see _keops_nudft1
            re -= isign * (ci[s:s + chunk] @ sin)
            im += ci[s:s + chunk] @ cos
    return re.cpu().numpy().astype(np.float64) + 1j * im.cpu().numpy().astype(np.float64)


def _torch_nudft2_real(x, a, k, isign, real):
    torch = _load("torch")
    dev, dt = _torch_device(torch), getattr(torch, np.dtype(real).name)
    kt = torch.as_tensor(2.0 * np.pi * k, dtype=dt, device=dev)
    ar = torch.as_tensor(np.real(a), dtype=dt, device=dev)                      # (M, P)
    ai = torch.as_tensor(np.imag(a), dtype=dt, device=dev)
    N, M = len(x), len(k)
    chunk = max(1, TORCH_CHUNK_ELEMS // max(M, 1))
    out = []
    for s in range(0, N, chunk):
        phase = torch.as_tensor(x[s:s + chunk], dtype=dt, device=dev) @ kt.T
        out.append((phase.cos() @ ar - isign * (phase.sin() @ ai)).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# finufft
# ---------------------------------------------------------------------------

def _finufft_grid(k):
    """Index of each wave vector in the smallest centred (odd) mode grid containing them."""
    k = np.asarray(k, dtype=np.int64)
    half = np.abs(k).max(axis=0) if len(k) else np.zeros(k.shape[1], dtype=np.int64)
    shape = tuple(int(2 * h + 1) for h in half)
    # finufft (modeord=0): index i <-> mode i - n // 2 = i - h
    return shape, tuple((k + half).T)


def _finufft_fn(D, kind):
    return getattr(_load("finufft"), f"nufft{D}d{kind}")


def _finufft_nudft1(x, c, k, isign, precision):
    N, D = x.shape
    shape, idx = _finufft_grid(k)
    coords = [np.ascontiguousarray(2.0 * np.pi * x[:, d], dtype=_real(precision)) for d in range(D)]
    cc = np.ones(N) if c is None else np.asarray(c)
    cc = np.ascontiguousarray(cc, dtype=_complex(precision))
    f = _finufft_fn(D, 1)(*coords, cc, shape, eps=FINUFFT_EPS[precision], isign=isign)
    return f[idx].astype(np.complex128)


def _finufft_nudft2_real(x, a, k, isign, precision):
    N, D = x.shape
    P = a.shape[1]
    shape, idx = _finufft_grid(k)
    coords = [np.ascontiguousarray(2.0 * np.pi * x[:, d], dtype=_real(precision)) for d in range(D)]
    f = np.zeros((P,) + shape, dtype=_complex(precision))
    for p in range(P):
        np.add.at(f[p], idx, a[:, p])
    if P == 1:
        g = _finufft_fn(D, 2)(*coords, f[0], eps=FINUFFT_EPS[precision], isign=isign)[None]
    else:
        g = _finufft_fn(D, 2)(*coords, f, eps=FINUFFT_EPS[precision], isign=isign)
    return np.ascontiguousarray(g.real.T)                               # (N, P)


# ---------------------------------------------------------------------------
# Public primitives
# ---------------------------------------------------------------------------

def nudft1(x, c, k, isign: int = -1, backend: Optional[str] = None) -> np.ndarray:
    """F_m = sum_j c_j exp(isign * 2i pi k_m . x_j), shape (M,), complex128.

    x : (N, D) float32 or float64 (sets the precision), c : (N,) real/complex or None (ones),
    k : (M, D) integer wave vectors. backend defaults to get_backend(D).
    """
    x = np.asarray(x)
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float64)
    precision = _precision(x)
    backend = backend or get_backend(x.shape[1])
    k = np.asarray(k)
    if len(k) == 0:
        return np.zeros(0, dtype=np.complex128)
    if backend == "keops":
        return _keops_nudft1(x, c, k, isign, _real(precision))
    if backend == "torch":
        return _torch_nudft1(x, c, k, isign, _real(precision))
    return _finufft_nudft1(x, c, k, isign, precision)


def nudft2_real(x, a, k, isign: int = 1, backend: Optional[str] = None) -> np.ndarray:
    """g_jp = Re sum_m a_mp exp(isign * 2i pi k_m . x_j), shape (N, P), dtype of x.

    x : (N, D) float32 or float64 (sets the precision), a : (M, P) complex,
    k : (M, D) integer wave vectors. backend defaults to get_backend(D).
    """
    x = np.asarray(x)
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float64)
    precision = _precision(x)
    backend = backend or get_backend(x.shape[1])
    a = np.asarray(a)
    a = a.reshape(len(a), -1)
    k = np.asarray(k)
    if len(k) == 0:
        return np.zeros((len(x), a.shape[1]), dtype=x.dtype)
    if backend == "keops":
        g = _keops_nudft2_real(x, a, k, isign, _real(precision))
    elif backend == "torch":
        g = _torch_nudft2_real(x, a, k, isign, _real(precision))
    else:
        g = _finufft_nudft2_real(x, a, k, isign, precision)
    return np.asarray(g, dtype=x.dtype)
