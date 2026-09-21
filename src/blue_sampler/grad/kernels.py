"""
Interaction kernels for energy gradient
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from ..math import torus_delta, clean_grad as clean



# ──────────────────────────────────────────────────────────────────────────────
# Kernel functions  (JAX)
# ──────────────────────────────────────────────────────────────────────────────

def gauss_fun(x, y, sigma2):
    delta = torus_delta(y - x)
    dist2 = jnp.sum(delta ** 2, axis=-1)
    return clean(jnp.exp(-dist2 / sigma2))

def gauss_sin_fun(x, y, sigma2):
    delta = jnp.sin(2*jnp.pi*(y-x))/jnp.pi
    dist2 = jnp.sum(delta ** 2, axis=-1)
    return clean(jnp.exp(-dist2 / sigma2))
    
def gauss_kernel(
    x: jnp.ndarray,
    y: jnp.ndarray,
    sigma2: float,
) -> jnp.ndarray:
    """
    Isotropic Gaussian repulsion kernel on the torus.

    Returns the *gradient* contribution  (y − x) * exp(−‖y−x‖² / σ²).
    """
    delta = torus_delta(y - x)
    dist2 = jnp.sum(delta ** 2, axis=-1, keepdims=True)
    return clean(delta * jnp.exp(-dist2 / sigma2))


def gauss_sin_kernel(
    x: jnp.ndarray,
    y: jnp.ndarray,
    a: float,
    b: float,
    c: float,
) -> jnp.ndarray:
    """
    Trigonometric Gaussian kernel — more stable in contexts were
    sigma2 is not << 1 e.g. high dimension or low number of points
    -> using discontinuous torus_delta would become problematic.

    Parameters
    ----------
    a, b, c : pre-computed scale factors (derived from sigma²).
    """
    delta    = a * (y - x)
    cos_term = b * (1.0 - jnp.cos(delta))
    sin_term = c * jnp.sin(delta)
    dist2 = jnp.sum(cos_term, axis=-1, keepdims=True)
    return clean(sin_term * jnp.exp(-dist2))

   
def spectral_kernel(x, k, k_):
    """
    spectral kernel directly target spectral energy. Only usable
    for small subsets of preselected wavevectors
    """
    phase   = jnp.sum(k * x, axis=-1, keepdims=True)
    ek      = clean(jnp.exp(phase))
    Sk      = jnp.sum(ek, axis=0, keepdims=True)
    return jnp.real(Sk * k_ * jnp.conjugate(ek))