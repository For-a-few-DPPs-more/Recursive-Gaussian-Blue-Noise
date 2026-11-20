"""
Elementwise operations with one spelling for numpy, cupy, jax and KeOps.

Kernels written against an `Ops` instance run unchanged on dense arrays and on
KeOps LazyTensors (which use methods, e.g. ``x.exp()``, instead of functions).
``sum`` reduces the coordinate axis and keeps it (broadcastable against x);
``reduce`` sums over the "other point" axis 1, producing a dense array.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Ops:
    exp: Callable
    sin: Callable
    round: Callable
    sum: Callable       # sum over the last (coordinate) axis, kept
    reduce: Callable    # sum over axis 1 (interaction partners)


NUMPY = Ops(
    exp=np.exp, sin=np.sin, round=np.round,
    sum=lambda x: np.sum(x, axis=-1, keepdims=True),
    reduce=lambda x: np.sum(x, axis=1),
)


def _jax_ops() -> Ops:
    import jax.numpy as jnp
    return Ops(
        exp=jnp.exp, sin=jnp.sin, round=jnp.round,
        sum=lambda x: jnp.sum(x, axis=-1, keepdims=True),
        reduce=lambda x: jnp.sum(x, axis=1),
    )


JAX = _jax_ops()

KEOPS = Ops(
    exp=lambda x: x.exp(), sin=lambda x: x.sin(), round=lambda x: x.round(),
    sum=lambda x: x.sum(dim=2),
    reduce=lambda x: x.sum_reduction(dim=1),
)
