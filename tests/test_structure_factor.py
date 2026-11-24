"""Structure factor, plots, and the quality of the samples (blue noise = low S(k) at low k)."""
import numpy as np
import pytest

import blue_sampler as blue

from conftest import DIMS, N


@pytest.mark.parametrize("D", DIMS)
def test_structure_factor(D):
    k, S = blue.structure_factor(np.random.rand(N, D))
    assert k.ndim == 2 and k.shape[1] == D and S.shape == (len(k),)
    assert np.isfinite(S).all() and (S >= 0).all()


def test_structure_factor_and_average():
    out = blue.structure_factor_and_average(np.random.rand(N, 2))
    assert len(out) == 4


def test_plots():
    x = np.random.rand(N, 2)
    blue.plot(x)
    blue.plot_structure_factor(x)


def low_k_mean(x, n_modes=50):
    """Mean of S(k) over the n_modes smallest |k|."""
    k, S = blue.structure_factor(x)
    return S[np.argsort(np.linalg.norm(k, axis=1))[:n_modes]].mean()


@pytest.mark.slow
@pytest.mark.parametrize("method", ["rgbn", "gaussian", "nufft", "nufft+", "cstit", "latjit", "latcurl"])
def test_blue_noise_beats_poisson(method):
    """Every sampler is far more uniform at large scale than i.i.d. points (S ~ 1).
    Measured ratios: latjit ~0.05, cstit ~0.01, the others < 1e-3."""
    x = blue.sample_points(1024, 2, method=method, verbose=0) % 1.0
    assert low_k_mean(x) < 0.1 * low_k_mean(np.random.rand(1024, 2))


@pytest.mark.slow
@pytest.mark.parametrize("method", ["rgbn", "nufft"])
def test_stealthy_quality(method):
    """The optimised samplers reach a low S(k) on the smallest wave vectors (2D)."""
    x = blue.sample_points(4096, 2, method=method, verbose=0)
    assert low_k_mean(x, n_modes=20) < 1e-4          # measured: rgbn ~1e-6, nufft ~1e-10
