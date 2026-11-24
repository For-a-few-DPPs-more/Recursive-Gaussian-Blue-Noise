"""
Shared settings of the test suite.

    pytest                      # everything except the slow tests
    pytest -m slow              # only the slow ones (quality of the samples)
    pytest -m "slow or not slow"  # everything
"""
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")                      # no window during the tests

N = 128                                    # small enough for every method, a power of 2 for cstit
DIMS = (2, 3, 4)
METHODS = ("rgbn", "gaussian", "nufft", "nufft+", "cstit", "latjit", "latcurl")
WARMSTARTS = ("Sobol", "latjit", "latcurl")


@pytest.fixture(autouse=True)
def _quiet_and_seeded():
    """Every test starts from the same numpy seed, without third-party warnings."""
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def assert_points(x, n, D, unit=True):
    """x is an (n, D) array of finite floats, in [0, 1)^D if unit."""
    x = np.asarray(x)
    assert x.shape == (n, D)
    assert np.isfinite(x).all()
    if unit:
        assert (x >= 0).all() and (x < 1).all(), f"points outside [0, 1): min {x.min()}, max {x.max()}"
