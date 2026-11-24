"""blue.sample_points: every method in every tested dimension, warm starts, argument checks."""
import numpy as np
import pytest

import blue_sampler as blue

from conftest import DIMS, METHODS, N, WARMSTARTS, assert_points


CSTIT_OUTSIDE = pytest.mark.xfail(
    reason="cstit: moment matching can push points slightly outside [0, 1)", strict=False)


@pytest.mark.parametrize("D", DIMS)
@pytest.mark.parametrize("method", [pytest.param(m, marks=CSTIT_OUTSIDE) if m == "cstit" else m
                                    for m in METHODS])
def test_methods(method, D):
    assert_points(blue.sample_points(N, D, method=method, verbose=0), N, D)


@pytest.mark.parametrize("D", DIMS)
def test_cstit_shape(D):
    """Shape and finiteness of cstit, whatever its range (see test_methods)."""
    assert_points(blue.sample_points(N, D, method="cstit", verbose=0), N, D, unit=False)


def test_default_is_rgbn_in_2d():
    assert_points(blue.sample_points(N, verbose=0), N, 2)


def test_rgbn_recursive():
    """Above config.SAMPLE_POINTS.bruteforce_max_n, rgbn runs its coarse-to-fine recursion."""
    n = 2 * blue.config.SAMPLE_POINTS.bruteforce_max_n
    assert_points(blue.sample_points(n, 2, verbose=0), n, 2)


@pytest.mark.parametrize("D", DIMS)
@pytest.mark.parametrize("warmstart", WARMSTARTS)
def test_warmstart_names(warmstart, D):
    assert_points(blue.sample_points(N, D, warmstart=warmstart, verbose=0), N, D)


def test_warmstart_array_is_not_modified():
    x0 = np.random.rand(N, 2)
    copy = x0.copy()
    assert_points(blue.sample_points(N, 2, warmstart=x0, verbose=0), N, 2)
    np.testing.assert_array_equal(x0, copy)


def test_n_iter_scale_zero_returns_the_warmstart():
    x0 = np.random.rand(N, 2)
    np.testing.assert_array_equal(blue.sample_points(N, 2, warmstart=x0, n_iter_scale=0, verbose=0), x0)
    assert_points(blue.sample_points(N, 2, n_iter_scale=0, verbose=0), N, 2)      # Sobol


def test_verbose_prints(capsys):
    blue.sample_points(N, 2, verbose=1)
    assert capsys.readouterr().out


def test_silent(capsys):
    blue.sample_points(N, 2, verbose=0)
    assert capsys.readouterr().out == ""


# ── Argument checks ──────────────────────────────────────────────────────────

def test_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        blue.sample_points(N, 2, method="nope")


@pytest.mark.parametrize("method", ["latjit", "latcurl"])
def test_method_without_target(method):
    with pytest.raises(ValueError, match="does not take a target"):
        blue.sample_points(N, 2, method=method, target=np.random.rand(1000, 2))


def test_cstit_needs_a_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        blue.sample_points(100, 2, method="cstit", verbose=0)


def test_bad_warmstart_shape():
    with pytest.raises(ValueError, match="shape"):
        blue.sample_points(N, 2, warmstart=np.random.rand(N + 1, 2), verbose=0)
