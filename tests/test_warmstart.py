"""Warm starts and lattices."""
import numpy as np
import pytest

from blue_sampler.lattices import curled_lattice, jittered_lattice, korobov_lattice, sobol
from blue_sampler.warmstart import warmstart_points

from conftest import DIMS, N, assert_points


@pytest.mark.parametrize("D", DIMS)
@pytest.mark.parametrize("method", [None, "Sobol", "latjit", "latcurl"])
def test_warmstart_points(method, D):
    assert_points(warmstart_points(N, D, method), N, D)


def test_warmstart_array_is_copied():
    x0 = np.random.rand(N, 2)
    x = warmstart_points(N, 2, x0)
    np.testing.assert_array_equal(x, x0)
    assert x is not x0


def test_warmstart_bad_array():
    with pytest.raises(ValueError, match="shape"):
        warmstart_points(N, 2, np.random.rand(N, 3))


def test_warmstart_unknown():
    with pytest.raises(ValueError, match="unknown warmstart"):
        warmstart_points(N, 2, "nope")


@pytest.mark.parametrize("D", DIMS)
@pytest.mark.parametrize("lattice", [korobov_lattice, jittered_lattice, curled_lattice, sobol])
def test_lattices(lattice, D):
    assert_points(lattice(N, D), N, D)


def test_sobol_seed():
    np.testing.assert_array_equal(sobol(N, 2, seed=1), sobol(N, 2, seed=1))
