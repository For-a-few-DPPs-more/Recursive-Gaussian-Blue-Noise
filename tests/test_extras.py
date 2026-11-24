"""blue_sampler.extras: partitions, pinwheel, tiling and plots."""
import numpy as np
import pytest

import blue_sampler as blue
import blue_sampler.extras as bluex

from conftest import DIMS, N, assert_points


def test_sobol():
    assert_points(bluex.sobol(N, 3), N, 3)


@pytest.mark.parametrize("D", DIMS)
def test_tile(D):
    x = np.random.rand(N, D)
    assert_points(bluex.tile(x, repeat=2), N * 2 ** D, D)


# ── Clusters ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("D", DIMS)
@pytest.mark.parametrize("target", ["korobov", None])
def test_sample_clusters(target, D):
    c = bluex.sample_clusters(N, D, target=target, atoms_per_cluster=8)
    assert c.shape == (N, 8, D)
    assert (c >= 0).all() and (c < 1).all()


def test_sample_clusters_point_set_target():
    c = bluex.sample_clusters(N, 2, target=np.random.rand(N * 4, 2))
    assert c.shape == (N, 4, 2)


@pytest.mark.parametrize("D", DIMS)
def test_clusters_to_points(D):
    x = bluex.clusters_to_points(bluex.sample_clusters(N, D), verbose=0)
    assert x.shape[0] == N and x.shape[-1] == D and np.isfinite(x).all()


def test_clusters_need_a_power_of_two():
    with pytest.raises(ValueError):
        bluex.sample_clusters(100, 2)


# ── Tessellations ────────────────────────────────────────────────────────────

def test_sample_tessels_cover_the_unit_square():
    from blue_sampler.extras.tessels import polygon_area
    t = bluex.sample_tessels(N)
    assert t.shape == (N, 4, 2)
    areas = np.abs(polygon_area(t))
    np.testing.assert_allclose(areas.sum(), 1.0, rtol=1e-6)
    np.testing.assert_allclose(areas, 1.0 / N, rtol=1e-4)         # fair: equal areas


def test_tessels_to_points():
    x = bluex.tessels_to_points(bluex.sample_tessels(N), verbose=0)
    assert x.shape == (N, 3, 2)
    assert_points(x.reshape(-1, 2), 3 * N, 2)


# ── Pinwheel ─────────────────────────────────────────────────────────────────

def test_pinwheel_base():
    assert bluex.pinwheel_base().shape == (3, 2)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_pinwheel_transform(depth):
    T = 4 * 5 ** depth
    assert bluex.pinwheel_transform(bluex.pinwheel_base(), depth=depth).shape == (T, 3, 2)
    assert bluex.pinwheel_transform(np.array([0.5, 0.25]), depth=depth).shape == (T, 2)


def test_pinwheel_points():
    assert_points(bluex.pinwheel_points(N), N, 2)


# ── Plots (smoke tests) ──────────────────────────────────────────────────────

def test_plots():
    bluex.plot_tessels(bluex.sample_tessels(16))
    bluex.plot_clusters(bluex.sample_clusters(16, 2))
    bluex.plot_polygons(bluex.pinwheel_transform(bluex.pinwheel_base(), depth=1))
