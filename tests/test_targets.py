"""Non-uniform targets: point sets (any frame) and images."""
import numpy as np
import pytest
from PIL import Image

import blue_sampler as blue
from blue_sampler.targets import Target, as_target

from conftest import N, assert_points


@pytest.fixture
def image_path(tmp_path):
    """A 64 x 64 image: black left half (dense), white right half (empty)."""
    a = np.full((64, 64), 255, dtype=np.uint8)
    a[:, :32] = 0
    path = tmp_path / "half.png"
    Image.fromarray(a).save(path)
    return str(path)


# ── as_target ────────────────────────────────────────────────────────────────

def test_points_in_unit_cube_are_kept():
    pts = np.random.rand(500, 2) * 0.5 + 0.25
    t = as_target(pts)
    assert t.kind == "points" and not t.moved and t.D == 2


def test_points_outside_are_moved_and_round_trip():
    pts = np.random.rand(500, 3) * 10 + 5
    t = as_target(pts)
    assert t.moved and t.D == 3
    assert (t.points >= 0).all() and (t.points < 1).all()
    np.testing.assert_allclose(t.from_unit(t.to_unit(pts)), pts, atol=1e-9)


def test_target_is_returned_as_is():
    t = as_target(np.random.rand(500, 2))
    assert as_target(t) is t and isinstance(t, Target)


def test_wrong_dimension():
    with pytest.raises(ValueError):
        as_target(np.random.rand(500, 3), D=2)


def test_image_target(image_path):
    t = as_target(image_path)
    assert t.kind == "image" and t.D == 2


# ── sample_points(target=...) ────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["rgbn", "gaussian", "nufft"])
def test_point_set_target_2d(method):
    x = blue.sample_points(N, 2, method=method, target=np.random.rand(4096, 2), verbose=0)
    assert_points(x, N, 2)


def test_cstit_point_set_target():
    x = blue.sample_points(N, 2, method="cstit", target=np.random.rand(N * 32, 2), verbose=0)
    assert_points(x, N, 2, unit=False)


@pytest.mark.parametrize("D", [3, 4])
def test_point_set_target_nufft_high_d(D):
    x = blue.sample_points(N, D, method="nufft", target=np.random.rand(4096, D) * 0.5, verbose=0)
    assert_points(x, N, D)


@pytest.mark.parametrize("method", ["rgbn", "nufft"])
def test_samples_come_out_in_the_target_frame(method):
    """A point set in [10, 20]^2 gives samples in (roughly) [10, 20]^2."""
    pts = np.random.rand(4096, 2) * 10 + 10
    x = blue.sample_points(N, 2, method=method, target=pts, verbose=0)
    assert_points(x, N, 2, unit=False)
    assert x.min() > 9 and x.max() < 21


@pytest.mark.parametrize("method", ["rgbn", "nufft"])
def test_image_target_follows_the_density(image_path, method):
    """Almost every point lands in the dark (left) half."""
    x = blue.sample_points(N, 2, method=method, target=image_path, verbose=0)
    assert_points(x, N, 2)
    assert np.mean(x[:, 0] < 0.5) > 0.9


def test_image_to_points(image_path):
    assert_points(blue.image_to_points(image_path, N=N), N, 2)


def test_image_to_points_fast(image_path):
    assert_points(blue.image_to_points(image_path, N=N, fast=True), N, 2)
