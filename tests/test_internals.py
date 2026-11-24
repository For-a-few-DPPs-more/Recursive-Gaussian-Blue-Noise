"""Internal building blocks: torus, moment matching, config, progress, backend."""
import numpy as np
import pytest

import blue_sampler as blue
from blue_sampler.config import by_dim
from blue_sampler.moments import fit_points, required_n
from blue_sampler.progress import Progress, fmt_duration
from blue_sampler.torus import ball, half_ball, integer_grid, wrap


# ── Torus ────────────────────────────────────────────────────────────────────

def test_wrap():
    x = np.array([[-0.25, 1.5], [0.0, 0.999]])
    np.testing.assert_allclose(wrap(x), [[0.75, 0.5], [0.0, 0.999]])


@pytest.mark.parametrize("D", [1, 2, 3, 4])
@pytest.mark.parametrize("r2", [0, 1, 2, 5, 10])
def test_ball_matches_brute_force(r2, D):
    R = int(np.sqrt(r2))
    grid = integer_grid(np.arange(-R, R + 1), D)
    expected = grid[(grid ** 2).sum(1) <= r2]
    got = ball(r2, D)
    assert got.dtype == np.int32
    assert {tuple(k) for k in got} == {tuple(k) for k in expected}
    assert len(got) == len(expected)


@pytest.mark.parametrize("D", [1, 2, 3, 4])
@pytest.mark.parametrize("r2", [1, 2, 5, 10])
def test_half_ball_is_one_per_pair(r2, D):
    h = half_ball(r2, D)
    full = {tuple(k) for k in ball(r2, D)} - {(0,) * D}
    halves = {tuple(k) for k in h}
    assert len(h) == len(full) // 2 == len(halves)
    assert halves | {tuple(-np.array(k)) for k in halves} == full
    first = h[np.arange(len(h)), np.argmax(h != 0, axis=1)]
    assert (first > 0).all()


# ── Moment matching ──────────────────────────────────────────────────────────

def test_required_n():
    assert required_n(2, 2) >= 1
    assert required_n(3, 2) == 3


def test_fit_points_matches_centroids():
    clusters = np.random.rand(8, 64, 2)
    pts = fit_points(clusters, "clusters", p=3, n_iter=50, verbose=0)
    assert pts.shape[0] == 8 and pts.shape[-1] == 2
    np.testing.assert_allclose(pts.mean(axis=1), clusters.mean(axis=1), atol=1e-3)


# ── Config, progress, backend ────────────────────────────────────────────────

def test_by_dim_clamps():
    table = {2: "a", 3: "b", 5: "c"}
    assert by_dim(table, 1) == "a" and by_dim(table, 3) == "b" and by_dim(table, 9) == "c"


def test_config_is_editable():
    old = blue.config.RGBN.presets[2].lr_spatial
    try:
        blue.config.RGBN.presets[2].lr_spatial = 0.05
        assert blue.config.RGBN.presets[2].lr_spatial == 0.05
    finally:
        blue.config.RGBN.presets[2].lr_spatial = old


def test_fmt_duration():
    assert fmt_duration(5) == "5s" and fmt_duration(65) == "1min05"


def test_progress(capsys):
    bar = Progress("[test]", total=3)
    for _ in range(3):
        bar.tick("x")
    bar.done()
    assert "[test] done in" in capsys.readouterr().out


def test_check_gpu_runs(capsys):
    blue.check_gpu()
    assert "GPU" in capsys.readouterr().out


def test_version():
    assert blue.__version__.count(".") == 2
