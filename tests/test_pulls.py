"""Synthetic-path tests for the shared pull metrics (no competition data needed)."""

import numpy as np
import pytest

from bdb27.pulls import onset_index, rep_metrics, to_intrinsic

DT = 0.1


def l_path(lat=8.0, up=6.0, speed=5.0, still=5, theta=0.0, mirror=False):
    """Stand still, run `lat` yd laterally, then `up` yd upfield, at constant speed.

    Built in (lateral, forward) coords, then rotated by `theta` and optionally mirrored,
    to mimic Combine drills run in arbitrary orientations.
    """
    step = speed * DT
    u = [0.0] * still + list(np.arange(step, lat + 1e-9, step))
    v = [0.0] * len(u)
    v_up = list(np.arange(step, up + 1e-9, step))
    u += [u[-1]] * len(v_up)
    v += v_up
    u, v = np.array(u), np.array(v)
    if mirror:
        u = -u
    c, s_ = np.cos(theta), np.sin(theta)
    x, y = 10 + c * u - s_ * v, 20 + s_ * u + c * v
    n = len(x)
    s = np.r_[np.zeros(still), np.full(n - still, speed)]
    return np.arange(n) * DT, x, y, s, np.zeros(n)


def test_onset_index_requires_sustained_speed():
    s = np.array([0, 1.2, 0, 0, 1.1, 1.5, 2.0, 3.0])
    assert onset_index(s) == 4
    assert onset_index(np.zeros(10)) is None


def test_to_intrinsic_signs_forward_toward_upfield():
    _, x, y, *_ = l_path()
    u, v = to_intrinsic(x, y)
    assert u.max() == pytest.approx(8.0, abs=0.01)
    assert v[-1] == pytest.approx(6.0, abs=0.01)


@pytest.mark.parametrize("theta,mirror", [(0.0, False), (1.1, False), (2.5, True), (-0.7, True)])
def test_metrics_invariant_to_drill_orientation(theta, mirror):
    # 4.7 yd/s keeps frames off the exact 3/5-yd thresholds, where rotation rounding flips them
    base = rep_metrics(*l_path(speed=4.7))
    m = rep_metrics(*l_path(speed=4.7, theta=theta, mirror=mirror))
    # At the synthetic path's perfectly sharp corner lateral/forward velocity tie exactly, so
    # rounding can move the detected turn by one frame (0.1 s / 0.47 yd).
    one_frame = {"time_to_turn": 0.11, "lat_at_turn": 0.48}
    for k, val in base.items():
        assert m[k] == pytest.approx(val, nan_ok=True, abs=one_frame.get(k, 1e-6)), k


def test_metric_values_on_known_path():
    m = rep_metrics(*l_path(lat=8.0, up=6.0, speed=5.0, still=5))
    assert m["onset_t"] == pytest.approx(0.5)
    assert m["time_to_lat5"] == pytest.approx(1.0, abs=0.11)  # 5 yd at 5 yd/s
    assert m["peak_speed"] == 5.0
    assert m["depth_lost"] == pytest.approx(0.0)
    assert m["path_eff_lat5"] == pytest.approx(1.0, abs=1e-6)
    assert m["lat_at_turn"] == pytest.approx(8.0, abs=0.01)
    assert m["turn_speed_retention"] == pytest.approx(1.0)


def test_depth_lost_detects_drop_step():
    t, x, y, s, a = l_path(still=3)
    y = y.copy()
    y[3:8] -= np.array([0.3, 0.6, 0.8, 0.6, 0.3])  # dip 0.8 yd behind the line early
    m = rep_metrics(t, x, y, s, a, upfield=np.array([0.0, 1.0]))
    assert m["depth_lost"] == pytest.approx(0.8, abs=0.05)


def test_short_pull_never_reaching_5_yards_is_nan():
    m = rep_metrics(*l_path(lat=4.0))
    assert np.isnan(m["time_to_lat5"])
    assert m["lat_at_turn"] == pytest.approx(4.0, abs=0.01)
