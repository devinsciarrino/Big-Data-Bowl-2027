"""Synthetic tests for drive-block and 40-start metrics (no competition data needed)."""

import math

import numpy as np
import pytest

from bdb27.drive import drive_snap_metrics, forty_metrics, interp_time_at

DT = 0.1


def test_interp_time_at():
    t = np.array([0.0, 0.1, 0.2])
    d = np.array([0.0, 1.0, 3.0])
    assert interp_time_at(t, d, 2.0) == pytest.approx(0.15)
    assert math.isnan(interp_time_at(t, d, 5.0))


def test_forty_constant_acceleration():
    # starts at 0.6 yd/s, accelerates at 4 yd/s^2
    t = np.arange(0, 3.0, DT)
    s = 0.6 + 4 * t
    d = 0.6 * t + 2 * t**2
    m = forty_metrics(t, d, s, np.full_like(t, 4.0))
    assert m["speed_05"] == pytest.approx(2.6)
    assert m["avg_accel_05"] == pytest.approx(4.0)
    assert m["dist_05"] == pytest.approx(0.8)
    assert m["time_to_2yd"] == pytest.approx(0.86, abs=0.02)  # 0.6t + 2t^2 = 2


def drive_path(fwd_speed=1.2, contact_t=0.8, after_speed=0.6, lateral=0.0, getoff=0.2, end=3.0):
    """Stand until getoff, ramp to fwd_speed, sharp slowdown at contact_t, then after_speed."""
    t = np.round(np.arange(0, end + 1e-9, DT), 1)
    s = np.where(t < getoff, 0.0, np.minimum(fwd_speed, (t - getoff + DT) * 6))
    s = np.where(t > contact_t, after_speed, s)
    fx = np.concatenate([[0.0], np.cumsum(s[1:] * DT)])
    fy = lateral * np.clip(t / 1.0, 0, 1)
    return t, fx, fy, s


def test_drive_snap_detects_contact_and_drive():
    t, fx, fy, s = drive_path()
    m = drive_snap_metrics(t, fx, fy, s)
    assert m is not None
    assert m["getoff_s"] == pytest.approx(0.2)
    assert m["t_block_contact"] == pytest.approx(0.8)
    assert m["drive_after_contact"] == pytest.approx(0.6, abs=0.07)  # 1 s at 0.6 yd/s
    assert 0 < m["sustain_ratio"] < 1


def test_lateral_snap_is_not_a_drive_block():
    t, fx, _, s = drive_path()
    fy = np.clip(t, 0, None) * 3.0  # first step mostly sideways
    assert drive_snap_metrics(t, fx, fy, s) is None


def test_drifting_wide_is_not_a_drive_block():
    t, fx, fy, s = drive_path(lateral=2.0)
    assert drive_snap_metrics(t, fx, fy, s) is None


def test_no_contact_leaves_contact_metrics_nan():
    t, fx, fy, s = drive_path(after_speed=1.2)  # never slows down
    m = drive_snap_metrics(t, fx, fy, s)
    assert math.isnan(m["t_block_contact"]) and math.isnan(m["drive_after_contact"])
