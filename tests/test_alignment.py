"""Slot solver tests on synthetic snap alignments."""

import math

import pytest

from bdb27.alignment import SPACING, solve_slots


def line(slots, offset=12.3, jitter=()):
    """Lateral positions for linemen at `slots` (+ = offense's left), ball at `offset`."""
    j = list(jitter) + [0.0] * (len(slots) - len(jitter))
    return [offset + SPACING * s + e for s, e in zip(slots, j)]


@pytest.mark.parametrize("labels,slots", [
    (["T", "T"], (2, -2)),
    (["T", "T"], (-2, 2)),
    (["G", "T"], (1, 2)),      # LG + LT: same side, 1 slot apart
    (["G", "T"], (1, -2)),     # LG + RT: opposite sides, 3 slots apart
    (["C", "G"], (0, -1)),
    (["G", "C", "T"], (-1, 0, 2)),
])
def test_recovers_true_slots(labels, slots):
    got, rmse, margin = solve_slots(labels, line(slots, jitter=(0.2, -0.15, 0.1)))
    assert got == slots
    assert rmse < 0.3
    assert margin > 0.3


def test_offset_is_free():
    a, *_ = solve_slots(["G", "T"], line((-1, -2), offset=0.0))
    b, *_ = solve_slots(["G", "T"], line((-1, -2), offset=40.0))
    assert a == b == (-1, -2)


def test_impossible_labels_return_none():
    slots, rmse, _ = solve_slots(["C", "C"], [10.0, 11.5])
    assert slots is None and math.isnan(rmse)
