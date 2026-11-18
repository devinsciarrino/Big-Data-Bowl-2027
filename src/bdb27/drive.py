"""Drive (downhill) run blocks: Combine burst traits vs. moving people in games.

Unlike pulls, there is no tracked Combine drill that matches a drive block: in games a drive
blocker meets the defender within ~1 yd (median 0.84 yd upfield at 1.0 s, peak speed ~1.4 yd/s),
and no OL drill involves contact. So this module links Combine *traits* to game *outcomes*:

Combine (``forty_metrics``): the first 0.5 s of the tracked 40 (3-point stance, like a run
block's start): speed / distance / peak acceleration, momentum (weight x speed), times to 2 and
5 yd.

Game (``drive_snap_metrics``): run snaps whose first step goes forward (within 60 deg of
upfield) and that stay narrow (|lateral| <= 1.5 yd through 1.5 s), which covers base, down
and angled drive blocks but not pulls / zone flow. Per snap: get-off, first-step speed,
upfield yards at 1.0/1.5/2.5 s, contact (first sharp slowdown), drive after contact, stall /
pushed-back flags.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from bdb27.data import load
from bdb27.pulls import combine_ol_ids, onset_index

DRIVE_ONSET_SPEED = 0.5  # yd/s; Combine 40 reps start already moving at ~0.6 yd/s
DRIVE_ONSET_FRAMES = 2
FIRST_STEP_DIST = 0.3  # yd from snap spot that defines the first-step direction
MAX_FIRST_STEP_ANGLE = 60.0  # degrees from upfield (down blocks / doubles step at 45-60)
MAX_LATERAL = 1.5  # yd, through 1.5 s
CONTACT_DROP = 0.25  # fractional speed drop within CONTACT_WINDOW s that marks contact
CONTACT_WINDOW = 0.3


def interp_time_at(t: np.ndarray, d: np.ndarray, target: float) -> float:
    """First time progress ``d`` reaches ``target``, linearly interpolated (NaN if never)."""
    idx = np.flatnonzero(d >= target)
    if not idx.size:
        return float("nan")
    i = idx[0]
    if i == 0:
        return float(t[0])
    return float(np.interp(target, [d[i - 1], d[i]], [t[i - 1], t[i]]))


def value_at(t: np.ndarray, v: np.ndarray, tt: float) -> float:
    """Linear interpolation of ``v`` at time ``tt`` (NaN outside the observed window)."""
    if tt < t[0] - 1e-9 or tt > t[-1] + 1e-9:
        return float("nan")
    return float(np.interp(tt, t, v))


# ------------------------------------------------------------------------------ Combine


def combine_forty_frames() -> pl.DataFrame:
    """OL 40-yard-dash reps; ``d`` = progress (yd) along the rep's start->finish line."""
    ct = load("combine_tracking", lazy=True)
    df = (ct.filter(pl.col("entity_type") == "PLAYER", pl.col("drill_type") == "FORTY_YARD_DASH",
                    pl.col("nfl_id").is_in(combine_ol_ids().implode()))
          .with_columns(pl.col("time").str.to_datetime(),
                        rep_id=pl.concat_str("event_id", "nfl_id", "attempt", separator="|"))
          .sort("rep_id", "time").collect())
    g = df.with_columns(
        x0=pl.col("x").first().over("rep_id"), y0=pl.col("y").first().over("rep_id"),
        x1=pl.col("x").last().over("rep_id"), y1=pl.col("y").last().over("rep_id"))
    length = ((pl.col("x1") - pl.col("x0")) ** 2 + (pl.col("y1") - pl.col("y0")) ** 2).sqrt()
    return g.with_columns(
        t=(pl.col("time") - pl.col("time").first().over("rep_id")).dt.total_milliseconds() / 1000,
        d=((pl.col("x") - pl.col("x0")) * (pl.col("x1") - pl.col("x0"))
           + (pl.col("y") - pl.col("y0")) * (pl.col("y1") - pl.col("y0"))) / length,
    ).select("rep_id", "nfl_id", "attempt", "t", "d", "s", "a")


def forty_metrics(t: np.ndarray, d: np.ndarray, s: np.ndarray, a: np.ndarray) -> dict[str, float]:
    """Start-of-40 burst metrics, time measured from movement onset."""
    keys = ("speed_05", "dist_05", "peak_accel_05", "avg_accel_05", "time_to_2yd", "time_to_5yd")
    out = dict.fromkeys(keys, float("nan"))
    i0 = onset_index(s, DRIVE_ONSET_SPEED, DRIVE_ONSET_FRAMES)
    if i0 is None:
        return out
    t, d, s, a = t[i0:] - t[i0], d[i0:] - d[i0], s[i0:], a[i0:]
    s05 = value_at(t, s, 0.5)
    out.update(speed_05=s05, dist_05=value_at(t, d, 0.5),
               peak_accel_05=float(a[t <= 0.5 + 1e-9].max()),
               avg_accel_05=(s05 - s[0]) / 0.5,
               time_to_2yd=interp_time_at(t, d, 2.0), time_to_5yd=interp_time_at(t, d, 5.0))
    return out


def combine_burst() -> pl.DataFrame:
    """One row per Combine OL 40 rep with burst metrics (player means: ``player_table``)."""
    frames = combine_forty_frames()
    rows = []
    for (rep_id,), r in frames.group_by(["rep_id"], maintain_order=True):
        m = forty_metrics(*(r[c].to_numpy() for c in ("t", "d", "s", "a")))
        rows.append({"rep_id": rep_id, "nfl_id": r["nfl_id"][0], "attempt": r["attempt"][0], **m})
    return pl.DataFrame(rows).fill_nan(None)


# --------------------------------------------------------------------------------- Game


def drive_snap_metrics(t: np.ndarray, fx: np.ndarray, fy: np.ndarray, s: np.ndarray
                       ) -> dict[str, float] | None:
    """Drive-block metrics for one run snap in the field frame (t from the snap).

    Returns None if the snap isn't a drive block (first step not forward, or drifts wide).
    """
    nan = float("nan")
    r = np.hypot(fx, fy)
    i_step = np.flatnonzero(r >= FIRST_STEP_DIST)
    if not i_step.size:
        return None  # never left his spot
    ang = np.degrees(np.arctan2(abs(fy[i_step[0]]), fx[i_step[0]]))
    if ang > MAX_FIRST_STEP_ANGLE or np.abs(fy[t <= 1.5 + 1e-9]).max() > MAX_LATERAL:
        return None

    i0 = onset_index(s, DRIVE_ONSET_SPEED, DRIVE_ONSET_FRAMES)
    getoff = float(t[i0]) if i0 is not None else nan
    out = {
        "first_step_angle": float(ang),
        "getoff_s": getoff,
        "first_step_speed": value_at(t, s, getoff + 0.3) if i0 is not None else nan,
        "fwd_10": value_at(t, fx, 1.0), "fwd_15": value_at(t, fx, 1.5),
        "fwd_25": value_at(t, fx, 2.5),
        "t_block_contact": nan, "speed_at_contact": nan, "fwd_at_contact": nan,
        "drive_after_contact": nan, "sustain_ratio": nan,
    }
    # Contact: the first sharp slowdown after onset. Find the first frame j whose speed is
    # CONTACT_DROP below the max of the preceding CONTACT_WINDOW s; contact = the last frame of
    # that max (the moment just before speed falls). Only contacts starting by 1.5 s count.
    if i0 is not None:
        n_win = round(CONTACT_WINDOW / 0.1)
        for j in range(i0 + 1, len(t)):
            lo = max(i0, j - n_win)
            win = s[lo:j]
            peak = win.max()
            if peak <= 0.5 or s[j] > (1 - CONTACT_DROP) * peak:
                continue
            i = lo + int(np.flatnonzero(win == peak)[-1])
            if t[i] > 1.5:
                break
            tc = float(t[i])
            after = value_at(t, fx, tc + 1.0)
            pre_v = fx[i] / tc if tc > 0 else nan  # avg upfield speed before contact
            out.update(t_block_contact=tc, speed_at_contact=float(s[i]),
                       fwd_at_contact=float(fx[i]), drive_after_contact=after - fx[i],
                       sustain_ratio=(after - fx[i]) / pre_v if pre_v and pre_v > 0 else nan)
            break
    return out


def game_drive_snaps(frames: pl.DataFrame) -> pl.DataFrame:
    """Drive-block snaps with metrics. ``frames`` = pulls.run_snap_frames(...)."""
    k = ["game_id", "play_id", "nfl_id"]
    rows = []
    for key, r in frames.filter(pl.col("t_snap") >= 0).group_by(k, maintain_order=True):
        m = drive_snap_metrics(*(r[c].to_numpy() for c in ("t_snap", "fx", "fy", "s")))
        if m is not None:
            rows.append(dict(zip(k, key), **m, t_max=float(r["t_snap"].max())))
    df = pl.DataFrame(rows).fill_nan(None)  # missing = null, so flags below stay null too
    return df.with_columns(
        pushed_back=pl.col("fwd_25") < 0,
        stalled=pl.col("drive_after_contact") < 0.25,
    )


# ------------------------------------------------------------------------------ Players


def player_table(reps: pl.DataFrame, snaps: pl.DataFrame) -> pl.DataFrame:
    """One row per Combine lineman: Combine burst + measurables, game drive summaries."""
    burst = reps.group_by("nfl_id").agg(
        [pl.col(c).mean().alias(c) for c in
         ("speed_05", "dist_05", "peak_accel_05", "avg_accel_05", "time_to_2yd", "time_to_5yd")],
        forty_reps=pl.len())
    game = snaps.group_by("nfl_id").agg(
        drive_snaps=pl.len(),
        getoff_s=pl.col("getoff_s").median(),
        first_step_speed=pl.col("first_step_speed").mean(),
        fwd_10=pl.col("fwd_10").mean(), fwd_15=pl.col("fwd_15").mean(),
        fwd_25=pl.col("fwd_25").mean(),
        drive_after_contact=pl.col("drive_after_contact").mean(),
        contact_rate=pl.col("t_block_contact").is_not_null().mean(),
        stall_rate=pl.col("stalled").mean(), pushed_back_rate=pl.col("pushed_back").mean())
    cr = load("combine_results").filter(pl.col("combine_position") == "OL").select(
        "nfl_id", "draft_year", "combine_height", "combine_weight", "hand_size", "arm_length",
        "wing_span", "ten_yd_split", "forty", "vertical", "broad_jump", "three_cone",
        "short_shuttle", "bench_reps", "ngs_athleticism_score")
    pl_ = load("players").select("nfl_id", "display_name", "nfl_position", "draft_overall_pick")
    out = (cr.join(pl_, on="nfl_id", how="left").join(burst, on="nfl_id", how="left")
           .join(game, on="nfl_id", how="left"))
    return out.with_columns(momentum_05=pl.col("combine_weight") * pl.col("speed_05"),
                            force_05=pl.col("combine_weight") * pl.col("peak_accel_05"))
