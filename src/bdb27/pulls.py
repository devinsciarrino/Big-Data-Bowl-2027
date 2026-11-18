"""Offensive-line pulls: Combine pull drills and in-game pulls, measured the same way.

Pipeline
--------
1. ``combine_pull_frames()``: every OL pull-drill rep from ``combine_tracking``.
2. ``run_snap_frames()`` + ``snap_features()`` + ``flag_pulls()``: every Combine lineman's
   run snap in ``game_tracking``, in a field frame (origin = his spot at the snap, +x upfield,
   +y = the offense's left), then a rule that flags pulls.
3. ``rep_metrics()``: the SAME metric function applied to a drill rep or a game pull, after
   both are put in a path-intrinsic frame (see ``to_intrinsic``). This is what makes the
   drill-to-game comparison apples to apples.

Frames and conventions
----------------------
- Time ``t`` is seconds from movement onset (speed first sustained >= ONSET_SPEED). Combine
  reps have no snap, so onset is the only common anchor. Game get-off (snap -> onset) is
  reported separately as ``getoff_s``.
- Intrinsic frame: ``u`` = lateral (along the pull's initial direction), ``v`` = forward
  (perpendicular, signed upfield). Game pulls use the true upfield direction; Combine reps
  are signed so the rep finishes upfield. Drill orientation on the Combine field varies by
  drill and some reps are mirrored, so this frame removes both.
- Speeds are from the tracking ``s`` column (yd/s); headings come from position differences,
  so no reliance on the ``dir`` angle convention.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from bdb27.data import game_tracking, load

ONSET_SPEED = 1.0  # yd/s, sustained for ONSET_FRAMES frames
ONSET_FRAMES = 3
AXIS_DIST = 3.0  # yd from origin used to fix the lateral axis
GAME_WINDOW_S = 3.0  # seconds after the snap kept for game snaps
PULL_DRILLS = (
    "OL_PULL_DRILL_FOLD_BLOCK_RIGHT",
    "OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT",
    "OL_PULL_DRILL_LONG_PULL_RIGHT",
)
OL_SPOTS = ("T", "G", "C")


# ----------------------------------------------------------------------------- inputs


def combine_ol_ids() -> pl.Series:
    cr = load("combine_results")
    return cr.filter(pl.col("combine_position") == "OL")["nfl_id"]


def combine_pull_frames() -> pl.DataFrame:
    """All OL pull-drill reps. One row per frame, keyed by ``rep_id``."""
    ct = load("combine_tracking", lazy=True)
    return (
        ct.filter(pl.col("entity_type") == "PLAYER", pl.col("drill_name").is_in(PULL_DRILLS))
        .with_columns(pl.col("time").str.to_datetime())
        .with_columns(rep_id=pl.concat_str("event_id", "nfl_id", "attempt", separator="|"))
        .sort("rep_id", "time")
        .with_columns(t_raw=(pl.col("time") - pl.col("time").first().over("rep_id"))
                      .dt.total_milliseconds() / 1000)
        .select("rep_id", "nfl_id", "draft_year", "drill_name", "attempt", "t_raw", "x", "y", "s", "a")
        .collect()
    )


def ol_run_plays() -> pl.DataFrame:
    """Run snaps for Combine OL lined up at T/G/C (kneels and aborted snaps dropped)."""
    pp = load("player_play")
    desc = pl.col("play_description").str.to_lowercase()
    return (
        pp.filter(
            pl.col("nfl_id").is_in(combine_ol_ids().implode()),
            pl.col("lined_up_position").is_in(OL_SPOTS),
            pl.col("pass_result").is_null(),
            pl.col("dropback_duration").is_null(),
            ~desc.str.contains("kneels|aborted|no play"),
        )
        .with_columns(run_dir=desc.str.extract(
            r"(left end|left tackle|left guard|up the middle|right guard|right tackle|right end)", 1))
        .select("game_id", "play_id", "nfl_id", "team_abbr", "lined_up_position", "play_direction",
                "run_dir", "down", "yards_to_go", "offense_formation", "yards_gained",
                "expected_points_added")
    )


def run_snap_frames(plays: pl.DataFrame | None = None) -> pl.DataFrame:
    """Field-frame tracking for each run snap, from the snap to GAME_WINDOW_S after it.

    Adds ``t_snap`` (s since snap), ``fx`` (upfield yd), ``fy`` (yd toward the offense's left),
    and ``t_contact`` (s from snap to the play's first_contact event, if any).
    """
    plays = ol_run_plays() if plays is None else plays
    keys = plays.select("game_id", "play_id", "nfl_id", "play_direction").lazy()
    gt = (game_tracking().join(keys, on=["game_id", "play_id", "nfl_id"])
          .with_columns(pl.col("time").str.to_datetime()))
    k = ["game_id", "play_id", "nfl_id"]
    snap = (gt.filter(pl.col("event") == "ball_snap").group_by(k)
            .agg(t0=pl.col("time").min(), x0=pl.col("x").first(), y0=pl.col("y").first()))
    contact = (gt.filter(pl.col("event") == "first_contact").group_by(k)
               .agg(t_c=pl.col("time").min()))
    sgn = pl.when(pl.col("play_direction") == "left").then(-1.0).otherwise(1.0)
    return (
        gt.join(snap, on=k).join(contact, on=k, how="left")
        .with_columns(t_snap=(pl.col("time") - pl.col("t0")).dt.total_milliseconds() / 1000,
                      t_contact=(pl.col("t_c") - pl.col("t0")).dt.total_milliseconds() / 1000)
        .filter(pl.col("t_snap").is_between(-0.05, GAME_WINDOW_S + 0.05))
        .with_columns(fx=(pl.col("x") - pl.col("x0")) * sgn, fy=(pl.col("y") - pl.col("y0")) * sgn)
        .select(*k, "t_snap", "fx", "fy", "s", "a", "t_contact")
        .sort(*k, "t_snap")
        .collect()
    )


# ----------------------------------------------------------------------- pull detection


def snap_features(frames: pl.DataFrame) -> pl.DataFrame:
    """Per-snap field-frame summaries used to flag pulls."""
    def at(col: str, tt: float) -> pl.Expr:
        return pl.col(col).filter((pl.col("t_snap") - tt).abs() < 0.051).first()

    k = ["game_id", "play_id", "nfl_id"]
    early = pl.col("t_snap") <= 1.5
    return (
        frames.with_columns(
            vy=pl.col("fy").diff().over(k) / pl.col("t_snap").diff().over(k))
        .group_by(k)
        .agg(lat10=at("fy", 1.0), lat20=at("fy", 2.0), fwd10=at("fx", 1.0), fwd15=at("fx", 1.5),
             fwd_max15=pl.col("fx").filter(early).max(),
             lat_speed_max=pl.col("vy").filter(pl.col("t_snap") <= 2.0).abs().max(),
             n_frames=pl.len(), t_contact=pl.col("t_contact").first())
    )


def flag_pulls(feats: pl.DataFrame, min_lat: float = 5.0, max_fwd: float = 1.0,
               min_lat_speed: float = 4.0, drop_fwd10: float = -0.1, drop_fwd_max15: float = 0.25
               ) -> pl.DataFrame:
    """Rule-based labels for lateral run-block movement.

    ``lateral`` (any fast, flat lateral run at the line):
    - |lateral| at 2.0 s >= ``min_lat`` yd
    - never more than ``max_fwd`` yd upfield in the first 1.5 s (zone climbs/reaches go upfield)
    - peak lateral speed in the first 2.0 s >= ``min_lat_speed`` yd/s

    ``move_type`` splits lateral snaps into
    - ``pull``: also drops behind his snap spot (fx at 1.0 s <= ``drop_fwd10``) and stays at or
      behind the line through 1.5 s (max fx <= ``drop_fwd_max15``): the open/bucket step of a
      gap-scheme pull.
    - ``lateral_flow``: the rest, mostly outside-zone flow with the whole line.
    Thresholds come from plays with 2+ tracked rookie linemen: when the teammate stays home
    (a solo pull) 90% have fx(1.0 s) <= -0.09; when the teammate flows the same way, the median
    is 0.0 (see notebooks/01_pulls.ipynb). ``pull`` is the column the drill comparison uses.
    """
    lateral = ((pl.col("lat20").abs() >= min_lat) & (pl.col("fwd_max15") <= max_fwd)
               & (pl.col("lat_speed_max") >= min_lat_speed))
    dropped = (pl.col("fwd10") <= drop_fwd10) & (pl.col("fwd_max15") <= drop_fwd_max15)
    return feats.with_columns(
        lateral=lateral,
        pull=lateral & dropped,
        move_type=pl.when(lateral & dropped).then(pl.lit("pull"))
        .when(lateral).then(pl.lit("lateral_flow")).otherwise(pl.lit("other")),
        pull_dir=pl.when(pl.col("lat20") > 0).then(pl.lit("left")).otherwise(pl.lit("right")),
    )


# ----------------------------------------------------------------- shared rep geometry


def onset_index(s: np.ndarray, thresh: float = ONSET_SPEED, frames: int = ONSET_FRAMES) -> int | None:
    """First index where speed stays >= thresh for ``frames`` consecutive frames."""
    ok = s >= thresh
    if len(ok) < frames:
        return None
    run = np.convolve(ok.astype(int), np.ones(frames, dtype=int), mode="valid")
    hits = np.flatnonzero(run == frames)
    return int(hits[0]) if hits.size else None


def to_intrinsic(x: np.ndarray, y: np.ndarray, upfield: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray] | None:
    """Rotate a path (origin at index 0) into (u = lateral, v = forward).

    The lateral axis points from the origin to the first point >= AXIS_DIST away. The forward
    axis is perpendicular, signed toward ``upfield`` if given (a 2-vector), else toward the
    path's final point.
    """
    dx, dy = x - x[0], y - y[0]
    d = np.hypot(dx, dy)
    far = np.flatnonzero(d >= AXIS_DIST)
    if not far.size:
        return None
    ex = np.array([dx[far[0]], dy[far[0]]]) / d[far[0]]
    ey = np.array([-ex[1], ex[0]])
    ref = upfield if upfield is not None else np.array([dx[-1], dy[-1]])
    if ey @ ref < 0:
        ey = -ey
    return dx * ex[0] + dy * ex[1], dx * ey[0] + dy * ey[1]


def rep_metrics(t: np.ndarray, x: np.ndarray, y: np.ndarray, s: np.ndarray, a: np.ndarray,
                upfield: np.ndarray | None = None) -> dict[str, float]:
    """Pull metrics for one rep (drill or game), identical definitions for both.

    Inputs are frames from wherever the rep starts; the rep is re-anchored at movement onset.
    Returns NaN for metrics the rep never reaches (e.g. a 4-yd fold never hits 5 yd lateral).
    """
    nan = float("nan")
    keys = ("onset_t", "time_to_lat3", "time_to_lat5", "peak_speed", "peak_accel_1s",
            "speed_at_lat5", "depth_lost", "path_eff_lat5", "lat_at_turn", "time_to_turn",
            "turn_speed_retention", "max_lat")
    out = dict.fromkeys(keys, nan)
    i0 = onset_index(s)
    if i0 is None:
        return out
    out["onset_t"] = float(t[i0] - t[0])
    j = max(i0 - 1, 0)  # last (near-)stationary frame = origin
    t, x, y, s, a = t[j:] - t[i0], x[j:], y[j:], s[j:], a[j:]
    uv = to_intrinsic(x, y, upfield)
    if uv is None:
        return out
    u, v = uv

    def first(mask: np.ndarray) -> int | None:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if idx.size else None

    out["peak_speed"] = float(s[t <= 2.5].max())
    out["peak_accel_1s"] = float(a[t <= 1.0].max())
    out["max_lat"] = float(u.max())
    i3, i5 = first(u >= 3.0), first(u >= 5.0)
    if i3 is not None:
        out["time_to_lat3"] = float(t[i3])
    if i5 is not None:
        seg = np.hypot(np.diff(x[: i5 + 1]), np.diff(y[: i5 + 1])).sum()
        out.update(time_to_lat5=float(t[i5]), speed_at_lat5=float(s[i5]),
                   path_eff_lat5=float(np.hypot(u[i5], v[i5]) / seg) if seg > 0 else nan)

    # Turn upfield: first frame after 2 yd lateral where forward velocity exceeds lateral velocity.
    du, dv = np.gradient(u), np.gradient(v)
    it = first((u >= 2.0) & (dv > np.abs(du)) & (dv > 0))
    pre = slice(0, it + 1) if it is not None else slice(None)
    out["depth_lost"] = float(max(0.0, -v[pre].min()))
    if it is not None:
        peak_before = s[: it + 1].max()
        out.update(lat_at_turn=float(u[it]), time_to_turn=float(t[it]),
                   turn_speed_retention=float(s[it] / peak_before) if peak_before > 0 else nan)
    return out


# ------------------------------------------------------------------------ metric tables


def combine_pull_metrics(frames: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per Combine pull-drill rep."""
    frames = combine_pull_frames() if frames is None else frames
    rows = []
    for (rep_id,), r in frames.group_by(["rep_id"], maintain_order=True):
        m = rep_metrics(*(r[c].to_numpy() for c in ("t_raw", "x", "y", "s", "a")))
        rows.append({"rep_id": rep_id, "nfl_id": r["nfl_id"][0], "drill_name": r["drill_name"][0],
                     "attempt": r["attempt"][0], **m})
    return pl.DataFrame(rows)


def game_pull_metrics(frames: pl.DataFrame, pulls: pl.DataFrame) -> pl.DataFrame:
    """One row per lateral game snap (pulls and lateral flow); ``pulls`` = flag_pulls(...)."""
    k = ["game_id", "play_id", "nfl_id"]
    sel = frames.join(pulls.filter(pl.col("lateral")).select(k), on=k)
    rows = []
    up = np.array([1.0, 0.0])  # field frame: +fx is upfield
    for key, r in sel.group_by(k, maintain_order=True):
        r = r.filter(pl.col("t_snap") >= 0)
        m = rep_metrics(*(r[c].to_numpy() for c in ("t_snap", "fx", "fy", "s", "a")), upfield=up)
        m["getoff_s"] = m.pop("onset_t")  # in games the anchor is the snap, so onset = get-off
        rows.append(dict(zip(k, key), **m))
    return pl.DataFrame(rows)
