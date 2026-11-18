"""Which side of the center each tracked lineman lines up on (LT/LG/C/RG/RT).

``player_play.lined_up_position`` only says T / G / C. Game tracking holds only the Combine
rookies, so there is no ball and no full line to read alignment from. But on plays where two or
more tracked rookie linemen are on the same offense, their lateral spacing at the snap pins
down the slots: adjacent linemen stand ~1.55 yd apart (C-G median 1.54, G-T same side 1.81,
G-T opposite sides 4.7, T-T 6.15).

1. ``solve_slots``: for one play, try every slot assignment allowed by the labels (T = ±2,
   G = ±1, C = 0; + = offense's left) and keep the one whose ideal spacing best fits the
   observed lateral positions.
2. ``player_sides``: majority side per player-game from those solved plays (falling back to
   player-season, then player-career), applied to all of that player's T/G snaps.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl

from bdb27.data import game_tracking, load

SPACING = 1.55  # yd between adjacent linemen at the snap
SLOTS = {"T": (2, -2), "G": (1, -1), "C": (0,)}
SLOT_NAME = {2: "LT", 1: "LG", 0: "C", -1: "RG", -2: "RT"}
MAX_RMSE = 0.6  # yd; worse fits are treated as unsolved (unbalanced lines, odd formations)


def solve_slots(labels: list[str], yl: list[float], spacing: float = SPACING
                ) -> tuple[tuple[int, ...] | None, float, float]:
    """Best slot assignment for linemen with ``labels`` at lateral positions ``yl``.

    ``yl`` is in the offense's frame (+ = offense's left). Returns (slots, rmse, margin), where
    margin = second-best rmse - best rmse (how decisively the best wins). Slots are None when
    no assignment with distinct slots exists.
    """
    y = np.asarray(yl, dtype=float)
    fits = []
    for combo in itertools.product(*(SLOTS[lab] for lab in labels)):
        if len(set(combo)) < len(combo):
            continue
        ideal = spacing * np.asarray(combo, dtype=float)
        resid = (y - ideal) - (y - ideal).mean()  # free offset: the ball's spot is unknown
        fits.append((float(np.sqrt((resid**2).mean())), combo))
    if not fits:
        return None, float("nan"), float("nan")
    fits.sort()
    margin = fits[1][0] - fits[0][0] if len(fits) > 1 else float("inf")
    return fits[0][1], fits[0][0], margin


def snap_positions(nfl_ids: pl.Series) -> pl.DataFrame:
    """Each T/G/C snap of these players with their field position at the ball snap.

    ``yl`` = lateral position in the offense's frame (+ = offense's left).
    """
    pp = load("player_play").filter(pl.col("nfl_id").is_in(nfl_ids.implode()),
                                    pl.col("lined_up_position").is_in(list(SLOTS)))
    keys = pp.select("game_id", "play_id", "nfl_id", "team_abbr", "lined_up_position",
                     "play_direction").lazy()
    sgn = pl.when(pl.col("play_direction") == "left").then(-1.0).otherwise(1.0)
    return (
        game_tracking().join(keys, on=["game_id", "play_id", "nfl_id"])
        .filter(pl.col("event") == "ball_snap")
        .group_by("game_id", "play_id", "nfl_id", "team_abbr", "lined_up_position", "season")
        .agg(y=pl.col("y").first(), sgn=sgn.first())
        .with_columns(yl=pl.col("y") * pl.col("sgn"))
        .drop("y", "sgn")
        .collect()
    )


def solve_plays(snaps: pl.DataFrame) -> pl.DataFrame:
    """Slot per lineman on every play with 2+ tracked teammates on the line."""
    k = ["game_id", "play_id", "team_abbr"]
    multi = snaps.filter(pl.len().over(k) >= 2).sort(*k, "nfl_id")
    rows = []
    for key, g in multi.group_by(k, maintain_order=True):
        slots, rmse, margin = solve_slots(g["lined_up_position"].to_list(), g["yl"].to_list())
        for nid, slot in zip(g["nfl_id"], slots or [None] * g.height):
            rows.append((*key, nid, slot, rmse, margin))
    out = pl.DataFrame(rows, schema=[*k, "nfl_id", "slot", "rmse", "margin"], orient="row")
    return out.with_columns(solved=pl.col("slot").is_not_null() & (pl.col("rmse") <= MAX_RMSE))


def player_sides(snaps: pl.DataFrame, solved: pl.DataFrame) -> pl.DataFrame:
    """Side (L/R) for every T/G snap; C snaps get side "C".

    Votes come from solved multi-lineman plays. Each snap takes the majority side of that
    player at that position in the same game; if he has no solved plays that game, the
    player-season majority; then the player-career majority. ``side_source`` and
    ``side_agreement`` (share of votes for the winning side) record how it was decided.
    """
    votes = (snaps.join(solved.filter("solved").select("game_id", "play_id", "nfl_id", "slot"),
                        on=["game_id", "play_id", "nfl_id"])
             .filter(pl.col("lined_up_position") != "C")
             .with_columns(left=pl.col("slot") > 0))
    out = snaps.with_columns(side=pl.when(pl.col("lined_up_position") == "C").then(pl.lit("C")),
                             side_source=pl.when(pl.col("lined_up_position") == "C")
                             .then(pl.lit("center")),
                             side_agreement=pl.when(pl.col("lined_up_position") == "C")
                             .then(pl.lit(1.0)))
    for level, keys in (("game", ["nfl_id", "lined_up_position", "game_id"]),
                        ("season", ["nfl_id", "lined_up_position", "season"]),
                        ("career", ["nfl_id", "lined_up_position"])):
        maj = votes.group_by(keys).agg(p_left=pl.col("left").mean(), n_votes=pl.len())
        maj = maj.with_columns(
            side_l=pl.when(pl.col("p_left") > 0.5).then(pl.lit("L"))
            .when(pl.col("p_left") < 0.5).then(pl.lit("R")),
            agree_l=pl.max_horizontal(pl.col("p_left"), 1 - pl.col("p_left")))
        out = (out.join(maj.select(*keys, "side_l", "agree_l"), on=keys, how="left")
               .with_columns(
                   side_source=pl.when(pl.col("side").is_null() & pl.col("side_l").is_not_null())
                   .then(pl.lit(level)).otherwise(pl.col("side_source")),
                   side_agreement=pl.coalesce("side_agreement",
                                              pl.when(pl.col("side").is_null()).then("agree_l")),
                   side=pl.coalesce("side", "side_l"))
               .drop("side_l", "agree_l"))
    return out.with_columns(
        spot=pl.when(pl.col("side") == "C").then(pl.lit("C"))
        .when(pl.col("side").is_not_null())
        .then(pl.concat_str(pl.col("side"), pl.col("lined_up_position"))))
