"""Build the pull tables into data/processed/pulls/.

    uv run python scripts/build_pulls.py

Outputs
- combine_reps.parquet : one row per Combine pull-drill rep, with metrics
- run_snaps.parquet    : one row per Combine-OL run snap, with field-frame features + pull flag
- game_pulls.parquet   : one row per lateral game snap (move_type pull / lateral_flow), with
                         the same metrics as the drill reps
"""

import time

import polars as pl

from bdb27 import pulls
from bdb27.data import PROCESSED

OUT = PROCESSED / "pulls"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()

    creps = pulls.combine_pull_metrics()
    creps.write_parquet(OUT / "combine_reps.parquet")
    print(f"combine reps: {creps.height} ({time.time() - t:.0f}s)")

    plays = pulls.ol_run_plays()
    frames = pulls.run_snap_frames(plays)
    snaps = pulls.flag_pulls(pulls.snap_features(frames)).join(
        plays, on=["game_id", "play_id", "nfl_id"], how="left")
    snaps.write_parquet(OUT / "run_snaps.parquet")
    print(f"run snaps: {snaps.height}, lateral: {snaps['lateral'].sum()}, "
          f"pulls: {snaps['pull'].sum()} ({time.time() - t:.0f}s)")

    gp = pulls.game_pull_metrics(frames, snaps).join(
        snaps, on=["game_id", "play_id", "nfl_id"], how="left")
    gp.write_parquet(OUT / "game_pulls.parquet")
    print(f"lateral snaps with metrics: {gp.height} ({time.time() - t:.0f}s)")

    print(snaps.group_by("lined_up_position").agg(
        snaps=pl.len(), pulls=pl.col("pull").sum(), pull_rate=pl.col("pull").mean().round(3),
        flow_rate=(pl.col("move_type") == "lateral_flow").mean().round(3)).sort("lined_up_position"))
    per = snaps.filter(pl.col("pull")).group_by("nfl_id").len()
    print("linemen with >=10 / >=20 pulls:", (per["len"] >= 10).sum(), (per["len"] >= 20).sum())


if __name__ == "__main__":
    main()
