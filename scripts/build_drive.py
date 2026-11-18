"""Build the drive-block tables into data/processed/drive/.

    uv run python scripts/build_drive.py

Outputs
- forty_reps.parquet  : one row per Combine OL 40 rep, start-of-run burst metrics
- drive_snaps.parquet : one row per game drive-block snap, with metrics + play context + spot
- players.parquet     : one row per Combine lineman: measurables, 40 burst, game drive summary
"""

import time

import polars as pl

from bdb27 import drive, pulls
from bdb27.data import PROCESSED

OUT = PROCESSED / "drive"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()

    reps = drive.combine_burst()
    reps.write_parquet(OUT / "forty_reps.parquet")
    print(f"40 reps: {reps.height} from {reps['nfl_id'].n_unique()} linemen ({time.time() - t:.0f}s)")

    plays = pulls.ol_run_plays()
    frames = pulls.run_snap_frames(plays)
    k = ["game_id", "play_id", "nfl_id"]
    snaps = drive.game_drive_snaps(frames).join(plays, on=k, how="left")
    sides = PROCESSED / "pulls" / "ol_sides.parquet"
    if sides.exists():  # written by build_pulls.py
        snaps = snaps.join(pl.read_parquet(sides).select(*k, "spot"), on=k, how="left")
    snaps.write_parquet(OUT / "drive_snaps.parquet")
    print(f"drive snaps: {snaps.height} of {plays.height} run snaps ({time.time() - t:.0f}s)")

    players = drive.player_table(reps, snaps)
    players.write_parquet(OUT / "players.parquet")
    n = players.filter(pl.col("forty_reps").is_not_null(), pl.col("drive_snaps") >= 50).height
    print(f"players: {players.height}; with tracked 40 and >=50 drive snaps: {n}")


if __name__ == "__main__":
    main()
