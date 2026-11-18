"""Data loading for the BDB 2027 files.

Raw CSVs live in data/raw/ (see scripts/download_data.sh). The first load of each
file converts it to Parquet in data/processed/, which is much faster to re-read,
especially for the large game_tracking_* files.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def available() -> list[str]:
    """Names (without extension) of the raw CSVs present."""
    return sorted(p.stem for p in RAW.glob("*.csv"))


def load(name: str, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Load a competition file by stem, e.g. load("combine_tracking").

    Pass lazy=True for the big tracking files so filters/selects are pushed down.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pq = PROCESSED / f"{name}.parquet"
    if not pq.exists():
        csv = RAW / f"{name}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv} not found; have: {available()}")
        pl.scan_csv(csv, null_values=["NA", ""], infer_schema_length=100_000).sink_parquet(pq)
    return pl.scan_parquet(pq) if lazy else pl.read_parquet(pq)


def game_tracking(seasons: tuple[int, ...] = (2023, 2024, 2025)) -> pl.LazyFrame:
    """All in-game tracking seasons as one lazy frame with a `season` column."""
    return pl.concat(
        [load(f"game_tracking_{s}", lazy=True).with_columns(pl.lit(s).alias("season"))
         for s in seasons],
        how="diagonal_relaxed",
    )
