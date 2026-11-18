"""Fit Combine-trait -> game drive-block models and write results to data/processed/drive/.

    uv run python scripts/build_pulls.py && uv run python scripts/build_drive.py
    uv run python scripts/analyze_drive.py      # ~40 s

Outputs
- trait_models.parquet        : snap-level mixed models, every (spec, outcome, trait)
- player_correlations.parquet : within-position Spearman r with bootstrap CIs (cross-check)
"""

import polars as pl

from bdb27 import drive_models as dm


def main() -> None:
    res = dm.fit_grid()
    res.write_parquet(dm.DRIVE / "trait_models.parquet")
    pc = dm.player_correlations()
    pc.write_parquet(dm.DRIVE / "player_correlations.parquet")

    for spec, g in res.group_by("spec", maintain_order=True):
        print(f"{spec[0]:8s} fits={g.height}  p<.05: {(g['p'] < 0.05).sum()}  "
              f"min q={g['q'].min():.2f}  favorable={g['favorable'].mean():.0%}  "
              f"median |effect|={g['effect'].abs().median():.2f}")
    sig = pc.filter((pl.col("lo") > 0) | (pl.col("hi") < 0))
    print(f"within-position correlations: {pc.height}, CI excludes 0: {sig.height}, "
          f"max |r| = {pc['r'].abs().max():.2f}")


if __name__ == "__main__":
    main()
