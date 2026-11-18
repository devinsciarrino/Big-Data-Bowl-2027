# CLAUDE.md

NFL Big Data Bowl 2027 entry. The prompt: link 10 Hz Combine tracking data to future NFL
regular-season performance. See PLAN.md for rules, rubric, timeline and ideas.

## Commands
- `uv sync` · `./scripts/download_data.sh` · `uv run jupyter lab`
- `uv run python scripts/build_pulls.py` → `data/processed/pulls/*.parquet` (~seconds)
- `uv run pytest -q tests` · `uv run ruff check .`

## Layout
- `src/bdb27/data.py`: `load(name, lazy=False)` reads `data/raw/<name>.csv`, caching to
  `data/processed/<name>.parquet`; `game_tracking()` concatenates the three seasons lazily.
  Use Polars lazy frames for game tracking (multi-GB).
- `src/bdb27/pulls.py`: the project core. Combine pull-drill reps and in-game OL run snaps,
  measured by ONE shared `rep_metrics()` in a path-intrinsic frame (u = lateral, v = upfield,
  t from movement onset) so drill and game numbers are directly comparable. Game snaps use a
  field frame (origin = spot at snap, +x upfield, +y = offense's left). `flag_pulls()` labels
  lateral snaps `pull` vs `lateral_flow`; its thresholds are provisional.
  Tests in `tests/test_pulls.py` use synthetic paths (no data needed).
- `notebooks/`: numbered exploration notebooks (`01_eda.ipynb`, …).
- `figures/`: exported figures for the writeup. `writeup/`: writeup drafts.

## Constraints
- Never commit data (CC BY-NC 4.0). The final notebook must run on Kaggle, so keep the
  analysis reproducible from the raw CSVs.
- Writeup: ≤ 2,000 words, ≤ 9 figures/tables. Must use the tracking data.
- Data quirks: on BALL rows in `combine_tracking`, `nfl_id` is the athlete doing the drill;
  `lined_up_position = NA` in `player_play` means a play killed by a pre-snap penalty;
  13 of 510 players never played.
