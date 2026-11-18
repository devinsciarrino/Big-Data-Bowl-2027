# CLAUDE.md

NFL Big Data Bowl 2027 entry. The prompt: link 10 Hz Combine tracking data to future NFL
regular-season performance. See PLAN.md for rules, rubric, timeline and ideas.

## Commands
- `uv sync` · `./scripts/download_data.sh` · `uv run jupyter lab`
- `uv run python scripts/build_pulls.py` → `data/processed/pulls/*.parquet` (~seconds; run first,
  it writes `ol_sides.parquet`)
- `uv run python scripts/build_drive.py` → `data/processed/drive/*.parquet` (~seconds)
- `uv run python scripts/analyze_drive.py` → trait models + correlations (~40 s)
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
- `src/bdb27/alignment.py`: infers LT/LG/C/RG/RT. `player_play` only says T/G/C and game
  tracking has no ball, so `solve_slots()` fits slot assignments (1.55 yd spacing) on plays
  with 2+ tracked rookie linemen; `player_sides()` propagates the majority side per
  player-game → season → career (`side_source`). Game-level sides are ~97% self-consistent;
  season/career fallbacks ~87% (rookies switch sides between games).
- `src/bdb27/drive.py`: **current focus.** Drive (downhill) run blocks. No Combine OL drill
  matches a drive block (contact comes within ~1 yd), so this links Combine *traits* (first
  0.5 s of the tracked 40, momentum = weight x speed, measurables) to game *outcomes* on drive
  snaps (first step within 60 deg of upfield, |lateral| <= 1.5 yd through 1.5 s): get-off,
  first-step speed, upfield yd at 1.0/1.5/2.5 s, contact + drive after contact.
  Pulls (`pulls.py`) are parked until the pull-vs-zone labeling question is settled.
- `src/bdb27/drive_models.py`: Combine trait → game drive outcome mixed models. Always control
  T/G/C position (not just T vs. interior). Result so far is null (see PLAN.md, Result 1).
- Missing metrics are null, not NaN (`fill_nan(None)`), so flags derived from them stay null.
- `notebooks/`: numbered exploration notebooks.
- `figures/`: exported figures for the writeup. `writeup/`: writeup drafts.

## Constraints
- Never commit data (CC BY-NC 4.0). The final notebook must run on Kaggle, so keep the
  analysis reproducible from the raw CSVs.
- Writeup: ≤ 2,000 words, ≤ 9 figures/tables. Must use the tracking data.
- Data quirks: on BALL rows in `combine_tracking`, `nfl_id` is the athlete doing the drill;
  `lined_up_position = NA` in `player_play` means a play killed by a pre-snap penalty;
  13 of 510 players never played.
