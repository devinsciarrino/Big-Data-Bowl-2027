# BDB 2027 Plan

## The competition (as of 2026-11-17)

- **Prompt:** build an analytical framework linking **10 Hz NFL Combine tracking data** to
  **future regular-season NFL performance**. Narrow focus is explicitly fine: one movement
  trait, one position group, or one drill.
- **Positions in the data:** WR, TE, DB, OL, DL. 510 rookies from the 2023–2025 Combines;
  6,310 tracked drill attempts; 1,002 games of in-game tracking (2023–2025).
- **Files (9 CSVs, ~2.3 GB, CC BY-NC 4.0):** `players`, `combine_results`, `combine_tracking`,
  `player_career_successes`, `player_play`, `games`, `game_tracking_{2023,2024,2025}`.
- **Hard rule:** a writeup that doesn't use the player tracking data is not scored.
- **Deadline:** **Wed 2027-01-06, 23:59 UTC** (3:59 PM Pacific). One submission per team; a draft
  is not judged; you must click Submit.
- **Deliverable:** one Kaggle Writeup (≤ 2,000 words, ≤ 9 figures/tables; appendix, captions and
  labels don't count) plus an attached **public** Kaggle Notebook. Optional video (no NFL/Combine
  film).
- **Scoring:** two pass/fail gates (explicit Combine→NFL linkage, null result OK; complete
  writeup with visuals + notebook), then Football 30% · Data Science 30% · Writeup 20% ·
  Data Viz 20%.
- **Prizes:** $100k pool. Open and student tracks each have 3 × $10k finalists and 3 × $5k
  runners-up; finalists present at the 2027 Combine in Indianapolis (in person) for a $10k
  Grand Prize.

### Open question: which track?
The NFL press release describes an "Undergraduate/Graduate Track" but the Kaggle page reportedly
calls it a University track limited to undergraduates, with grad students in Open. **Confirm on
the Kaggle Rules tab before submitting** (and never select the Grand Prize track).

## Timeline (~7 weeks)

| Week | Dates | Goal |
|---|---|---|
| 1 | Nov 17–23 | Data download, schema notes, EDA of every file, reproduce basic Combine times from tracking |
| 2 | Nov 24–30 | Feature library: per-drill movement metrics from combine tracking; in-game metrics per player from game tracking. Shortlist 2 ideas (Thanksgiving week, budget light) |
| 3 | Dec 1–7 | **Pick one idea.** First end-to-end result: Combine metric → in-game outcome, with baseline vs. stopwatch times |
| 4 | Dec 8–14 | Modeling rigor: hierarchical / position-adjusted models, uncertainty, out-of-sample check by draft class |
| 5 | Dec 15–21 | Figures (≤ 9), writeup draft 1, public notebook cleaned |
| 6 | Dec 22–28 | Holiday buffer; outside reader feedback |
| 7 | Dec 29–Jan 5 | Final edits, word count, links work logged-out, **submit by Jan 5** (a day early) |

## What wins (from past BDB finalists)

- A **new, named metric** that a coach or scout would actually use, grounded in what the
  tracking data can see that a stopwatch can't.
- Football first: the "so what" for evaluators and player development (the prompt calls out
  spotting hidden talent and position-specific development protocols).
- Honest validation: show the new metric adds signal **beyond** the official Combine times
  (40, shuttle, 3-cone), and how stable it is.
- Clean, few, annotated visuals; a short writeup that reads like a scouting report.

## Chosen direction: offensive line, run blocking vs. pass protection

**Question:** which Combine movement qualities predict NFL *pass protection*, which predict *run
blocking*, and are they the same qualities? (LT, LG, C, RG, RT.)

### What the data gives us (checked 2026-11-17)
- **120 Combine OL**, all of whom appear in games: 50 T, 51 G, 20 C (2023: 37, 2024: 46,
  2025: 38). 88 have ≥ 200 in-game snaps; median ≈ 540.
- **OL Combine drills tracked at 10 Hz** (~120 players each unless noted):
  `FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE`, `PASS_RUSH_DROP` (2 reps), `PASS_PRO_MIRROR_DRILL`
  (spelled two ways, `PASS_PRO-MIRROR_DRILL` too; merge, 121 total), `SCREEN_DRILL`,
  `OL_PULL_DRILL_FOLD_BLOCK_RIGHT`, `OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT` (98),
  `OL_PULL_DRILL_LONG_PULL_RIGHT` (93). Plus forty/10-yd split (most), shuttle and 3-cone
  (fewer), and measurables (arm length, hand size, wingspan, weight).
  89 linemen have all 7 drill names, 32 are missing 1–2.
- **Pass protection labels are per-player** in `player_play` (~60k OL pass-block snaps):
  `pressure_allowed` (T 9.6%, G 7.7%, C 5.4%), `sack_allowed`, `peak_pressure_probability_allowed`
  (continuous, best target), `pass_rushers_encountered`, `dropback_duration`.
- **Run blocking has no per-player label** (~43k OL run snaps). Options: team `yards_gained` /
  `expected_points_added` on runs, run direction from `play_description` ("right guard",
  "left end") to credit the lineman at the point of attack, and the lineman's own tracking
  (get-off at the snap, drive displacement, pull speed and path, reaching the second level).
- **Game tracking holds only the 510 Combine players, not all 22.** There are no defenders and no
  ball, so we can't measure blocks against a defender directly; we measure the lineman's own
  movement plus play outcomes.
- **Left vs. right isn't labeled:** `lined_up_position` is just T / G / C. Side can be inferred
  from tracking (lateral position relative to a second tracked rookie lineman on the same play:
  21.8k such plays) or from the player's typical alignment. To verify in EDA.

### Focus (decided 2026-11-17): drill-to-game pull comparison
The headline is the **Combine pull drills vs. the same lineman's pulls in NFL games**,
measured with the same tools. Pass protection and broader run blocking become supporting
context, not the core.

Feasibility checks (scratch analysis, 2026-11-17):
- **Combine pulls are clean and consistent**: 3 drills (fold block R, long pull R, deep/short
  pull L), 93–119 linemen each, ~4–5 s at 10 Hz. Paths are tight within a drill; some reps are
  mirrored (drill run facing the other way), so normalize the orientation.
- **Game pulls can be detected**: after the snap, normalize the play direction and measure the
  lineman's displacement. A rough rule (lateral > 5 yd at 2.0 s, forward < 0.3 yd at 1.0 s and
  < 1.0 yd at 1.5 s) picks out flat lateral runs at the line that turn upfield, which look like
  the Combine drill. Pullers also move toward the side where the run goes (e.g. 534 of 535 on
  "left tackle" runs moved left).
- **Sample:** ~3.1k flagged pulls on runs. 65 linemen have ≥ 10, 41 have ≥ 20, all 65 with
  pull-drill tracking (27 T, 26 G, 12 C).
- **The rule is too loose**: it flags 2–4% of pass snaps (screens, sprint-outs) and ~10% of
  centers' runs, which is too high, so it's catching some wide outside-zone flow. Next, refine it
  with speed sustained through the pull, path shape, and/or similarity to the Combine drill paths.

Side detection (2026-11-17, `bdb27.alignment`): 94.7% of OL snaps get a spot (LT/LG/C/RG/RT).
Fit RMSE median 0.085 yd; game-level sides 97% unanimous; leave-one-game-out season side 87.5%.
It shows the drop-back rule mixes schemes: only 30–40% of tackle "pulls" cross the formation
(likely playside zone reach steps), while 77% of RT "lateral_flow" snaps cross (backside zone
flow). **Next: cluster resampled 0–2.5 s paths (with side/crossing as features) to separate gap
pulls, zone flow and reach steps, and check clusters against hand-labelled snaps.**

### Metrics to compare (same definition in drill and game)
Get-off (time to first movement / first 1 yd), lateral acceleration and top lateral speed,
time to 5 yd lateral, depth lost (how far he drops behind the line), path tightness (distance
run ÷ straight-line distance), turn-up sharpness (speed kept through the turn upfield), total
time to the point of attack.

### Approach
1. **Combine movement features** per lineman per drill: get-off/first-step time, peak
   acceleration, lateral speed and how well it's held in the slide/mirror, change-of-direction
   cost (speed lost at each redirect), pull path efficiency (distance run vs. straight line,
   time to depth), hip/footwork proxies from `dir` vs. movement direction.
2. **In-game outcomes** per lineman: pass pro = pressure/peak-pressure rate adjusted for dropback
   time and opponent; run = point-of-attack EPA/yards plus tracking-derived movement on runs.
   The tracking also lets us *mirror* Combine drills in games: a pull on a run play vs. the pull
   drill; the pass set vs. the slide/drop drills.
3. **Linkage models**: position-pooled (T vs. interior, partial pooling with C), controlling for
   draft pick and snaps; test whether tracking features add signal beyond stopwatch times and
   measurables. Report effect sizes with intervals.
4. **Story**: a "pass-pro profile" and a "run-block profile" from Combine movement, where they
   overlap and where they diverge, and what that means for drafting a tackle vs. a guard.

### Watch-outs specific to OL
- n is small per spot (20 centers); 5-way splits (LT/LG/C/RG/RT) are for description, not
  separate models.
- Pressure is shared: QB hold time, scheme and the defender all drive it.
- 2025 class has at most one season of games.

## Risks
- **Small n per position** → favor partial pooling across positions, effect sizes with
  intervals, and per-player visuals over leaderboards from noisy models.
- **Selection bias:** playing time depends on draft position, which depends on Combine results.
  Control for draft slot / snaps, and say so.
- **Rookie-heavy outcome window** (2025 class has one season at most).
