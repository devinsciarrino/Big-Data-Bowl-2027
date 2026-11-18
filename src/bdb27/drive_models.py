"""Which Combine traits predict game drive-block outcomes?

Snap-level linear mixed models (statsmodels MixedLM), one per (outcome, trait):

    outcome ~ trait_z + position(T/G/C) + down + short_yardage + formation + [extra] + (1 | lineman)

- ``trait_z`` is the Combine trait standardized across linemen, so coefficients are "outcome
  units per 1 SD of the trait". The random intercept per lineman handles unequal snap counts
  (a lineman with 400 snaps doesn't count 400x) and repeated snaps of the same player.
- Model specs: ``base``; ``+ten_yd`` adds the stopwatch 10-yd split (does the tracked trait add
  signal beyond the stopwatch?); ``+draft`` adds log draft pick (beyond what teams already
  priced in).
- p-values get Benjamini-Hochberg FDR within each spec.
- Position is the snap's T/G/C alignment, not just tackle vs. interior: T > G > C on arm
  length, weight and first-step speed (centers also snap the ball), so a tackle-only control
  let center-vs-guard differences pose as size effects (arm length ~ first-step speed went
  from r = 0.52 overall to 0.16 within position).

Small-n caveat: ~65 linemen have both a tracked 40 and >= 20 drive snaps, so effects are
estimated mostly between players. Read intervals, not stars.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from bdb27.data import PROCESSED

DRIVE = PROCESSED / "drive"

OUTCOMES = {  # name: (label, higher_is_better)
    "getoff_s": ("Get-off (s, snap to moving)", False),
    "first_step_speed": ("First-step speed (yd/s)", True),
    "fwd_10": ("Upfield at 1.0 s (yd)", True),
    "fwd_25": ("Upfield at 2.5 s (yd)", True),
    "drive_after_contact": ("Drive after contact (yd)", True),
}
TRAITS = {  # name: (label, tracked?); timed traits are "lower is better" (see LOWER_IS_BETTER)
    "speed_05": ("40 start: speed at 0.5 s", True),
    "peak_accel_05": ("40 start: peak accel, first 0.5 s", True),
    "dist_05": ("40 start: distance at 0.5 s", True),
    "time_to_2yd": ("40 start: time to 2 yd", True),
    "momentum_05": ("40 start: momentum (wt x speed)", True),
    "force_05": ("40 start: force (wt x peak accel)", True),
    "ten_yd_split": ("10-yd split (stopwatch)", False),
    "forty": ("40 time (stopwatch)", False),
    "combine_weight": ("Weight", False),
    "arm_length": ("Arm length", False),
    "broad_jump": ("Broad jump", False),
    "vertical": ("Vertical", False),
    "bench_reps": ("Bench reps", False),
}
LOWER_IS_BETTER = {"time_to_2yd", "ten_yd_split", "forty"}
SPECS = {"base": "", "+ten_yd": " + ten_yd_z", "+draft": " + log_pick_z"}
MIN_SNAPS = 20


def model_frame() -> pd.DataFrame:
    """Drive snaps joined to per-lineman Combine traits (standardized), as pandas for statsmodels."""
    snaps = pl.read_parquet(DRIVE / "drive_snaps.parquet")
    players = pl.read_parquet(DRIVE / "players.parquet")
    keep = snaps.group_by("nfl_id").len().filter(pl.col("len") >= MIN_SNAPS).select("nfl_id")
    players = players.join(keep, on="nfl_id")

    def z(c: str) -> pl.Expr:
        return ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(f"{c}_z")

    players = (players.with_columns(log_pick=pl.col("draft_overall_pick").log())
               .with_columns(*[z(c) for c in TRAITS], z("log_pick").alias("log_pick_z"),
                             z("ten_yd_split").alias("ten_yd_z")))
    df = snaps.join(players.select("nfl_id", "nfl_position", *[f"{c}_z" for c in TRAITS],
                                   "ten_yd_z", "log_pick_z"), on="nfl_id")
    df = df.with_columns(
        pos=pl.col("lined_up_position"),
        down=pl.col("down").cast(pl.Utf8),
        short_yd=(pl.col("yards_to_go") <= 2).cast(pl.Int8),
        formation=pl.col("offense_formation").fill_null("OTHER"),
    )
    return df.to_pandas()


def fit_one(df: pd.DataFrame, outcome: str, trait: str, extra: str = "") -> dict | None:
    cols = [outcome, f"{trait}_z", "nfl_id", "pos", "down", "short_yd", "formation"]
    cols += [c.strip() for c in extra.split("+") if c.strip()]
    d = df[cols].dropna()
    if d["nfl_id"].nunique() < 20:
        return None
    formula = f"{outcome} ~ {trait}_z + C(pos) + C(down) + short_yd + C(formation){extra}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.mixedlm(formula, d, groups=d["nfl_id"]).fit(reml=True, method="lbfgs")
    b, se = res.params[f"{trait}_z"], res.bse[f"{trait}_z"]
    # Scale: SD of lineman-level means (random-intercept SD + this trait's share), i.e. how far
    # one lineman's typical outcome sits from another's. Snap-level SD is mostly play noise.
    sd_player = float(np.sqrt(float(np.asarray(res.cov_re)[0, 0]) + b**2))
    return {"outcome": outcome, "trait": trait, "coef": b, "se": se, "p": res.pvalues[f"{trait}_z"],
            "sd_player": sd_player, "n_players": d["nfl_id"].nunique(), "n_snaps": len(d)}


def fit_grid(df: pd.DataFrame | None = None) -> pl.DataFrame:
    """All (spec, outcome, trait) fits with BH-FDR q-values within each spec."""
    df = model_frame() if df is None else df
    rows = []
    for spec, extra in SPECS.items():
        for o in OUTCOMES:
            for t in TRAITS:
                if spec == "+ten_yd" and t in ("ten_yd_split", "forty"):
                    continue
                r = fit_one(df, o, t, extra)
                if r:
                    rows.append({"spec": spec, **r})
    res = pl.DataFrame(rows)
    out = []
    for spec, g in res.group_by("spec", maintain_order=True):
        q = multipletests(g["p"].to_numpy(), method="fdr_bh")[1]
        out.append(g.with_columns(q=pl.Series(q)))
    res = pl.concat(out)
    better = {o: hib for o, (_, hib) in OUTCOMES.items()}
    # effect = outcome change per 1 SD of trait, in lineman-level outcome SDs (correlation-like).
    # favorable = a better trait value (faster time, more weight/jump/accel) -> better outcome.
    eff = pl.col("coef") / pl.col("sd_player")
    return res.with_columns(
        effect=eff, lo=(pl.col("coef") - 1.96 * pl.col("se")) / pl.col("sd_player"),
        hi=(pl.col("coef") + 1.96 * pl.col("se")) / pl.col("sd_player"),
        favorable=pl.struct("outcome", "trait", "coef").map_elements(
            lambda r: ((r["coef"] > 0) == better[r["outcome"]]) != (r["trait"] in LOWER_IS_BETTER),
            return_dtype=pl.Boolean),
        tracked=pl.col("trait").replace_strict({t: v[1] for t, v in TRAITS.items()}),
    )


def player_correlations(n_boot: int = 2000, seed: int = 0) -> pl.DataFrame:
    """Cross-check: within-position Spearman r between adjusted lineman outcomes and each trait
    (trait centered by the lineman's position), with bootstrap CI."""
    po = player_outcomes_shrunk()
    pt = pl.read_parquet(DRIVE / "players.parquet").select("nfl_id", "nfl_position", *TRAITS)
    pt = pt.with_columns([(pl.col(t) - pl.col(t).mean().over("nfl_position")).alias(t)
                          for t in TRAITS])
    j = po.join(pt, on="nfl_id")
    rng = np.random.default_rng(seed)
    rows = []
    for o in OUTCOMES:
        for t in TRAITS:
            d = j.select(o, t).drop_nulls().to_numpy()
            if len(d) < 20:
                continue
            ranks = pd.DataFrame(d).rank().to_numpy()
            r = np.corrcoef(ranks.T)[0, 1]
            idx = rng.integers(0, len(d), (n_boot, len(d)))
            boots = [np.corrcoef(pd.DataFrame(d[i]).rank().to_numpy().T)[0, 1] for i in idx]
            lo, hi = np.nanquantile(boots, [0.025, 0.975])
            rows.append({"outcome": o, "trait": t, "r": r, "lo": lo, "hi": hi, "n": len(d)})
    return pl.DataFrame(rows)


def player_outcomes_shrunk(df: pd.DataFrame | None = None) -> pl.DataFrame:
    """Per-lineman, position- and context-adjusted outcomes (the random intercepts).

    Each value is how far the lineman sits from an average lineman at his position in the same
    context, partially pooled toward 0 for low-snap players. Raw means are noisy and carry
    position differences.
    """
    df = model_frame() if df is None else df
    rows: dict[int, dict] = {}
    for o in OUTCOMES:
        d = df[[o, "nfl_id", "pos", "down", "short_yd", "formation"]].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.mixedlm(f"{o} ~ C(pos) + C(down) + short_yd + C(formation)", d,
                              groups=d["nfl_id"]).fit(reml=True, method="lbfgs")
        for pid, re in res.random_effects.items():
            rows.setdefault(int(pid), {"nfl_id": int(pid)})[o] = float(np.asarray(re)[0])
    return pl.DataFrame(list(rows.values()))
