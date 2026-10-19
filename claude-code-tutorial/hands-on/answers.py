#!/usr/bin/env python3
"""answers.py: the reference key for the hands-on exercise.

This file is for you, not for Claude. Claude is blocked from reading it by a deny rule in
.claude/settings.json and told to leave it alone in CLAUDE.md; keep both in place, or the
verification steps will look like they pass while proving nothing.

Each section below names the step it checks, holds the code that produces the answer, and shows
the expected output in comments. Run the whole file from this folder with the exercise's Python:

    python answers.py

It reads data/raw/penguins.csv when that copy exists (step 1, 1-setup.md, makes it) and the URL
otherwise. Needs pandas and numpy only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
LOCAL = Path(__file__).resolve().parent / "data" / "raw" / "penguins.csv"
df = pd.read_csv(LOCAL if LOCAL.exists() else URL)

MEAS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def within_species_r(frame):
    return {sp: round(float(g["flipper_length_mm"].corr(g["body_mass_g"])), 3)
            for sp, g in frame.groupby("species")}


# ============================================================================
# 1-setup · the download
# ============================================================================
print("shape:", df.shape)
# shape: (344, 7)


# ============================================================================
# 2-explore · a first look
# ============================================================================
print(df["species"].value_counts().to_dict())
print(df["island"].value_counts().to_dict())
print(pd.crosstab(df["species"], df["island"]).to_string())
print("missing per column:", df.isna().sum().to_dict())
print("rows with any missing value:", int(df.isna().any(axis=1).sum()))
print("missing every measurement:", int(df[MEAS].isna().all(axis=1).sum()))
print("missing only sex:", int((df["sex"].isna() & df[MEAS].notna().all(axis=1)).sum()))
print(df[MEAS].agg(["min", "max"]).to_string())
# {'Adelie': 152, 'Gentoo': 124, 'Chinstrap': 68}
# {'Biscoe': 168, 'Dream': 124, 'Torgersen': 52}
# island     Biscoe  Dream  Torgersen
# species
# Adelie         44     56         52
# Chinstrap       0     68          0
# Gentoo        124      0          0
# missing per column: {'species': 0, 'island': 0, 'bill_length_mm': 2, 'bill_depth_mm': 2,
#                      'flipper_length_mm': 2, 'body_mass_g': 2, 'sex': 11}
# rows with any missing value: 11
# missing every measurement: 2
# missing only sex: 9
#      bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
# min            32.1           13.1              172.0       2700.0
# max            59.6           21.5              231.0       6300.0
#
# Worth a closer look: Gentoo live only on Biscoe and Chinstrap only on Dream, so species and
# island are tangled together; sex is recorded in capitals (MALE / FEMALE). Every min and max is
# a plausible penguin measurement, so nothing needs correcting. The right handling of the 11 rows
# is to report them, not drop them.


# ============================================================================
# 3-analysis · plan, then analyze
# ============================================================================
summary = (df.groupby("species")[["body_mass_g", "flipper_length_mm"]]
             .agg(["count", "mean", "std"]).round(1))
print(summary.to_string())
means = df.groupby("species")["body_mass_g"].mean()
print("heaviest:", means.idxmax(), "gap to lightest:", round(means.max() - means.min(), 1))
print("pooled r:", round(df["flipper_length_mm"].corr(df["body_mass_g"]), 3))
print("within-species r:", within_species_r(df))
#           body_mass_g                flipper_length_mm
#                 count    mean    std             count   mean  std
# species
# Adelie            151  3700.7  458.6               151  190.0  6.5
# Chinstrap          68  3733.1  384.3                68  195.8  7.1
# Gentoo            123  5076.0  504.1               123  217.2  6.5
# heaviest: Gentoo gap to lightest: 1375.4
# pooled r: 0.871
# within-species r: {'Adelie': 0.468, 'Chinstrap': 0.642, 'Gentoo': 0.703}
#
# Verdict: Gentoo is heaviest by about 1,375 g, roughly three within-species standard deviations.
# The pooled r = 0.87 is inflated because species differ in both flipper length and mass; within a
# species the link is real but weaker (0.47 to 0.70). Trust the within-species picture.


# ============================================================================
# 4-verify · two independent checks
# ============================================================================
# The means rest on 151 Adelie, 68 Chinstrap, and 123 Gentoo rows (342 in all), not 152/68/124:
# the two rows missing every measurement drop out of each statistic. A recount that says the
# means use 152/68/124 rows has not checked what it averaged. The result to be most skeptical
# of is the pooled r = 0.87, for the reason in step 3.


# ============================================================================
# 5-finish-line · a finished script with /goal
# ============================================================================
# analyze.py should print the numbers from step 3: counts 151 / 68 / 123, mean body
# mass 3700.7 / 3733.1 / 5076.0 g, pooled r 0.871, within-species r 0.468 / 0.642 / 0.703.
# results/summary.csv should hold the same per-species rows. Its layout is Claude's choice.


# ============================================================================
# extra · bootstrap intervals for the within-species correlations
# ============================================================================
rng = np.random.default_rng(2026)
B = 10_000
for sp, g in df.dropna(subset=["flipper_length_mm", "body_mass_g"]).groupby("species"):
    x, y = g["flipper_length_mm"].to_numpy(), g["body_mass_g"].to_numpy()
    idx = rng.integers(0, len(x), size=(B, len(x)))
    xs, ys = x[idx], y[idx]
    xm, ym = xs - xs.mean(axis=1, keepdims=True), ys - ys.mean(axis=1, keepdims=True)
    r = (xm * ym).sum(axis=1) / np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
    lo, hi = np.percentile(r, [2.5, 97.5])
    print(f"{sp}: r = {np.corrcoef(x, y)[0, 1]:.3f}, 95% CI [{lo:.2f}, {hi:.2f}]")
# Adelie: r = 0.468, 95% CI [0.36, 0.57]
# Chinstrap: r = 0.642, 95% CI [0.51, 0.75]
# Gentoo: r = 0.703, 95% CI [0.61, 0.78]
#
# A bootstrap is random, so yours will differ in the second decimal. Adelie overlaps Chinstrap and
# Chinstrap overlaps Gentoo; Adelie and Gentoo do not overlap. (Overlap is a rough guide, not a
# formal test of a difference.)


# ============================================================================
# extra · the means as compact JSON, as a headless run would reply
# ============================================================================
print(json.dumps({k: round(v) for k, v in means.items()}))
# {"Adelie": 3701, "Chinstrap": 3733, "Gentoo": 5076}


# ============================================================================
# 6-skill · /check-numbers, and 9-batch-job · the batch run
# ============================================================================
# Both should reproduce the step 3 numbers exactly. A mismatch almost always means a
# different input (a stale or edited copy of the data) rather than a different answer.
