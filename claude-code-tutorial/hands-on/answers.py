#!/usr/bin/env python3
"""answers.py — reference answers for the penguins hands-on lab.

For each Part 2 task card (Tasks 9–16) you'll find the prompt (as a comment), the Python code that
produces the answer, and the expected output commented out beneath it. Read it to check Claude's
numbers, or run the whole file to reproduce them:

    python answers.py

The data is read live from the URL in CLAUDE.md, so you need internet on the node. Requires
pandas (and matplotlib for Task 14) — both are in the workshop's `AI` environment.

This is the answer key. While you are doing a task, let Claude Code do the analysis — do not paste
from here, and do not ask Claude to read this file.
"""
import json

import matplotlib
matplotlib.use("Agg")            # headless cluster: save figures, never open a window
import matplotlib.pyplot as plt
import pandas as pd

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(URL)

MEAS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


# ============================================================================
# Task 9 · First look
#   "...overview: rows, columns and types, the three species and their counts,
#    and how many rows have any missing values."
# ============================================================================
print("shape:", df.shape)
print(df.dtypes.to_string())
print(df["species"].value_counts().to_string())
print("rows with any missing value:", int(df.isna().any(axis=1).sum()))
# shape: (344, 7)
# species               object
# island                object
# bill_length_mm       float64
# bill_depth_mm        float64
# flipper_length_mm    float64
# body_mass_g          float64
# sex                   object
# species
# Adelie       152
# Gentoo       124
# Chinstrap     68
# rows with any missing value: 11


# ============================================================================
# Task 10 · Summary by species
#   "For each species, count + mean and sd of body_mass_g and flipper_length_mm."
# ============================================================================
summary = (df.groupby("species")[["body_mass_g", "flipper_length_mm"]]
             .agg(["count", "mean", "std"]).round(1))
print(summary.to_string())
#           body_mass_g                flipper_length_mm
#                 count    mean    std             count   mean  std
# species
# Adelie            151  3700.7  458.6               151  190.0  6.5
# Chinstrap          68  3733.1  384.3                68  195.8  7.1
# Gentoo            123  5076.0  504.1               123  217.2  6.5
#
# Missing values are skipped by mean()/std(), so the counts are 151/68/123 (not 152/68/124).
# Gentoo is clearly the heaviest.


# ============================================================================
# Task 11 · The heaviest species
#   "Which species is heaviest, by how much vs the lightest, and is the gap large
#    relative to the within-species spread? Numbers + a one-sentence verdict."
# ============================================================================
means = df.groupby("species")["body_mass_g"].mean().round(1)
sds = df.groupby("species")["body_mass_g"].std().round(1)
print("means:", means.to_dict())
print("gap heaviest - lightest:", round(means.max() - means.min(), 1))
print("within-species sd:", sds.to_dict())
# means: {'Adelie': 3700.7, 'Chinstrap': 3733.1, 'Gentoo': 5076.0}
# gap heaviest - lightest: 1375.3
# within-species sd: {'Adelie': 458.6, 'Chinstrap': 384.3, 'Gentoo': 504.1}
#
# Verdict: Gentoo (~5076 g) is substantially heavier than Adelie (~3701 g) — a ~1375 g gap,
# roughly three within-species standard deviations, so the difference is large and real.


# ============================================================================
# Task 12 · Flipper length vs body mass
#   "Correlation overall, then within each species. Does it differ, and why?"
# ============================================================================
overall = df["flipper_length_mm"].corr(df["body_mass_g"])
within = df.groupby("species").apply(
    lambda g: g["flipper_length_mm"].corr(g["body_mass_g"]), include_groups=False)
print("overall corr:", round(overall, 3))
print("within-species corr:", {k: round(v, 3) for k, v in within.items()})
# overall corr: 0.871
# within-species corr: {'Adelie': 0.468, 'Chinstrap': 0.642, 'Gentoo': 0.703}
#
# The strong overall 0.87 is inflated because species differ in BOTH flipper length and mass;
# mixing them exaggerates the link. Within a species the relationship is real but weaker (~0.5-0.7).
# The honest picture is the within-group one.


# ============================================================================
# Task 13 · Data quality
#   "Find every row with a missing value and flag implausible measurements —
#    but do not change or drop any rows."
# ============================================================================
na_rows = df[df.isna().any(axis=1)]
all_meas_missing = int(df[MEAS].isna().all(axis=1).sum())
only_sex_missing = int((df["sex"].isna() & df[MEAS].notna().all(axis=1)).sum())
print("rows with any missing value:", len(na_rows), "-> index", list(na_rows.index))
print("missing every measurement:", all_meas_missing)
print("missing only sex:", only_sex_missing)
print(df[MEAS].agg(["min", "max"]).round(1).to_string())   # implausibility check
# rows with any missing value: 11 -> index [3, 8, 9, 10, 11, 47, 246, 286, 324, 336, 339]
# missing every measurement: 2
# missing only sex: 9
#      bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
# min            32.1           13.1              172.0       2700.0
# max            59.6           21.5              231.0       6300.0
#
# Every min/max is a physically sensible penguin measurement (no negatives, no absurd values),
# so nothing is implausible. Conservative handling of the missing rows: exclude the 2 empty rows
# from measurement analyses, keep the 9 (they still have all measurements), and DOCUMENT the
# choice — never silently drop.


# ============================================================================
# Task 14 · Make a figure
#   "Save a boxplot of body_mass_g by species to body_mass_by_species.png; describe it."
# ============================================================================
ax = df.boxplot(column="body_mass_g", by="species", grid=False, figsize=(6, 4))
ax.set_title("Body mass by species")
ax.set_xlabel("species")
ax.set_ylabel("body mass (g)")
plt.suptitle("")
plt.tight_layout()
plt.savefig("body_mass_by_species.png", dpi=150)
plt.close()
print("saved body_mass_by_species.png")
# saved body_mass_by_species.png
# The Gentoo box sits well above Adelie and Chinstrap (median ~5000 g vs ~3700 g each);
# Adelie and Chinstrap overlap heavily.


# ============================================================================
# Task 15 · Write it up  (no computation — this is a reference paragraph)
# ============================================================================
# "We analysed body measurements from 344 penguins across three species (Adelie n=152,
#  Gentoo n=124, Chinstrap n=68). Gentoo penguins were substantially heavier (mean body mass
#  5076 g) than Adelie (3701 g) and Chinstrap (3733 g); the ~1375 g gap between Gentoo and the
#  others is roughly three times the within-species standard deviation (~380-500 g). Flipper
#  length and body mass were strongly correlated overall (r = 0.87), but the association was more
#  modest within each species (r = 0.47-0.70), indicating that much of the overall correlation
#  reflects between-species size differences rather than a within-species scaling law. Eleven
#  rows contained missing values (two lacked all measurements and nine lacked only sex); these
#  were retained and flagged rather than dropped."


# ============================================================================
# Task 16 (bonus) · Headless equivalent
#   `claude -p "...mean body_mass_g per species, as compact JSON" --output-format json --model haiku`
# ============================================================================
print(json.dumps({k: round(v) for k, v in df.groupby("species")["body_mass_g"].mean().items()}))
# {"Adelie": 3701, "Chinstrap": 3733, "Gentoo": 5076}
