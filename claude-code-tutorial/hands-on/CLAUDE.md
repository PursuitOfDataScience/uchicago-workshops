# Palmer Penguins — project notes for Claude Code

Claude Code loads this file automatically at the start of a session in this folder. It records
the facts about this project.

## What this is
A small data-analysis project on the **Palmer Penguins** dataset: 344 field measurements of
penguins from three islands near Palmer Station, Antarctica.

## The dataset
Hosted online — nothing to download:

    https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv

Columns: `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`,
`body_mass_g`, `sex`. Some rows have missing values. This environment has internet access and
pandas, so the file can be read directly from the URL.

## Working style
At the start of each task, first **restate it in one short line** — a plain-language summary of
what the prompt is asking — so the workshop audience can see which task this is. Then carry it out.
