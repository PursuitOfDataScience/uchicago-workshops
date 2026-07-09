---
description: Write a plotting script for one column of the cleaned dataset
argument-hint: <column name, e.g. water_temp_c>
allowed-tools: Read, Write(figures/**), Edit(figures/**)
---
Write a small, self-contained Python plotting script that visualises the column
`$ARGUMENTS` over time, with one line per site, using the cleaned dataset at
`data/clean/lakewatch_clean.csv`.

Requirements:
- Read the cleaned CSV with the standard-library `csv` module (do not assume pandas).
- Use `matplotlib`; plot `date` on the x-axis and `$ARGUMENTS` on the y-axis, one series per `site`.
- Label the axes (include the unit from `docs/data_dictionary.md`), add a legend and a title.
- Save the figure to `figures/$ARGUMENTS.png` and also `plt.show()`.
- Put the script at `figures/plot_$ARGUMENTS.py`. Do not run it — I will run it where matplotlib is installed.

If `data/clean/lakewatch_clean.csv` does not exist yet, say so and stop (finish Task 2 first).
