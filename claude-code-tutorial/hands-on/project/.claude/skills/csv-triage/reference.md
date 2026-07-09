# CSV data-quality checklist

Work through each category. A finding in any one is worth reporting.

## Headers & units
- Inconsistent naming across files (`Temp (F)` vs `water_temp_c` vs `TEMPERATURE_C`).
- Units hidden in the header text (`(F)`, `_c`, `_mg_l`) — flag any mismatch between files.
- Stray whitespace or capitalisation differences in header names.

## Missing values
- Empty cells, or sentinel strings like `NA`, `N/A`, `null`, `-999`.
- Whole columns that are mostly blank.

## Impossible or out-of-range values
- pH outside 0–14.
- Negative concentrations (dissolved oxygen, turbidity) — physically impossible.
- Temperatures far outside a plausible environmental range.
- Values that are off by a unit-conversion factor (e.g. Fahrenheit stored in a Celsius column).

## Structural issues
- Duplicate rows (same key, e.g. same site + date).
- Inconsistent date formats (`2023-06-05` vs `6/7/2023`).
- Extra spaces around delimited values (` 17.9 ` instead of `17.9`).
