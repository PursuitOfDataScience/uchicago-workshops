# LakeWatch data dictionary

This describes the **cleaned, merged** dataset that the cleaning step produces at
`data/clean/lakewatch_clean.csv`. Several rows below are unfinished — Task 4 completes them.

| column | unit | description | valid range |
|---|---|---|---|
| `site` | — | monitoring site: `alpha`, `bravo`, or `charlie` | — |
| `date` | ISO 8601 (`YYYY-MM-DD`) | sampling date | — |
| `water_temp_c` | °C | surface water temperature | _TODO_ |
| `dissolved_oxygen_mg_l` | mg/L | dissolved oxygen concentration | _TODO_ |
| `ph` | unitless | acidity / alkalinity | _TODO_ |
| `turbidity_ntu` | _TODO_ | _TODO_ | _TODO_ |

## Notes
- The raw exports (`data/raw/`) are **not** documented here on purpose — each site's
  export uses its own column names and units. Cleaning normalises them to the schema above.
- Impossible readings (for example a pH outside 0–14, or a negative dissolved-oxygen
  value) are recording errors and are removed during cleaning.
