# Valid units & ranges for LakeWatch columns

Use these to decide whether a value is plausible. Anything outside the range is a
recording error and should be flagged (not silently kept).

| column | unit | valid range | notes |
|---|---|---|---|
| `water_temp_c` | °C | 0 – 40 | surface water; freezing to very warm shallow lake |
| `dissolved_oxygen_mg_l` | mg/L | 0 – 20 | never negative; ~14 is saturation in cold water |
| `ph` | unitless | 0 – 14 | most lakes sit 6.5 – 8.5 |
| `turbidity_ntu` | NTU | 0 – 1000 | never negative; high after storms |

Tip: a Fahrenheit value stored in a Celsius column shows up as a temperature in the
50–90 range — plausible as °F, impossible as °C for these lakes.
