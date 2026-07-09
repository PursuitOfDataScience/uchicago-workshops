# LakeWatch

Weekly water-quality monitoring for three lake sites — **alpha**, **bravo**, and
**charlie**. A sensor "sonde" at each site records water temperature, dissolved
oxygen, pH, and turbidity. Raw exports land in `data/raw/`; analysis helpers live
in `src/`; field logs live in `notes/`.

This repository was handed over mid-project and is deliberately unfinished — that is
what you will work on. See the workshop task cards in `../tasks/`.

## Known state (inherited)
- `data/raw/` holds one export per site. Each site's export uses **different column
  names, units, and date formats**, and a few readings are missing or clearly wrong.
- There is **no cleaned, merged dataset yet** (`data/clean/` is empty).
- `src/waterquality.py` has a **failing test** — something is off in the analysis.
- `docs/data_dictionary.md` is **incomplete**.

## Running the tests
```bash
python -m pytest -q      # from this directory
```
