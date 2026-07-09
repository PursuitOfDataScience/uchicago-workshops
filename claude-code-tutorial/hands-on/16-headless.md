# Bonus · Run it headless

Run this in a terminal (in this folder):

```bash
claude -p "Read the penguins CSV at https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv and reply with only the mean body_mass_g for each species, as compact JSON." --output-format json --model haiku --dangerously-skip-permissions
```

In headless (`-p`) mode there is no one to answer the approval prompts, so any tool Claude needs
(here, running Python to read the CSV) is **auto-denied** and the command stalls asking for
approval. `--dangerously-skip-permissions` skips those prompts so the one-shot command can run to
completion. (Only pass it for a run you trust — like this read-only fetch.)
