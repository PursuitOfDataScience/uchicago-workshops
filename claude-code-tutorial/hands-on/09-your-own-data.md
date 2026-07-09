# Take-home · Your own data

The real payoff is doing this with data you care about. The whole pattern transfers with one
change: **swap the URL.**

## Try it
Find any CSV your work uses that is reachable by URL — a published dataset, a file in a GitHub
repo (use the **raw** link), an export from a data portal — and start a session in a fresh
folder with a short `CLAUDE.md` naming it:

```markdown
# CLAUDE.md
Dataset: https://example.org/path/to/your.csv
Columns: ... (list them). Load it with pandas; show the code; never drop rows silently.
```

Then drive it the same way:

> Load the dataset in CLAUDE.md, give me an overview, then tell me [your question].

## A few good first prompts
- *"Summarize each numeric column, and flag anything that looks wrong."*
- *"Group by [category] and compare [measure] across groups, with the numbers."*
- *"Plot [x] against [y], colored by [group], and save it — then describe the trend."*
- *"Draft a short Results paragraph from what you found."*

## Keep in mind
- On a cluster, run on a node with **internet egress** so the agent can fetch the URL.
- If the data is **unpublished, PHI, or export-controlled**, do not send it without
  institutional approval — the same rule as any tool that transmits data.
- **Always check its work.** Ask to see the code, and sanity-check a number before you cite it.

## Where to go next
- Full documentation: **code.claude.com/docs**
- The lecture deck (`../claude-code-tutorial.pptx`) has the concepts behind every task —
  project memory, permissions, skills, scaling, and safety.
- RCC support: user guide at **docs.rcc.uchicago.edu**, the help desk, and office hours.
