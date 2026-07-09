#!/usr/bin/env python3
"""Build 'Claude Code on Midway' — an accessible, formal lecture for a general
research audience, followed by a hands-on lab (see ../hands-on/).

Design intent (see BUILD.md):
  - declarative slide titles (never opening with What / How / When);
  - <=3 bullets per slide, each reading as a sentence (bold lead, then plain text);
  - captions complement the figure, they do not re-narrate it;
  - clean section dividers with a progress bar, no oversized letters;
  - no speaker notes — everything the audience needs is on the slide face;
  - claims that need a source carry a small attribution; a References section closes.

Run:  python build_deck.py <figures_dir> <out.pptx>
"""
import sys
import deck_engine as D
from deck_engine import caption_line

NPARTS = 7


def build(MEDIA, OUT):
    def img(n):
        return f"{MEDIA}/image{n}.png"

    prs = D.new_deck()

    # ============================= OPENING =============================
    s = D.add_title(prs,
        "RCC WORKSHOP · UNIVERSITY OF CHICAGO",
        "Claude Code on Midway",
        "An AI coding agent for everyday research work — in your terminal, on the cluster.",
        "Youzhi Yu · Research Computing Center")

    D.add_agenda(prs, "Today's session", [
        ("Meet Claude Code", "what an AI agent in the terminal actually is, and why it helps research."),
        ("Getting started on Midway", "install, log in, and run your first session."),
        ("Working efficiently", "tokens, context, and cost — getting more done for less money."),
        ("Make it yours", "teach it your project, save prompts, add your own tools."),
        ("Staying in control", "the permission system that makes it safe on shared hardware."),
        ("Scaling up", "turn one command into a batch, and connect your own cluster tools."),
        ("Trust and honest limits", "prompt injection, your data, and when not to use it."),
    ], note="We close with a hands-on lab: you drive Claude Code to analyze a real dataset.")

    # ===================== PART 1 — MEET CLAUDE CODE =====================
    D.add_divider(prs, 1, NPARTS, "Meet Claude Code",
        "An AI agent that does research work in your terminal.")

    s = D.add_image(prs, "An agent does the work, not just the typing",
        "The step that matters is the last one.", img(1))

    s = D.add_content(prs, "Claude Code, in plain terms",
        "You do not need to be a programmer. If you use a terminal, you can use it.", [
        ("An agent, not autocomplete.", "It reads your files, runs commands, and edits them in a loop until the job is done."),
        ("It lives in your terminal.", "The same place you already run Python and submit cluster jobs."),
        ("One tool, many front doors.", "Terminal, VS Code, JetBrains, GitHub, or driven from a script."),
    ])

    s = D.add_image(prs, "It works in a loop: gather, act, check, repeat",
        "Every setting later in this talk tunes one step of this loop.", img(2),
        source="Loop framing: Anthropic, \"Building effective agents\" (2024) and the Claude Agent SDK.")

    s = D.add_image(prs, "Two parts: a program you control, a model it calls",
        "The program — the diagrams call it the harness — is yours to shape; the model just answers.",
        img(3))

    s = D.add_content(prs, "It finds its way around your project", None, [
        ("It looks the way you do.", "It searches and opens your real files — nothing to index or set up first."),
        ("Your layout is its context.", "A tidy project folder helps it find the right things quickly."),
        ("It checks its own work.", "A test passing, a program finishing cleanly, or a code checker confirms the change worked."),
    ])

    s = D.add_content(prs, "Where it earns its place in research", None, [
        ("Not only code.", "Data wrangling, log triage, plotting, a first-draft methods paragraph, or lit-review notes."),
        ("It scripts.", "One command in, a machine-readable answer out — drop it into a pipeline or a Slurm job."),
        ("Everything is on the record.", "Every action is logged, permission-checked, and costed."),
    ])

    # ===================== PART 2 — GETTING STARTED =====================
    D.add_divider(prs, 2, NPARTS, "Getting started on Midway",
        "Install once, log in once — then you are working.", accent=D.BLUE)

    s = D.add_image(prs, "Install and log in — once, and without admin rights",
        "It installs into your home directory; log in once, then grab a node with internet.",
        img(11))

    s = D.add_two_column(prs, "Two ways to work with it", None,
        {"head": "Interactive — your daily driver", "head_color": D.BLUE, "lines": [
            "Run `claude` in a folder and talk to it.",
            "`@file` points at a file, `!` runs a shell command, `/` fires a saved command.",
            "Where you will spend most of your time — and the hands-on lab.",
        ]},
        {"head": "Headless — `claude -p \"…\"`", "head_color": D.TEAL, "lines": [
            "One prompt in, one answer out.",
            "For scripts, batch jobs, and pipelines.",
            "The 'Scaling up' section, later.",
        ]})

    s = D.add_image(prs, "A session, step by step",
        "It reads, edits, runs the tests, and reports the cost — and you watch every step.",
        img(18))

    s = D.add_table(prs, "Choosing a model for the job",
        "Switch anytime with `/model`. Bigger is smarter but pricier; smaller is fast and cheap for bulk.",
        ["Model", "Best for", "Relative cost"],
        [["Opus", "The hardest reasoning", "$$$"],
         ["Sonnet", "Balanced, everyday work", "$$"],
         ["Haiku", "Fast and cheap — ideal for batch", "$"]],
        colw=[1.5, 4.6, 1.9], row_h=0.62,
        source="Current models and per-token pricing: Anthropic pricing page.")

    s = D.add_content(prs, "A few moves worth knowing", None, [
        ("Shift+Tab", "cycles the permission mode — ask-first, then auto-edit, then read-only plan."),
        ("Esc, then Esc Esc", "stops it; or rewinds to before its last edit (a local checkpoint)."),
        ("/cost  ·  /clear  ·  /help", "check spend, clear its memory for a fresh start, or list every command."),
    ], body_size=14.5, gap=14)

    # ===================== PART 3 — WORKING EFFICIENTLY =====================
    D.add_divider(prs, 3, NPARTS, "Working efficiently",
        "Tokens, context, and cost — how to get more done for less.", accent=D.BLUE)

    s = D.add_content(prs, "What you pay for: input and output",
        "You are billed by the token — very roughly ¾ of a word. Every call has two parts.", [
        ("Input — everything it reads.", "Your prompt, the files it opens, and the whole conversation so far."),
        ("Output — everything it writes.", "Its replies, its edits, and any files it produces."),
        ("Output costs about 5× input.", "On every model, writing a token is priced roughly five times reading one — so a chatty, rewrite-everything agent runs up the bill fast."),
    ], body_size=15, gap=15)
    D.source_tag(prs.slides[-1], "Per-token prices (e.g. Haiku ≈ $1 in / $5 out): Anthropic pricing page.")

    s = D.add_content(prs, "Context is a budget too",
        "The context window is everything the model sees at once — and it re-reads all of it on every turn.", [
        ("It fills as the session grows.", "Every file you open and every message stays in view, and is paid for again as input on the next step."),
        ("A bloated context costs and confuses.", "Long sessions get slower, pricier, and lower-quality as the window fills up."),
        ("Keep it lean.", "`/clear` starts fresh between tasks, `/compact` summarizes, `/context` shows what's loaded — and open only the files you need."),
    ], body_size=15, gap=14)

    s = D.add_content(prs, "Match the effort to the task",
        "`/effort` sets how hard the model thinks before it acts — from low to max.", [
        ("More effort means more thinking.", "Better answers on genuinely hard problems, but more tokens and more time."),
        ("Keep it low for routine work.", "A rename, a summary, or a quick plot needs little deliberation."),
        ("Spend high effort where it pays off.", "A subtle bug or a tricky design — not an everyday edit."),
    ], body_size=15, gap=15)

    s = D.add_content(prs, "A playbook for spending less", None, [
        ("Pick the smallest model that works.", "Haiku for bulk and routine; save Opus for the hardest reasoning."),
        ("Clear context between tasks.", "`/clear` so old files and chatter don't ride along as paid input."),
        ("Scope the ask.", "Open only the files you need; ask for a diff or a number, not a full rewrite."),
        ("Cap runaways; reuse the rest.", "`--max-turns` and a budget stop a loop; repeated context is cached, so follow-ups cost less."),
    ], body_size=14, gap=12)

    # ===================== PART 4 — MAKE IT YOURS =====================
    D.add_divider(prs, 4, NPARTS, "Make it yours",
        "Teach it your project, save your prompts, add your own tools.")

    s = D.add_image(prs, "Everything around the model is yours to shape",
        "A map of what you configure. The model sits in the middle; the harness surrounds it.",
        img(4))

    s = D.add_content(prs, "Project memory: a file it reads every time",
        "`CLAUDE.md` — a plain Markdown file, loaded at the start of every session in that folder.", [
        ("Write down your project's facts.", "“tests live here,” “use the conda env AI,” “raw data is read-only.”"),
        ("`/init` writes a starter for you.", "Keep it short — under about 200 lines. It is advice, not a hard rule."),
        ("It is shared, via git.", "Check it in and the whole lab gets the same onboarding."),
    ])

    s = D.add_image(prs, "Save a prompt you reuse: slash commands",
        "A prompt you write once and fire by name — versioned and shared with your lab.", img(13))

    s = D.add_image(prs, "Skills: abilities the model reaches for itself",
        "You do not call a skill — the model reaches for it when your request matches its description.", img(6),
        source="Agent Skills: Anthropic, \"Equipping agents for the real world with Agent Skills\" (2025).")

    # ===================== PART 5 — STAYING IN CONTROL =====================
    D.add_divider(prs, 5, NPARTS, "Staying in control",
        "The part that makes it safe on a shared cluster.", accent=D.BLUE)

    s = D.add_table(prs, "You decide how much it can do",
        "Shift+Tab cycles these. You set the mode — the model cannot change it.",
        ["Mode", "What it does"],
        [["Ask first (default)", "Reads freely; asks before its first edit or command"],
         ["Accept edits", "Approves its own edits inside the current folder"],
         ["Plan", "Read-only — it proposes changes but makes none"]],
        colw=[2.3, 5.7], row_h=0.6, body_size=13)
    caption_line(prs.slides[-1],
        "A fourth mode skips all checks — only ever in a throwaway container or VM, never on shared files.",
        y=4.55)

    s = D.add_image(prs, "Before anything risky, it stops and asks",
        "You choose: allow once, always allow this kind, or say no and redirect it.", img(19))

    s = D.add_image(prs, "Rules live in a settings file — and a deny always wins",
        "Listed in `.claude/settings.json`. A hook can also veto, as we will see next.",
        img(7), source="Precedence and rule syntax: Anthropic, Claude Code permissions docs.")

    s = D.add_image(prs, "Hooks: a check the program cannot skip",
        "Set it once in `settings.json`; it runs before the tool, on every matching event.",
        img(14))

    s = D.add_content(prs, "Safe on a shared cluster", None, [
        ("Use the permission ladder.", "Plan to explore; auto-edit only in a clean git repo; never bypass on shared files."),
        ("It can read what you can.", "Start it in your project folder, and deny sensitive paths (e.g. `~/.ssh`, `.env`)."),
        ("Keep secrets out of jobs.", "Log in with the token, not a raw API key, so nothing secret rides into a batch job."),
    ])

    # ===================== PART 6 — SCALING UP =====================
    D.add_divider(prs, 6, NPARTS, "Scaling up",
        "From one command to a batch — and out to your own cluster tools.")

    s = D.add_image(prs, "One command, a structured answer",
        "One turn in, one machine-readable object out — the building block of every pipeline.",
        img(5))

    s = D.add_image(prs, "From one call to a Slurm batch",
        "One call is the unit of a batch job — loop it, sum the cost, submit it.", img(9))

    s = D.add_image(prs, "Give it your own tools with MCP",
        "The Model Context Protocol (MCP) is an open standard — the same tool server works in any MCP-aware app.",
        img(15), source="MCP: Anthropic, \"Introducing the Model Context Protocol\" (2024); modelcontextprotocol.io.")

    # ===================== PART 7 — TRUST & LIMITS =====================
    D.add_divider(prs, 7, NPARTS, "Trust and honest limits",
        "Non-negotiable on shared research infrastructure.", accent=D.BLUE)

    s = D.add_content(prs, "The real risk is trust", None, [
        ("Prompt injection is real.", "A web page, a pull request, or a tool's output can carry hidden instructions."),
        ("A skill or tool is software.", "Installing one runs its code as you — accept only tools and hooks you trust."),
    ], body_size=15, gap=16)
    D.source_tag(prs.slides[-1], "OWASP Gen AI Security Project, \"LLM01:2025 Prompt Injection.\"")

    s = D.add_content(prs, "Your data, and your obligations", None, [
        ("Restricted data needs approval.", "IRB, protected health information (PHI), or export-controlled data must clear institutional review first."),
        ("Disclose AI assistance.", "Follow your venue's policy, and keep the transcript as a record."),
        ("Generated code is still yours.", "Review and license it like any other dependency you take on."),
    ])
    D.source_tag(prs.slides[-1], "Anthropic Privacy Center, \"Is my data used for model training?\"")

    s = D.add_content(prs, "Where it falls short", None, [
        ("It can be confidently wrong.", "Invented functions, plausible-but-broken fixes — verify everything it produces."),
        ("It is not reproducible bit-for-bit.", "The model samples, so two runs can differ. Pin what must be exact."),
        ("Skip it for trivia or the unverifiable.", "If you genuinely cannot check the output, do not ship it."),
    ])

    # ===================== HANDS-ON =====================
    D.add_divider(prs, None, NPARTS, "Now you try it",
        "On a real dataset — Claude does the analysis, you review the results.",
        accent=D.TEAL, kicker="HANDS-ON LAB", frac=1.0)

    s = D.add_content(prs, "The lab: analyze a real dataset",
        "A flat folder: Markdown task cards, a `CLAUDE.md`, and an `answers.py` key — no notebook, nothing to download.", [
        ("The data lives online.", "Palmer Penguins — 344 field measurements — read straight from a URL."),
        ("You ask; it analyzes.", "Open `claude` in the `hands-on` folder and paste each task card — it writes and runs the analysis and reports back."),
        ("You stay the reviewer.", "Approve each step, then check its numbers — an agent can be confidently wrong."),
    ])

    s = D.add_table(prs, "The tasks map to what you just learned",
        "Seven short analyses — Claude does the work, you check it — then two optional extras: headless, and your own data.",
        ["Task", "You practice"],
        [["1 · First look", "reading remote data; project memory (CLAUDE.md)"],
         ["2 · Summary by species", "group-and-summarize, reported as a table"],
         ["3 · The heaviest species", "a comparison and a judgment call"],
         ["4 · Flipper vs body mass", "correlations — overall and within groups"],
         ["5 · Data quality", "spotting missing / odd values (without dropping them)"],
         ["6 · Make a figure", "write + run a plotting script → a saved figure"],
         ["7 · Write it up", "turn the numbers into a Results paragraph"]],
        colw=[2.7, 5.3], row_h=0.36, body_size=11, hdr_size=12)

    # ===================== CLOSING =====================
    s = D.add_content(prs, "Takeaways", None, [
        ("You shape the program; the model is swappable.", "Configure the harness — the model just answers."),
        ("It is a loop you govern.", "Advice (CLAUDE.md, skills) guides it; rules (permissions, hooks) bind it."),
        ("Always give it a check.", "A test or an exit code — never ship what you have not verified."),
    ], body_size=15, gap=18)

    REFS_A = [
        "Claude Code: Overview — Anthropic. code.claude.com/docs/en/overview",
        "Advanced setup — Anthropic. code.claude.com/docs/en/setup",
        "Authentication — Anthropic. code.claude.com/docs/en/authentication",
        "Configure permissions — Anthropic. code.claude.com/docs/en/permissions",
        "Choose a permission mode — Anthropic. code.claude.com/docs/en/permission-modes",
        "Hooks reference — Anthropic. code.claude.com/docs/en/hooks",
        "Memory (CLAUDE.md) — Anthropic. code.claude.com/docs/en/memory",
        "Custom subagents — Anthropic. code.claude.com/docs/en/sub-agents",
        "Extend Claude with Skills — Anthropic. code.claude.com/docs/en/skills",
        "Connect tools via MCP — Anthropic. code.claude.com/docs/en/mcp",
        "Run Claude Code programmatically — Anthropic. code.claude.com/docs/en/headless",
    ]
    REFS_B = [
        "Agent SDK overview — Anthropic. code.claude.com/docs/en/agent-sdk/overview",
        "Security — Anthropic. code.claude.com/docs/en/security",
        "Pricing — Anthropic. platform.claude.com/docs/en/docs/about-claude/pricing",
        "Building effective agents (2024) — Anthropic. anthropic.com/engineering/building-effective-agents",
        "Equipping agents with Agent Skills (2025) — Anthropic. anthropic.com/engineering",
        "Introducing the Model Context Protocol (2024) — Anthropic. anthropic.com/news/model-context-protocol",
        "Model Context Protocol — modelcontextprotocol.io",
        "Is my data used for model training? — Anthropic Privacy Center. privacy.claude.com",
        "LLM01:2025 Prompt Injection — OWASP Gen AI Security Project. genai.owasp.org",
        "Interactive Jobs — UChicago RCC. docs.rcc.uchicago.edu/slurm/sinteractive",
        "Python & Jupyter — UChicago RCC. docs.rcc.uchicago.edu/software/apps-and-envs/python",
    ]
    D.add_references(prs, "References (1 of 2)", REFS_A)
    D.add_references(prs, "References (2 of 2)", REFS_B)

    D.add_divider(prs, None, NPARTS, "Thank you — questions?",
        "RCC support: user guide at docs.rcc.uchicago.edu, the help desk, and office hours.",
        accent=D.TEAL, kicker="", frac=1.0)

    D.finalize(prs, OUT, skip_numbers=(1,))
    return prs


if __name__ == "__main__":
    media = sys.argv[1] if len(sys.argv) > 1 else "figures"
    out = sys.argv[2] if len(sys.argv) > 2 else "../claude-code-tutorial.pptx"
    prs = build(media, out)
    n = len(prs.slides._sldIdLst)
    print(f"built {out} — {n} slides")
