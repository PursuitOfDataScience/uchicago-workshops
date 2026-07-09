#!/usr/bin/env python3
"""Build 'Claude Code on Midway' — an accessible, formal lecture for a general
research audience, followed by a hands-on lab (see ../hands-on/).

Design intent (see BUILD.md):
  - declarative slide titles (never opening with What / How / When);
  - <=3 bullets per slide, each reading as a sentence (bold lead, then plain text);
  - captions complement the figure, they do not re-narrate it;
  - clean section dividers with a progress bar, no oversized letters;
  - concise, factual speaker notes with citations — no 'say aloud' scripts;
  - claims that need a source carry a small attribution; a References section closes.

Run:  python build_deck.py <figures_dir> <out.pptx>
"""
import sys
import deck_engine as D
from deck_engine import set_notes as N, caption_line

NPARTS = 6


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
    N(s, "A lecture followed by a hands-on lab. By the end you can install Claude Code on "
         "Midway, use it for daily research work, keep it safe on a shared cluster, and scale "
         "one command into a batch job. CPU + internet only — no GPU required.")

    D.add_agenda(prs, "Today's session", [
        ("Meet Claude Code", "what an AI agent in the terminal actually is, and why it helps research."),
        ("Getting started on Midway", "install, log in, and run your first session."),
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
    N(s, "Autocomplete finishes a line; a chat assistant answers when you paste code in; an "
         "agent reads files, runs commands, edits, and checks the result — in a loop — while "
         "you supervise. Claude Code is the third kind.")

    s = D.add_content(prs, "Claude Code, in plain terms",
        "You do not need to be a programmer. If you use a terminal, you can use it.", [
        ("An agent, not autocomplete.", "It reads your files, runs commands, and edits them in a loop until the job is done."),
        ("It lives in your terminal.", "The same place you already run Python and submit cluster jobs."),
        ("One tool, many front doors.", "Terminal, VS Code, JetBrains, GitHub, or driven from a script."),
    ])
    N(s, "Anthropic's definition: \"an agentic coding tool that reads your codebase, edits "
         "files, runs commands, and integrates with your development tools.\" It is generally "
         "available (v2.x) with frequent releases. Every action is logged, permission-checked, "
         "and priced; a small-model call costs about a cent. Source: Claude Code overview docs.")

    s = D.add_image(prs, "It works in a loop: gather, act, check, repeat",
        "Every setting later in this talk tunes one step of this loop.", img(2),
        source="Loop framing: Anthropic, \"Building effective agents\" (2024) and the Claude Agent SDK.")
    N(s, "Anthropic describes an agent as an LLM \"using tools based on environmental feedback "
         "in a loop\": gather context, take an action (one tool call), verify the result, repeat "
         "until the goal is met. The 'act' step always clears a permission check first (Part 4).")

    s = D.add_image(prs, "Two parts: a program you control, a model it calls",
        "The program — the diagrams call it the harness — is yours to shape; the model just answers.",
        img(3))
    N(s, "Key mental model. The harness is the claude program on your machine — the thing this "
         "workshop configures. The model is a large language model (Claude) reached over the "
         "network, and it is swappable. Everything else in this talk configures the harness.")

    s = D.add_content(prs, "It finds its way around your project", None, [
        ("It looks the way you do.", "It searches and opens your real files — nothing to index or set up first."),
        ("Your layout is its context.", "A tidy project folder helps it find the right things quickly."),
        ("It checks its own work.", "A test passing, a program finishing cleanly, or a code checker confirms the change worked."),
    ])
    N(s, "'Context' is everything the agent currently has in view. Unlike a search engine that "
         "builds an index first, there is nothing to embed and nothing to go stale — it reads the "
         "live files each session. This is why a clear repo structure pays off.")

    s = D.add_content(prs, "Where it earns its place in research", None, [
        ("Not only code.", "Data wrangling, log triage, plotting, a first-draft methods paragraph, or lit-review notes."),
        ("It scripts.", "One command in, a machine-readable answer out — drop it into a pipeline or a Slurm job."),
        ("Everything is on the record.", "Every action is logged, permission-checked, and costed."),
    ])
    N(s, "Concrete uses: point it at a folder of CSVs and standardize the column names; hand it a "
         "failing analysis script and ask why. It is auditable and can be isolated — but because "
         "the model samples, runs are not bit-for-bit reproducible. Verify the output.")

    # ===================== PART 2 — GETTING STARTED =====================
    D.add_divider(prs, 2, NPARTS, "Getting started on Midway",
        "Install once, log in once — then you are working.", accent=D.BLUE)

    s = D.add_image(prs, "Install and log in — once, and without admin rights",
        "It installs into your home directory; log in once, then grab a node with internet.",
        img(11))
    N(s, "On Midway3: (1) get a node with internet egress — sinteractive/sbatch on the test or "
         "caslake partition, or a login node; many other compute nodes have no egress. "
         "(2) install with the one-line script (no sudo, no Node). (3) 'claude setup-token' does "
         "a one-time paste-URL login that works over SSH; or export ANTHROPIC_API_KEY. Run "
         "chmod 700 ~/.claude — cluster homes can be group-readable. Source: Claude Code setup docs.")

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
    N(s, "Interactive is the conversational REPL. Headless (-p / --print) runs a single "
         "non-interactive turn and exits — the building block of automation. It pauses and asks "
         "before anything risky; we show exactly how in Part 4.")

    s = D.add_image(prs, "A session, step by step",
        "It reads, edits, runs the tests, and reports the cost — and you watch every step.",
        img(18))
    N(s, "The trace runs top to bottom: the user prompt, then Read/Grep/Edit/Bash tool calls with "
         "their results, then a plain-language summary with cost and turn count. The approve/deny "
         "prompts and Shift+Tab modes shown here are explained in Part 4.")

    s = D.add_table(prs, "Choosing a model for the job",
        "Switch anytime with `/model`. Bigger is smarter but pricier; smaller is fast and cheap for bulk.",
        ["Model", "Best for", "Relative cost"],
        [["Opus", "The hardest reasoning", "$$$"],
         ["Sonnet", "Balanced, everyday work", "$$"],
         ["Haiku", "Fast and cheap — ideal for batch", "$"]],
        colw=[1.5, 4.6, 1.9], row_h=0.62,
        source="Current models and per-token pricing: Anthropic pricing page.")
    N(s, "As of mid-2026 the family is Opus 4.8, Sonnet 5, and Haiku 4.5. Approximate list "
         "price per million tokens (input/output): Opus ~$5/$25, Sonnet ~$3/$15, Haiku ~$1/$5 — "
         "confirm on the pricing page, as prices change. Aliases (opus/sonnet/haiku) resolve to "
         "the current version; pin a full model id in shared configs. The lab runs on Haiku.")

    s = D.add_content(prs, "A few moves worth knowing", None, [
        ("Shift+Tab", "cycles the permission mode — ask-first, then auto-edit, then read-only plan."),
        ("Esc, then Esc Esc", "stops it; or rewinds to before its last edit (a local checkpoint)."),
        ("/cost  ·  /clear  ·  /help", "check spend, clear its memory for a fresh start, or list every command."),
    ], body_size=14.5, gap=14)
    N(s, "Also useful: '#' jots a quick note into project memory, '@' mentions a file, '/status' "
         "shows your account and model. Sessions are saved to disk: '--resume' continues one and "
         "'--fork-session' branches it. Esc-Esc rewind undoes Claude's edits locally — it is not a "
         "replacement for git.")

    # ===================== PART 3 — MAKE IT YOURS =====================
    D.add_divider(prs, 3, NPARTS, "Make it yours",
        "Teach it your project, save your prompts, add your own tools.")

    s = D.add_image(prs, "Everything around the model is yours to shape",
        "A map of what you configure. The model sits in the middle; the harness surrounds it.",
        img(4))
    N(s, "This is the roadmap for Parts 3 and 4. Each spoke is a way to shape the harness. One "
         "we will not give its own slide: subagents — a scoped helper (say, a reviewer that can "
         "only read) that keeps your main session focused. The hands-on lab exercises project "
         "memory (CLAUDE.md) and the approve-each-action permission prompt on a real data-analysis "
         "task; commands, skills, subagents, and hooks are covered here in the deck.")

    s = D.add_content(prs, "Project memory: a file it reads every time",
        "`CLAUDE.md` — a plain Markdown file, loaded at the start of every session in that folder.", [
        ("Write down your project's facts.", "“tests live here,” “use the conda env AI,” “raw data is read-only.”"),
        ("`/init` writes a starter for you.", "Keep it short — under about 200 lines. It is advice, not a hard rule."),
        ("It is shared, via git.", "Check it in and the whole lab gets the same onboarding."),
    ])
    N(s, "There is a hierarchy: enterprise/managed policy, then your personal ~/.claude, then the "
         "project file, then a local override. For guarantees you cannot rely on advice — that is "
         "what permissions and hooks (Part 4) are for. Source: Claude Code memory docs.")

    s = D.add_image(prs, "Save a prompt you reuse: slash commands",
        "A prompt you write once and fire by name — versioned and shared with your lab.", img(13))
    N(s, "A slash command is a Markdown file in .claude/commands/<name>.md. Frontmatter can scope "
         "which tools it may use; $ARGUMENTS, inline !commands, and @files expand into the prompt. "
         "This is the prompt YOU fire — contrast with a skill, next. Source: Claude Code commands docs.")

    s = D.add_image(prs, "Skills: abilities the model reaches for itself",
        "You do not call a skill — the model reaches for it when your request matches its description.", img(6),
        source="Agent Skills: Anthropic, \"Equipping agents for the real world with Agent Skills\" (2025).")
    N(s, "A skill is a folder (SKILL.md plus optional scripts and reference files). Progressive "
         "disclosure: the one-line description is always in view (cheap); the body loads only when "
         "your request matches; extra files load only if needed. So a whole library of expertise "
         "costs almost no context until it is relevant. It is the model-invoked twin of a command.")

    # ===================== PART 4 — STAYING IN CONTROL =====================
    D.add_divider(prs, 4, NPARTS, "Staying in control",
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
    N(prs.slides[-1],
        "Use Ask-first for everyday work, Plan to explore safely, Accept-edits for a tight loop but "
        "only in a clean git repo. The fourth mode is bypassPermissions (--dangerously-skip-"
        "permissions): never on a shared filesystem — a bad rm or chmod there hits your whole lab. "
        "Source: Claude Code permission-modes docs.")

    s = D.add_image(prs, "Before anything risky, it stops and asks",
        "You choose: allow once, always allow this kind, or say no and redirect it.", img(19))
    N(s, "This is the everyday interactive experience. The agent cannot approve itself — the "
         "harness asks you. Read-only commands (ls, cat, git status) run without a prompt; edits "
         "and shell commands need approval unless you have allowed them.")

    s = D.add_image(prs, "Rules live in a settings file — and a deny always wins",
        "Listed in `.claude/settings.json`. A hook can also veto, as we will see next.",
        img(7), source="Precedence and rule syntax: Anthropic, Claude Code permissions docs.")
    N(s, "allow / ask / deny lists, e.g. Edit(analysis/**), Read(~/.ssh/**), Bash(rm -rf*). "
         "Precedence: deny beats everything, then a hook can veto, then an allow or mode lets it "
         "run, else it is denied (and asks, if interactive). You set these; the model cannot loosen "
         "them. Managed/enterprise settings can be locked so users cannot override them.")

    s = D.add_image(prs, "Hooks: a check the program cannot skip",
        "Set it once in `settings.json`; it runs before the tool, on every matching event.",
        img(14))
    N(s, "A hook runs on a lifecycle event (e.g. before every Write). Exit 0 allows the action; "
         "exit 2 blocks it and hands the reason back to the model. CLAUDE.md is advice the model "
         "can forget; a hook is policy the harness enforces every time. Source: Claude Code hooks docs.")

    s = D.add_content(prs, "Safe on a shared cluster", None, [
        ("Use the permission ladder.", "Plan to explore; auto-edit only in a clean git repo; never bypass on shared files."),
        ("It can read what you can.", "Start it in your project folder, and deny sensitive paths (e.g. `~/.ssh`, `.env`)."),
        ("Keep secrets out of jobs.", "Log in with the token, not a raw API key, so nothing secret rides into a batch job."),
    ])
    N(s, "On the key leak: a plaintext ANTHROPIC_API_KEY in your shell would be copied by 'sbatch "
         "--export=ALL' into the job environment, readable via scontrol. Prefer the CLAUDE_CONFIG_DIR "
         "login token, or source a chmod-600 key at runtime. Also chmod 700 ~/.claude on shared homes.")

    # ===================== PART 5 — SCALING UP =====================
    D.add_divider(prs, 5, NPARTS, "Scaling up",
        "From one command to a batch — and out to your own cluster tools.")

    s = D.add_image(prs, "One command, a structured answer",
        "One turn in, one machine-readable object out — the building block of every pipeline.",
        img(5))
    N(s, "'claude -p \"…\" --output-format json' runs a single non-interactive turn and returns an "
         "object you can parse from any language: result (the text), total_cost_usd, session_id "
         "(to resume or audit), num_turns, is_error, permission_denials. Add --json-schema to force "
         "the reply into fields validated against a schema. Source: Claude Code headless docs.")

    s = D.add_image(prs, "From one call to a Slurm batch",
        "One call is the unit of a batch job — loop it, sum the cost, submit it.", img(9))
    N(s, "A launcher script loops the headless call over many inputs and totals the cost; a Slurm "
         "job runs it unattended. Preflight one cheap call first, so a node with no internet fails "
         "in seconds. Watch rate limits: dozens of simultaneous calls from one org key will throttle "
         "— stagger them or use per-user keys. The lab's Bonus A does exactly this.")

    s = D.add_image(prs, "Give it your own tools with MCP",
        "The Model Context Protocol (MCP) is an open standard — the same tool server works in any MCP-aware app.",
        img(15), source="MCP: Anthropic, \"Introducing the Model Context Protocol\" (2024); modelcontextprotocol.io.")
    N(s, "Wrap a Slurm submitter, a dataset catalog, or a SQL warehouse as an MCP tool and Claude "
         "calls it like any built-in (named mcp__<server>__<tool>). MCP is an open standard — \"a "
         "USB-C port for AI applications.\" Tools are deny-by-default until you allow them. The lab's "
         "Bonus B ships a tiny MCP server you extend.")

    # ===================== PART 6 — TRUST & LIMITS =====================
    D.add_divider(prs, 6, NPARTS, "Trust and honest limits",
        "Non-negotiable on shared research infrastructure.", accent=D.BLUE)

    s = D.add_content(prs, "The real risk is trust", None, [
        ("Prompt injection is real.", "A web page, a pull request, or a tool's output can carry hidden instructions."),
        ("A skill or tool is software.", "Installing one runs its code as you — accept only tools and hooks you trust."),
    ], body_size=15, gap=16)
    N(s, "Prompt injection is the #1 risk in the OWASP Top 10 for LLM applications: external "
         "content the agent reads can try to alter its behavior. Defenses: least privilege "
         "(permissions), human approval for risky actions, and treating any ingested text as "
         "untrusted. Workspace trust means .mcp.json, hooks, and skills take effect only after you "
         "accept them.")
    D.source_tag(prs.slides[-1], "OWASP Gen AI Security Project, \"LLM01:2025 Prompt Injection.\"")

    s = D.add_content(prs, "Your data, and your obligations", None, [
        ("Restricted data needs approval.", "IRB, PHI, or export-controlled data must clear institutional review first."),
        ("Disclose AI assistance.", "Follow your venue's policy, and keep the transcript as a record."),
        ("Generated code is still yours.", "Review and license it like any other dependency you take on."),
    ])
    N(s, "Anthropic's commercial data policy: \"By default, we will not use your inputs or outputs "
         "from our commercial products to train our models.\" Still, check your plan's retention "
         "and residency terms, and route through Bedrock or Vertex if you need a BAA. "
         "Source: Anthropic Privacy Center.")
    D.source_tag(prs.slides[-1], "Anthropic Privacy Center, \"Is my data used for model training?\"")

    s = D.add_content(prs, "Where it falls short", None, [
        ("It can be confidently wrong.", "Invented functions, plausible-but-broken fixes — verify everything it produces."),
        ("It is not reproducible bit-for-bit.", "The model samples, so two runs can differ. Pin what must be exact."),
        ("Skip it for trivia or the unverifiable.", "If you genuinely cannot check the output, do not ship it."),
    ])
    N(s, "Long sessions degrade as the context window fills — use /clear between tasks and keep "
         "prompts scoped. The honest rule for research: anchor every result on something "
         "deterministic — a test, an exit code, a file that must or must not exist.")

    # ===================== HANDS-ON =====================
    D.add_divider(prs, None, NPARTS, "Now you try it",
        "On a real dataset — Claude does the analysis, you review the results.",
        accent=D.TEAL, kicker="HANDS-ON LAB", frac=1.0)

    s = D.add_content(prs, "The lab: analyze a real dataset",
        "A flat folder of Markdown task cards and a `CLAUDE.md` — no notebook, no setup, nothing to download.", [
        ("The data lives online.", "Palmer Penguins — 344 field measurements — read straight from a URL."),
        ("You ask; it analyzes.", "Open `claude` in the `hands-on` folder and paste each task card — it writes and runs the analysis and reports back."),
        ("You stay the reviewer.", "Approve each step, then check its numbers — an agent can be confidently wrong."),
    ])
    N(s, "The lab is a flat set of Markdown task cards plus a CLAUDE.md naming a hosted CSV "
         "(Palmer Penguins). Claude reads the URL, writes and runs pandas analysis, and reports the "
         "results — the everyday data-analysis loop. Nothing to download; the node needs internet.")

    s = D.add_table(prs, "The tasks map to what you just learned",
        "Seven short analyses — Claude does the work, you check it — then a bonus on your own data.",
        ["Task", "You practice"],
        [["1 · First look", "reading remote data; project memory (CLAUDE.md)"],
         ["2 · Summary by species", "group-and-summarize, reported as a table"],
         ["3 · The heaviest species", "a comparison and a judgment call"],
         ["4 · Flipper vs body mass", "correlations — overall and within groups"],
         ["5 · Data quality", "spotting missing / odd values (without dropping them)"],
         ["6 · Make a figure", "write + run a plotting script → a saved figure"],
         ["7 · Write it up", "turn the numbers into a Results paragraph"]],
        colw=[2.7, 5.3], row_h=0.36, body_size=11, hdr_size=12)
    N(s, "Each analysis builds on the last. Claude loads the hosted CSV, writes and runs the code, "
         "and reports the numbers; the learner reviews and spot-checks. Task 1 also proves CLAUDE.md "
         "loaded, via a codename planted there. A headless bonus and a bring-your-own-data take-home follow.")

    # ===================== CLOSING =====================
    s = D.add_content(prs, "Takeaways", None, [
        ("You shape the program; the model is swappable.", "Configure the harness — the model just answers."),
        ("It is a loop you govern.", "Advice (CLAUDE.md, skills) guides it; rules (permissions, hooks) bind it."),
        ("Always give it a check.", "A test or an exit code — never ship what you have not verified."),
    ], body_size=15, gap=18)
    N(s, "By the end, attendees can install, use, govern, and scale Claude Code on Midway. The "
         "hands-on lab makes each idea concrete on a real dataset.")

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
