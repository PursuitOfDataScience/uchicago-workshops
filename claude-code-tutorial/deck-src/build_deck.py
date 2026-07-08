#!/usr/bin/env python3
"""Build 'Claude Code on Midway' — accessible redesign (critique-applied).

Reuses the existing high-quality figures (diagrams + terminal mockups) from MEDIA,
rebuilds every text/table slide: plain language, day-to-day-first, <=3 bullets,
captions that COMPLEMENT (never re-narrate) the figure, HPC detail in speaker notes.
Run:  python build_deck.py <media_dir> <out.pptx>
"""
import sys
import deck_engine as D
from deck_engine import set_notes as N, caption_line

def build(MEDIA, OUT):
    def img(n):
        return f"{MEDIA}/image{n}.png"

    prs = D.new_deck()

    # ===================== PART A — WHAT IT IS =====================
    s = D.add_title(prs,
        "RCC WORKSHOP · UNIVERSITY OF CHICAGO",
        "Claude Code on Midway",
        "An AI coding agent for your everyday research work — in the terminal, on the cluster.",
        "Youzhi Yu · Research Computing Center")
    N(s, "Welcome. This is a lecture-plus-notebook workshop. Goal: leave able to install "
         "Claude Code on Midway, use it interactively for daily research work, keep it safe on a "
         "shared cluster, and script it into a Slurm batch. CPU + internet only — no GPU needed.")

    D.add_content(prs, "What you'll leave knowing", None, [
        ("What it is,", "and why it helps with research."),
        ("How to install it and use it", "day-to-day, on Midway."),
        ("How to make it yours — and keep it safe —", "on a shared cluster."),
        ("How to turn one command", "into a batch job over thousands of inputs."),
    ], body_size=16, gap=16)

    D.add_divider(prs, "A", "What is Claude Code?",
        "And why you'd put an AI agent in your terminal.")

    s = D.add_image(prs, "From autocomplete to an agent",
        "The jump that matters is the last one.", img(1))
    N(s, "Autocomplete finishes your line; a chat assistant answers when you paste code in; an "
         "agent does the work — reads files, runs commands, edits, checks — while you supervise.")

    s = D.add_content(prs, "What is Claude Code?",
        "You don't need to be a programmer — if you use a terminal, you can use this.", [
        ("An agent, not autocomplete.", "It reads your files, runs commands, and edits code — in a loop, until the job is done."),
        ("It lives in your terminal.", "The same place you already run Python and your cluster jobs."),
        ("One tool, many ways to use it.", "Terminal, VS Code, GitHub, or run from Python."),
    ])
    N(s, "Reassure the room: this is not just for software engineers. A call costs about a cent on "
         "Haiku, and everything it does is logged and undoable. v2, generally available, weekly releases.")

    s = D.add_image(prs, "How it works: a loop",
        "Everything later in this talk is just a setting on this loop.", img(2))
    N(s, "Gather context, decide the next step, act (one tool call), check the result — repeat until "
         "done. The 'act' step always clears a permission check first; we unpack permissions in Part D.")

    s = D.add_image(prs, "Two parts: the program and the model",
        "The program (the diagrams call it the *harness*) is yours to shape; the model just answers.",
        img(3))
    N(s, "Key mental model. 'harness' = the claude program on your machine — what this workshop "
         "configures. The model is a large language model (Claude) reached over the internet; it's "
         "swappable. Don't memorize the inner boxes — we take each one later.")

    s = D.add_content(prs, "How it finds its way around your code", None, [
        ("It looks like you do.", "It searches and opens your real files — nothing to index or set up first."),
        ("Your layout is its context.", "A tidy project folder helps it find the right things."),
        ("It checks its work.", "A test, an exit code, or a linter (which flags code mistakes) confirms the change worked."),
    ])
    N(s, "'context' = everything it currently has in view. Contrast with search engines that build an "
         "index first: here there's nothing to embed and nothing to go stale — it reads the live files.")

    s = D.add_content(prs, "Why researchers find it useful", None, [
        ("Not just code.", "Data wrangling, log triage, plotting, a LaTeX draft, or your lit-review notes."),
        ("It scripts.", "One command, machine-readable output — drop it into a pipeline or a Slurm job."),
        ("Everything's on the record.", "Every action is logged, permission-checked, and priced."),
    ])
    N(s, "Concrete example to say aloud: point it at a folder of CSVs and ask it to standardize "
         "column names, or hand it a failing analysis script and ask why. Auditable and isolable — "
         "but because the model samples, runs are not bit-for-bit reproducible.")

    # ===================== PART B — GET STARTED ON MIDWAY =====================
    D.add_divider(prs, "B", "Getting started on Midway",
        "Install once, log in once — then you're working.", accent=D.BLUE)

    s = D.add_image(prs, "Install and log in — once, no admin rights",
        "Into your home dir; log in once — then grab an internet node (test or caslake).", img(11))
    N(s, "Run it on Midway3, three steps: (1) grab a node with internet — sinteractive/sbatch on the "
         "test or caslake partition (login nodes also reach the internet; many other compute nodes "
         "do NOT); (2) activate the shared env: "
         "source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI ; "
         "then export CLAUDE_CONFIG_DIR=$HOME/.claude and DISABLE_AUTOUPDATER=1; (3) claude --version "
         "should print 2.x. Run chmod 700 ~/.claude — your login token lives there and cluster homes "
         "can be group-readable. 'claude setup-token' does the paste-URL OAuth flow once (works over SSH).")

    s = D.add_two_column(prs, "Two ways to use it", None,
        {"head": "Interactive — your daily driver", "head_color": D.BLUE, "lines": [
            "Run `claude` in a folder and just talk to it.",
            "`@file` points at a file, `!` runs a shell command, `/` fires a saved command.",
            "Where you'll spend most of your time.",
        ]},
        {"head": "Headless — `claude -p \"…\"`", "head_color": D.TEAL, "lines": [
            "One command in, answer out.",
            "For scripts, batch jobs, and pipelines.",
            "The second half of this deck.",
        ]})
    N(s, "It pauses and asks before doing anything risky — we show exactly how in 'Who's in control' "
         "(Part D). Interactive can't run headless, so the notebook's REPL lab is a guided take-home.")

    s = D.add_image(prs, "What a session looks like",
        "It reads, edits, runs the tests, and reports the cost — and you watch every step.", img(18))
    N(s, "Walk through the trace top to bottom: prompt, then Read/Grep/Edit/Bash tool calls with "
         "results, then a plain-English summary with cost and turn count. The approve/deny prompts and "
         "Shift+Tab modes shown here are explained in Part D — flag that now so nobody feels lost.")

    s = D.add_table(prs, "Pick a model for the job",
        "Switch anytime with `/model`. Bigger = smarter but pricier; smaller = fast and cheap for bulk.",
        ["Model", "Best for", "Cost"],
        [["Opus", "The hardest reasoning", "$$$"],
         ["Sonnet", "Everyday, balanced work", "$$"],
         ["Haiku", "Fast, cheap — great for batch", "$   (about 1¢ a call)"]],
        colw=[1.4, 4.4, 2.0], row_h=0.6)
    N(s, "Rough prices per million tokens (in/out): Opus ~$5/$25, Sonnet ~$3/$15, Haiku ~$1/$5. "
         "Aliases (opus/sonnet/haiku) resolve per provider — pin a full model id in shared configs. "
         "The notebook and run.sh use Haiku so a full run is well under $1.")

    s = D.add_content(prs, "Handy moves in an interactive session", None, [
        ("Shift+Tab", "changes how much it can do on its own — from ask-first to hands-off (see “Who's in control”)."),
        ("Esc  /  Esc Esc", "stops it, or rewinds to before its last edit."),
        ("`/cost`  ·  `/clear`  ·  `/help`", "check spend, start a fresh context, list commands."),
    ], body_size=15.5, gap=15)
    N(s, "Also: '#' jots a quick project memory, '@' mentions a file. Sessions are saved to disk — "
         "'--resume' continues one and '--fork-session' branches it; Esc Esc rewind is a local undo of "
         "Claude's edits (not a replacement for git). /context shows what's loaded; /usage shows plan limits.")

    # ===================== PART C — MAKE IT FIT YOUR PROJECT =====================
    D.add_divider(prs, "C", "Make it yours",
        "Teach it your project, save your prompts, add your tools.")

    s = D.add_image(prs, "Everything around the agent is yours to shape",
        "A map of what you can shape — the small §-numbers point to the notebook, not this deck.",
        img(4))
    N(s, "This is the roadmap for Parts C and D. One box we won't give its own slide: subagents — a "
         "scoped helper (say, a reviewer that can only read) that keeps your main session focused. "
         "The §-numbers are hands-on-notebook sections, not slide numbers.")

    s = D.add_content(prs, "Project memory: CLAUDE.md", None, [
        ("A plain Markdown file", "it reads at the start of every session in that folder."),
        ("Put your project's facts in it", "“tests live here,” “use conda env AI,” your house style."),
        ("`/init` writes a starter", "keep it short — it's advice, not a hard rule."),
    ])
    N(s, "There's a hierarchy: org policy, then your ~/.claude, then the project file (check it into "
         "version control so the whole lab shares it). Keep it under ~200 lines. For guarantees you "
         "can't rely on advice — that's what permissions and hooks are for.")

    s = D.add_image(prs, "Save a prompt you reuse: slash commands",
        "A prompt you save once and fire by name.", img(13))
    N(s, "Lives in .claude/commands/<name>.md. Check it into git (version control) and the whole lab "
         "shares it. Frontmatter can scope which tools it may use; $ARGUMENTS, !commands and @files "
         "expand inline. This is the prompt YOU fire — contrast with a skill, next.")

    s = D.add_image(prs, "Skills: prompts the model runs itself",
        "You don't call it — the model reaches for it when your request fits.", img(6))
    N(s, "A skill is a folder (SKILL.md + optional scripts). It loads in layers: the one-line "
         "description is always in view (cheap); the body loads only when your request matches; extra "
         "files load only if needed. The model-invoked counterpart to a slash command.")

    # ===================== PART D — STAYING IN CONTROL =====================
    D.add_divider(prs, "D", "Who's in control",
        "The part that makes it safe on a shared cluster.", accent=D.BLUE)

    s = D.add_table(prs, "You set how much it can do",
        "Shift+Tab cycles these. You set the mode — the model can't.",
        ["Mode", "What it does"],
        [["Ask first  (default)", "Asks before its first edit or command"],
         ["Plan", "Read-only — it proposes, never changes anything"],
         ["Auto-edit", "Approves its own edits inside this folder"]],
        colw=[2.2, 5.6], row_h=0.56, body_size=13)
    caption_line(prs.slides[-1],
        "A fourth mode, Bypass, skips all checks — only in a throwaway container or VM, never on shared files.",
        y=4.58)
    N(prs.slides[-1],
        "Use Ask-first for everyday work; Plan to explore safely; Auto-edit for tight loops but only "
        "in a clean git repo (git = version control that lets you undo). Bypass = bypassPermissions: "
        "never on a shared filesystem — a bad rm or chmod there hits your whole lab.")

    s = D.add_image(prs, "What “asking first” looks like",
        "Before anything risky, it stops and asks — it can't approve itself.", img(19))
    N(s, "You choose: allow once, allow always for this kind of command, or say no and redirect it. "
         "This is the everyday day-1 experience of interactive mode.")

    s = D.add_image(prs, "Rules in a settings file — and who wins",
        "Listed in `.claude/settings.json`; a deny always wins. (A hook can also veto — next slide.)",
        img(7))
    N(s, "allow / ask / deny lists, e.g. Edit(analysis/**), Read(~/.ssh/**), Bash(rm -rf*). Precedence: "
         "deny beats everything, then a hook can veto, then an allow/mode lets it run, else it's denied "
         "(asks, if interactive). You set these; the model cannot loosen them.")

    s = D.add_image(prs, "Hooks: a rule it can't forget",
        "A check the program runs itself, every time — the model can't skip it.", img(14))
    N(s, "A hook is a shell command the harness runs on a matching event (e.g. before every Write). "
         "Exit 0 allows the action; exit 2 blocks it and hands the reason back to the model. CLAUDE.md "
         "is advice the model can forget; a hook is deterministic policy. Admins can ship a locked "
         "settings.json that users can't loosen — right for a cluster.")

    # ===================== PART E — SCALING UP =====================
    D.add_divider(prs, "E", "Scaling up",
        "From one command to a batch of thousands.")

    s = D.add_image(prs, "One command, a structured answer",
        "One turn in, one machine-readable object out — the building block of every script.", img(5))
    N(s, "`claude -p \"…\" --output-format json` runs a single non-interactive turn and returns an "
         "object you can parse from any language: result (the text), total_cost_usd, session_id "
         "(to resume/audit), num_turns, is_error, permission_denials. This is Door 1.")

    s = D.add_image(prs, "Get back exactly the fields you want",
        "Ask for specific fields and types; get validated data back, not prose to parse.", img(12))
    N(s, "`--json-schema` hands the model a schema and returns validated fields, with retries on "
         "mismatch. Ideal for extraction across many files — e.g. pull organism, sample size, and a "
         "significance flag out of hundreds of paper abstracts into a clean table.")

    s = D.add_content(prs, "Every call has a price tag", None, [
        ("The cost is in the output", "dollars, tokens (the units it's billed in), turns (back-and-forth steps), and whether it errored."),
        ("Cap it up front", "`--max-budget-usd` and `--max-turns` stop a runaway."),
        ("In a session, `/cost`", "shows your spend at a glance."),
    ])
    N(s, "total_cost_usd is a client-side estimate — reconcile grant spend against the Console. The "
         "notebook keeps a running ledger as it goes. /usage shows plan limits in the REPL.")

    s = D.add_image(prs, "Watch it fix a failing test",
        "It edits and re-runs the test; you re-check the exit code yourself (0 = passed).", img(8))
    N(s, "Capstone in the notebook. The permission mode — not the model — is what lets it edit "
         "autonomously. Ground truth is the exit code WE check, never the model's claim. TDD is the "
         "strongest pattern: a failing test gives an unattended run a clear place to stop. This one "
         "call is the unit that the next slide fans out.")

    s = D.add_image(prs, "From one call to a Slurm batch",
        "One call is the unit of a batch job — loop it, sum the cost, submit it.", img(9))
    N(s, "The shipped run.sh does exactly this and preflights a cheap Haiku call first, so a node with "
         "no internet fails in seconds instead of after a long job. Watch rate limits: dozens of "
         "simultaneous calls from one org key will throttle — stagger them or use per-user keys.")

    s = D.add_image(prs, "Give it your own cluster tools (MCP)",
        "MCP (Model Context Protocol) lets Claude call tools you write.", img(15))
    N(s, "Wrap a Slurm submitter, a dataset catalog, or a SQL warehouse as an MCP tool and Claude "
         "calls it like any built-in (named mcp__<server>__<tool>). Deny-by-default until you allow it. "
         "The SDK can even define tools in-process, closing over live Python objects.")

    s = D.add_image(prs, "Three ways to script it",
        "Drive it from your own code: the `claude` command, a Python library, or an automated pipeline.",
        img(10))
    N(s, "Door 1: the CLI as a subprocess — any language, shell/Slurm glue. Door 2: the Python Agent "
         "SDK (pip install claude-agent-sdk) — same engine, typed messages, in-process tools, policy "
         "as Python. Door 3: automation — GitHub Actions (@claude on a PR), streamed stdin, the "
         "TypeScript SDK. The notebook's Part II goes deep on the Agent SDK.")

    # ===================== PART F — SAFETY & HONEST LIMITS =====================
    D.add_divider(prs, "F", "Safety, trust & honest limits",
        "Non-negotiable on shared research infrastructure.", accent=D.BLUE)

    s = D.add_content(prs, "Safe on a shared cluster", None, [
        ("Use the permission ladder.", "Plan to explore; auto-edit only in a clean git repo; never bypass on shared files."),
        ("It can read what you can.", "Start it in your project folder, and block sensitive folders (e.g. `~/.ssh`)."),
        ("Keep secrets out of your jobs.", "Log in with the token, not a raw API key, so nothing secret rides into a batch job."),
    ])
    N(s, "On the key leak: the danger is a plaintext ANTHROPIC_API_KEY in your shell — sbatch "
         "--export=ALL would copy it into the job env, readable via scontrol. The shipped run.sh uses "
         "--export=ALL safely BECAUSE it relies on the CLAUDE_CONFIG_DIR OAuth token, not a raw key. "
         "Also chmod 700 ~/.claude on shared homes.")

    s = D.add_content(prs, "Trust is the real risk", None, [
        ("Prompt injection is real.", "A web page, a pull request (a code-review request), or a tool's result can carry hidden instructions."),
        ("A skill is software.", "Installing one runs its code as you — only accept tools and hooks you trust."),
    ], body_size=15.5, gap=16)
    N(s, "Workspace trust: .mcp.json, hooks, and skill grants take effect only after you accept them. "
         "Anything the agent ingests is a potential instruction channel — treat untrusted input with care.")

    s = D.add_content(prs, "Your data and your obligations", None, [
        ("Don't send restricted data.", "IRB, PHI, or export-controlled data needs institutional approval first."),
        ("Disclose AI help", "per your venue's policy; keep the transcript as a record."),
        ("Generated code is still yours.", "Review and license it like any other dependency."),
    ])
    N(s, "API-submitted data is not used for training by default, but check your plan's retention and "
         "residency terms; route through Bedrock/Vertex for a BAA if you need one.")

    s = D.add_content(prs, "When not to reach for it", None, [
        ("It can be confidently wrong.", "Invented APIs, plausible-but-broken fixes — verify everything."),
        ("Not reproducible bit-for-bit.", "The model varies from run to run."),
        ("Skip it for one-liners", "or for work you genuinely can't check."),
    ])
    N(s, "Long sessions degrade as context fills — /clear between tasks and keep prompts scoped. The "
         "honest rule: if you can't verify the output, don't ship it.")

    # ===================== PART G — WRAP =====================
    s = D.add_content(prs, "Takeaways", None, [
        ("You shape the program; the model is swappable.", "Configure the harness; the model just answers."),
        ("It's a loop you govern.", "Advice (CLAUDE.md, skills) guides it; rules (permissions, hooks) bind it."),
        ("Always give it a check.", "A test or an exit code — never ship what you haven't verified."),
    ], body_size=16, gap=18)

    s = D.add_content(prs, "Next: the hands-on notebook", None, [
        ("Run it for real.", "Your first call, memory, commands, permissions, and a live bug-fix."),
        ("Then scale up.", "The Python Agent SDK, and a Slurm batch."),
        ("Budget.", "The whole notebook runs on Haiku for under $1."),
    ], body_size=16, gap=16)
    caption_line(prs.slides[-1],
        "Docs: code.claude.com/docs      ·      Repo: github.com/…/uchicago-workshops", y=4.75)

    D.add_divider(prs, "?", "Thank you — questions?",
        "RCC support: help desk, user guide, and office hours — rcc.uchicago.edu.", accent=D.TEAL)

    D.finalize(prs, OUT, skip_numbers=(1,))
    return prs


if __name__ == "__main__":
    media = sys.argv[1] if len(sys.argv) > 1 else "media"
    out = sys.argv[2] if len(sys.argv) > 2 else "claude-code-tutorial.pptx"
    prs = build(media, out)
    n = len(prs.slides._sldIdLst)
    print(f"built {out} — {n} slides")
