<h1 align="center">Claude Code in Practice</h1>

<p align="center"><b>Learn Claude Code by doing: seven short prompts, on any computer.</b><br>
The hands-on exercise of the RCC workshop.</p>

## Set up

1. Install Claude Code (on Windows PowerShell: `irm https://claude.ai/install.ps1 | iex`).
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   ```
2. Get the files and start Claude Code. Log in when asked, then trust the folder.
   ```bash
   git clone https://github.com/PursuitOfDataScience/uchicago-workshops
   cd uchicago-workshops/claude-code-tutorial/hands-on
   claude
   ```
3. In Claude Code, send these two lines one at a time (Sonnet 5.5 at medium effort), then press
   Shift+Tab until the mode line reads `manual mode on`.
   ```text
   /model sonnet
   ```
   ```text
   /effort medium
   ```

## The steps

Send each line in Claude Code, in order, in the same session.

**1 · Permission prompts.** Claude sets up Python and downloads the data; answer each prompt.
```text
set up @1-setup.md
```
**2 · Steering while Claude works.** Type a message as it runs, or ask `/btw` a side question.
```text
explore @2-explore.md
```
**3 · Plan mode.** Press Shift+Tab until it reads `plan mode on`, send this, then approve the
plan with "Yes, manually approve edits".
```text
analyze @3-analysis.md
```
**4 · Subagents.** Two helpers check the result, each in its own context.
```text
verify @4-verify.md
```
**5 · `/goal`.** Claude keeps working until a separate checker confirms the finish line.
```text
/goal @5-finish-line.md
```
**6 · Skills.** Save the check as a skill, then run it with `/check-numbers`.
```text
create @6-skill.md
```
**7 · Take-home.** A starter kit for your own data.
```text
start @7-my-data.md
```

`hands-on/answers.py` is the answer key: Claude is blocked from reading it, so you check the
numbers yourself.

## The same analysis from Python

```bash
pip install claude-agent-sdk
export ANTHROPIC_API_KEY=sk-ant-...   # a Console API key; keep it out of files
cd sdk && python 01_one_shot.py       # then 02, 03, 04: each caps its own spend
```
