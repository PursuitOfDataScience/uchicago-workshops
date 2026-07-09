#!/bin/bash
#SBATCH --job-name=lakewatch-batch
#SBATCH --account=<your-pi-account>
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=lakewatch-batch.out
#SBATCH --error=lakewatch-batch.err

# ---------------------------------------------------------------------------
# Bonus A — run the headless batch (classify_logs.sh) as an unattended Slurm job.
# This is a CPU + INTERNET job (no GPU). It drives the real `claude` CLI, which needs:
#   (1) network egress to api.anthropic.com. On Midway3, login nodes and the
#       test/caslake partitions have egress; many other compute nodes do NOT.
#   (2) a valid Claude login. Auth lives under $CLAUDE_CONFIG_DIR; we export it so this
#       batch job (not a child of an interactive Claude session) can find it.
#       Do NOT bake an ANTHROPIC_API_KEY into --export=ALL — it is visible via scontrol.
#       Prefer the CLAUDE_CONFIG_DIR login, or source a chmod-600 key file at runtime.
# Submit with:  sbatch --export=ALL,CLAUDE_CONFIG_DIR=$HOME/.claude run.sh
# ---------------------------------------------------------------------------

export PYTHONUNBUFFERED=1
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
export DISABLE_AUTOUPDATER=1

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "Host: $(hostname) | claude: $(claude --version 2>/dev/null) | config: $CLAUDE_CONFIG_DIR"

# Fail fast if auth/network is broken, before spending a full run.
if ! claude -p "ping" --model haiku --output-format json < /dev/null > /tmp/cc_pre.$$ 2>&1; then
  echo "ERROR: claude preflight failed (auth or network egress)." >&2
  cat /tmp/cc_pre.$$ >&2; rm -f /tmp/cc_pre.$$; exit 3
fi
rm -f /tmp/cc_pre.$$

bash classify_logs.sh
echo "Batch complete."
