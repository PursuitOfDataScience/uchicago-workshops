#!/bin/bash
#SBATCH --job-name=claude-code-tutorial
#SBATCH --account=<your-account>
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=claude-code-tutorial.out
#SBATCH --error=claude-code-tutorial.err

# ---------------------------------------------------------------------------
# This is a CPU + INTERNET workshop (no GPU). It drives the real `claude` CLI,
# which needs:
#   (1) network egress to api.anthropic.com  -- Midway `test`/`caslake` compute
#       nodes and login nodes have egress; many other clusters do NOT.
#   (2) a valid Claude login. Auth lives under $CLAUDE_CONFIG_DIR; we export it so
#       this batch job (not a child of an interactive Claude session) finds it.
#       Do NOT bake an ANTHROPIC_API_KEY into --export=ALL (it leaks via scontrol);
#       prefer the CLAUDE_CONFIG_DIR OAuth path, or source a chmod-600 key at runtime.
# Submit with:  sbatch --export=ALL,CLAUDE_CONFIG_DIR=$HOME/.claude run.sh
# ---------------------------------------------------------------------------

export PYTHONUNBUFFERED=1
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI

export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
export DISABLE_AUTOUPDATER=1
# NOTE: unlike the GPU workshops, do NOT set HF_HUB_OFFLINE -- this one needs the network.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "Host: $(hostname)  |  claude: $(claude --version 2>/dev/null)  |  config: $CLAUDE_CONFIG_DIR"

# Fail fast if auth/network is broken before spending a full run.
if ! claude -p "ping" --model haiku --output-format json < /dev/null > /tmp/ccpre.$$ 2>&1; then
  echo "ERROR: claude preflight failed (auth or network egress)." >&2
  cat /tmp/ccpre.$$ >&2; rm -f /tmp/ccpre.$$; exit 3
fi
rm -f /tmp/ccpre.$$

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 \
  claude-code-tutorial.ipynb

# Confirm every code cell executed.
python - <<'PY'
import nbformat, sys
nb = nbformat.read("claude-code-tutorial.ipynb", as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
done = sum(1 for c in code if c.get("execution_count") is not None)
errs = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
print(f"Executed {done}/{len(code)} code cells; {len(errs)} errors.")
sys.exit(1 if (done != len(code) or errs) else 0)
PY
echo "Workshop notebook executed successfully."
