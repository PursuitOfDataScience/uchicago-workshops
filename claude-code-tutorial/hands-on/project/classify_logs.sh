#!/usr/bin/env bash
# Bonus A — turn ONE headless call into a batch over many files.
#
# For each instrument log, ask Claude for a one-word failure category, then total the
# cost. Each call is `claude -p` reading the file on standard input (the Unix-composable
# pattern: `cat file | claude -p "..."`). Run from the project root:
#     bash classify_logs.sh
set -euo pipefail

MODEL="${MODEL:-haiku}"
PROMPT="In ONE word — CALIBRATION, BATTERY, CONNECTION, or OK — classify the instrument \
problem described on standard input. Reply with only the word."

total=0.0
printf "%-26s  %-12s  %s\n" "log" "category" "cost_usd"
printf "%-26s  %-12s  %s\n" "--------------------------" "------------" "--------"
for f in logs/*.log; do
  out="$(claude -p "$PROMPT" --model "$MODEL" --output-format json < "$f")"
  category="$(printf '%s' "$out" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'].strip().split()[0])")"
  cost="$(printf '%s' "$out" | python3 -c "import sys,json;print(json.load(sys.stdin).get('total_cost_usd') or 0)")"
  printf "%-26s  %-12s  %s\n" "$(basename "$f")" "$category" "$cost"
  total="$(python3 -c "print(${total} + ${cost})")"
done
printf "\nTOTAL: \$%.4f over %d logs\n" "$total" "$(ls logs/*.log | wc -l)"
