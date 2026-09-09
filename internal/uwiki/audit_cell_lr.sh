#!/bin/bash
# Report the learning rate every trained cell actually used.
#
#   bash internal/uwiki/audit_cell_lr.sh                 # every cell
#   bash internal/uwiki/audit_cell_lr.sh '1B-p3d-*'      # one sweep
#
# WHY THIS EXISTS
# ---------------
# The RUN_TAG is not a reliable record of the learning rate. `parse_tag` in
# export_results.py only recovers it when the tag ends in lr<value>, which is
# false for ~20 sweeps -- `1B-lr1e-5-base` carries the rate in the middle,
# `1B-p2-satimp-rt` not at all. Worse, unlearn_cell_body.sh passes the rate
# only when LR is set:
#
#     [ -n "${LR:-}" ] && LR_ARGS=(--learning-rate "$LR")
#
# so a cell launched without LR= silently takes the driver's own default and
# nothing in the tag or the job banner records what that was.
#
# Each driver does dump its real configuration into the cell directory as
# <driver>_config.json (reweighted_ga_config.json for wga and satimp, plus
# npo_, simnpo_, graddiff_, ga_, ce_u_, rmu_). That file is the ground truth
# and this walks it. export_results.py reads the same file for its `lr`
# column; this script is the standalone view, for auditing a sweep before
# trusting a comparison that holds the learning rate fixed.
#
# NOTE the exclusions: HuggingFace writes tokenizer_config.json and
# generation_config.json into the same tree and they match *_config.json too.
# Globbing without excluding them returns a tokenizer config with no
# learning_rate key, which looks exactly like a missing config.

set -u
set -o pipefail

PATTERN="${1:-*}"
ROOT="${OUTPUT_ROOT:-${PE_DATA:-${DATA:-$HOME/pretrain-experiments}}/unlearning-pareto}"

[ -d "$ROOT" ] || { echo "ERROR: no sweep root at $ROOT (set OUTPUT_ROOT)" >&2; exit 1; }

echo "sweep root: $ROOT"
echo "pattern:    $PATTERN"
echo

n=0
missing=0
while IFS= read -r cfg; do
  cell=$(dirname "$cfg")
  rel=${cell#"$ROOT"/}
  lr=$(grep -o '"learning_rate":[^,}]*' "$cfg" | head -1 | tr -d ' "' | cut -d: -f2)
  if [ -z "$lr" ]; then lr="MISSING"; missing=$((missing + 1)); fi
  printf '%-52s %-28s %s\n' "$rel" "$(basename "$cfg")" "$lr"
  n=$((n + 1))
done < <(find "$ROOT"/$PATTERN -name '*_config.json' \
              ! -name 'tokenizer_config.json' \
              ! -name 'generation_config.json' 2>/dev/null | sort)

echo
echo "$n cells, $missing without a recorded learning rate"

# A sweep that varies a method hyperparameter is only interpretable if every
# cell in it shares one learning rate. Flag the ones that do not.
echo
echo "--- distinct rates per method (a method with >1 row here mixes rates) ---"
find "$ROOT"/$PATTERN -name '*_config.json' \
     ! -name 'tokenizer_config.json' ! -name 'generation_config.json' 2>/dev/null \
  | while IFS= read -r cfg; do
      cell=$(dirname "$cfg"); rel=${cell#"$ROOT"/}
      method=$(echo "$rel" | awk -F/ '{print $2}')
      lr=$(grep -o '"learning_rate":[^,}]*' "$cfg" | head -1 | tr -d ' "' | cut -d: -f2)
      echo "$method $lr"
    done | sort -u | awk '{c[$1]=c[$1]" "$2} END {for (m in c) printf "  %-18s%s\n", m, c[m]}'
