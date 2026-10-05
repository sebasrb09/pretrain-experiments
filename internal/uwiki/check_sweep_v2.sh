#!/bin/bash
# check_sweep_v2.sh -- is the 1.5B sweep actually training? Read-only.
#
#   bash internal/uwiki/check_sweep_v2.sh
#
# Answers the question the previous two attempts got wrong for 24 hours: not
# "are jobs in the queue" but "did the jobs receive the right settings and are
# they writing checkpoints". A job can sit in R state all day doing the wrong
# thing, which is exactly what happened.
set -u

PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
ROOT="${OUTPUT_ROOT:-$PE/unlearning-pareto-1B}"
TAG_GLOB="${TAG_GLOB:-1B-v2-lr*}"

hr () { printf '%s\n' "------------------------------------------------------------"; }

hr; echo "1. QUEUE"; hr
if command -v squeue >/dev/null 2>&1; then
  tot=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
  run=$(squeue -u "$USER" -h -t R 2>/dev/null | wc -l)
  pen=$(squeue -u "$USER" -h -t PD 2>/dev/null | wc -l)
  mine=$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -c '^1B-v2-lr' || true)
  echo "  total $tot   running $run   pending $pen   named 1B-v2-lr* : $mine"
  squeue -u "$USER" -h -o '  %.10i %.2t %.10M %.11l %j' 2>/dev/null | head -8
  [ "$tot" -gt 8 ] && echo "  ... $((tot - 8)) more"
else
  echo "  no squeue on PATH (not on the cluster?)"
fi

hr; echo "2. CHECKPOINTS  (the signal that actually matters)"; hr
n_ck=$(find "$ROOT"/$TAG_GLOB -maxdepth 3 -type d -name 'step-*' 2>/dev/null | wc -l)
n_cell=$(find "$ROOT"/$TAG_GLOB -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)
echo "  cells created:        $n_cell   (expect 43 once all have started)"
echo "  step-* checkpoints:   $n_ck   (expect up to 9 per cell, 387 total)"
if [ "$n_ck" -eq 0 ]; then
  echo "  *** ZERO CHECKPOINTS. This is the failure mode from the last two runs."
  echo "      Keep reading: section 4 says whether the settings arrived."
else
  echo "  newest 5:"
  find "$ROOT"/$TAG_GLOB -maxdepth 3 -type d -name 'step-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -5 | while read -r _ p; do
        echo "    $(du -sh "$p" 2>/dev/null | cut -f1)  ${p#$ROOT/}"
      done
fi

hr; echo "3. DISK"; hr
du -sh "$ROOT" 2>/dev/null | sed 's/^/  /'
if command -v lfs >/dev/null 2>&1; then
  lfs quota -h -p "$(stat -c %u "$ROOT" 2>/dev/null)" "$ROOT" 2>/dev/null | tail -2 | sed 's/^/  /' || true
fi

hr; echo "4. DID THE SETTINGS ARRIVE?  (newest job log)"; hr
# The cell echoes its resolved configuration before training. If these lines are
# wrong the job is burning time on the wrong experiment, and nothing else will
# say so.
log=$(ls -t unlearn-lumi_*.out 2>/dev/null | head -1)
if [ -z "$log" ]; then
  log=$(ls -t "$PE"/unlearn-lumi_*.out 2>/dev/null | head -1)
fi
if [ -z "$log" ]; then
  echo "  no unlearn-lumi_*.out found here or in $PE"
  echo "  (run this from the directory you launched sbatch in)"
else
  echo "  log: $log"
  grep -m1 -E 'optim(izer)? (state|repo)|resume.*optim|optim\.pt' "$log" 2>/dev/null | sed 's/^/    /'
  grep -m1 'micro batch:'  "$log" 2>/dev/null | sed 's/^/    /'
  grep -m1 'max steps'     "$log" 2>/dev/null | sed 's/^/    /'
  grep -m1 'checkpoint'    "$log" 2>/dev/null | sed 's/^/    /'
  grep -m1 'forget set:'   "$log" 2>/dev/null | sed 's/^/    /'
  echo "  --- last 3 progress lines ---"
  grep -E '^\s*step\s|step [0-9]+/|loss' "$log" 2>/dev/null | tail -3 | sed 's/^/    /'
fi

hr; echo "5. KNOWN FAILURE SIGNATURES"; hr
pat='split_with_sizes|sum exactly to 8640|out of memory|HIP out of memory|CUDA out of memory|Traceback|could not resolve|does not point at a file|No such file'
hits=0
for f in $(ls -t unlearn-lumi_*.out unlearn-lumi_*.err 2>/dev/null | head -12); do
  m=$(grep -nE "$pat" "$f" 2>/dev/null | head -2)
  if [ -n "$m" ]; then
    hits=$((hits+1)); echo "  $f"; echo "$m" | sed 's/^/      /'
  fi
done
[ "$hits" -eq 0 ] && echo "  none in the 12 newest logs"

hr; echo "VERDICT"; hr
if [ "$n_ck" -gt 0 ]; then
  echo "  Training is producing checkpoints. Healthy."
elif [ "${run:-0}" -gt 0 ]; then
  echo "  Jobs are RUNNING but no checkpoint yet."
  echo "  The first rung is step 1, so a healthy cell writes one within minutes"
  echo "  of starting. If these have been running for over an hour with nothing,"
  echo "  check section 4: MAX_STEPS should be 55 and the optim repo must be 1B."
else
  echo "  Nothing running and nothing written. Check section 5, then the queue."
fi
