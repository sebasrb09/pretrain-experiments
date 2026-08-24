#!/bin/bash
# Smoke test the MUSICA submission path before fanning out any real grid.
#
# Two stages, the second chained with --dependency=afterok so it only runs if
# the first succeeded:
#
#   A. 179M, 2 optimizer steps   -- does the path work at all?
#   B. 1B,  20 optimizer steps   -- how fast is an H100 really, and does 1B fit?
#
# Stage B matters more than it looks: every compute estimate for the sweep rests
# on a guessed 30-50k tokens/sec. Twenty real steps replaces the guess.
#
# Both stages restrict the forget set to ONE experiment. The full library
# default is 5.2M texts, and tokenizing it takes minutes that a 2-step test
# has no reason to spend.
#
# Usage:
#   bash internal/asc/smoke_test.sh              # submit both stages
#   DRY_RUN=1 bash internal/asc/smoke_test.sh    # print, don't submit
#   STAGE=a bash internal/asc/smoke_test.sh      # just the 179M path test
#   bash internal/asc/smoke_test.sh report       # read results once they finish
#
# Env vars:
#   STAGE        a | b | both              (default: both)
#   DRY_RUN      1 to print without submitting
#   FORGET_EXPS  single experiment name    (default: memorization-patterns-rare-1-token-1x)
#                NOTE: one name only -- a space would break the --export list.
#   METHOD       driver to exercise        (default: ce-u, the only one needing no OLMo)
#   TIME         walltime per stage        (default: 1:00:00)
#   MICRO_BATCH  override to probe memory headroom on the H100

set -u
set -o pipefail

CELL="internal/asc/unlearn_cell_1B.sh"
[ -f "$CELL" ] || { echo "ERROR: run this from the repo root ($CELL not found)" >&2; exit 1; }

DRY_RUN="${DRY_RUN:-0}"
STAGE="${STAGE:-both}"
METHOD="${METHOD:-ce-u}"
FORGET_EXPS="${FORGET_EXPS:-memorization-patterns-rare-1-token-1x}"
TIME="${TIME:-1:00:00}"

MODEL_179M="sbordt/OLMo-2-179M-Exp-Unlearning"
MODEL_1B="sbordt/OLMo-2-1B-Exp-Unlearning"
REVISION="stage1-step100000-tokens210B"

BASE="ALL,KEEP_CHECKPOINTS=0,METHOD=${METHOD},VALUE=1e-6,FORGET_EXPS=${FORGET_EXPS}"
[ -n "${MICRO_BATCH:-}" ] && BASE="${BASE},MICRO_BATCH=${MICRO_BATCH}"

# ---------------------------------------------------------------- report mode
if [ "${1:-}" = "report" ]; then
  root="${OUTPUT_ROOT:-${DATA:-/data/fs201378/sr44833}/unlearning-pareto}"
  for tag in smoke-179M smoke-1B; do
    d=$(find "$root/$tag" -name metrics.jsonl 2>/dev/null | head -1)
    echo "=== $tag ==="
    if [ -z "$d" ]; then echo "  no metrics.jsonl yet"; continue; fi
    python - "$d" "$tag" <<'PYEOF'
import json, os, subprocess, sys
path, tag = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path) if l.strip()]
d = os.path.dirname(path)
cfg_name = next(f for f in os.listdir(d) if f.endswith('_config.json'))
cfg = json.load(open(os.path.join(d, cfg_name)))
micro = cfg.get('batch_size', cfg.get('forget_batch_size', 0))
accum = cfg['gradient_accumulation_steps']
seq   = cfg['max_seq_len']
steps = rows[-1].get('optimizer_step', 0)
tok   = steps * accum * micro * seq
print(f'  {steps} steps  micro {micro}  accum {accum}  {tok:,} tokens')

# Steady state, from the in-run clock: excludes model load and tokenization.
if 'elapsed_s' in rows[0]:
    span = rows[-1]['elapsed_s'] - rows[0]['elapsed_s']
    n_micro = len(rows) - 1
    if span > 0 and n_micro > 0:
        tok_span = n_micro * micro * seq
        print(f'  training window {span:.1f}s for {tok_span:,} tokens')
        print(f'  -> {tok_span/span:,.0f} tok/s  STEADY STATE (startup excluded)')
else:
    print('  (no elapsed_s in metrics -- rerun with the updated drivers)')

try:
    out = subprocess.run(['sacct','-X','-n','--name',tag,
                          '--format=Elapsed,State'],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if out:
        el, state = out[-1].split()[0], out[-1].split()[1]
        h, m, s = (el.split('-')[-1].split(':') + ['0','0'])[:3]
        secs = int(h)*3600 + int(m)*60 + int(s)
        print(f'  job elapsed {el} ({state})')
        if secs > 0 and 'elapsed_s' in rows[0]:
            startup = secs - rows[-1]['elapsed_s']
            print(f'  -> startup (load + tokenize) {startup:.0f}s of {secs}s')
except Exception as e:
    print(f'  sacct unavailable: {e}')
PYEOF
  done
  exit 0
fi

submit () {   # $1 job name, $2 extra exports, $3 dependency (may be empty)
  local name="$1" extra="$2" dep="$3" depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterok:${dep}"
  if [ "$DRY_RUN" = "1" ]; then
    # to stderr: stdout of this function is captured as the job id
    echo "  [dry] sbatch -J $name --time=$TIME $depflag --export=${BASE},${extra} $CELL" >&2
    echo "DRYRUN"
  else
    # shellcheck disable=SC2086
    sbatch --parsable -J "$name" --time="$TIME" $depflag \
           --export="${BASE},${extra}" "$CELL"
  fi
}

echo "============================================"
echo "  MUSICA smoke test"
echo "  method:      $METHOD"
echo "  forget set:  $FORGET_EXPS"
echo "  stage:       $STAGE     dry run: $DRY_RUN"
echo "============================================"
echo ""

jid_a=""
if [ "$STAGE" = "a" ] || [ "$STAGE" = "both" ]; then
  echo "--- A: 179M, 2 steps (does the path work) ---"
  jid_a=$(submit "smoke-179M" \
    "MAX_STEPS=2,RUN_TAG=smoke-179M,MODEL=${MODEL_179M},REVISION=${REVISION}" "")
  echo "  job: $jid_a"
  echo ""
fi

if [ "$STAGE" = "b" ] || [ "$STAGE" = "both" ]; then
  echo "--- B: 1B, 20 steps (throughput + memory) ---"
  dep=""
  [ "$STAGE" = "both" ] && [ "$jid_a" != "DRYRUN" ] && dep="$jid_a"
  jid_b=$(submit "smoke-1B" \
    "MAX_STEPS=20,RUN_TAG=smoke-1B,MODEL=${MODEL_1B},REVISION=${REVISION}" "$dep")
  echo "  job: $jid_b${dep:+  (waits for $dep)}"
  echo ""
fi

cat <<EOF
============================================
  Watch:    squeue --me
            tail -f smoke-179M_*.out smoke-1B_*.out
  Results:  bash internal/asc/smoke_test.sh report

  A pass leaves metrics.jsonl and a *_config.json but NO epoch-*/ directory --
  KEEP_CHECKPOINTS=0 removes checkpoints on purpose. An empty directory means
  the job did not get that far; check sacct and the .out file.
============================================
EOF
