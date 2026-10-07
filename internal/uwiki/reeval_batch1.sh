#!/bin/bash
# reeval_batch1.sh -- re-evaluate the 1B-v2 sweep's checkpoints at batch 1.
#
# WHY. On LUMI every eval that pads on the left gave wrong numbers at
# INFERENCE_MAX_NUM_SEQS=8, the setting the sweep was evaluated with: the 1.5B
# baseline reads insertion perplexity 3.60 at batch 1 and 12.84 at batch 8
# (see internal/lumi/env.sh). Affected: knowledge, insertion, contamination
# (log-likelihoods) and verbatim, prompt extraction, DoS (generation). Not
# affected: C4 (always batch 1), the watermark (no padding), MIA (right padding).
#
# Run from the repo root, in the HOST shell on the LUMI login node. It can wait
# for queue room, so run it detached once the dry run looks right:
#
#   TASKS="fk il bm" DRY_RUN=1 bash internal/uwiki/reeval_batch1.sh
#   TASKS="fk il bm" DRY_RUN=0 setsid nohup bash internal/uwiki/reeval_batch1.sh \
#       > "$PE_WORK/reeval_b1.log" 2>&1 < /dev/null &
#   disown
#
# TASKS, space separated, any of: fk il bm vm pe dos mia
#
# The chosen tasks are recomputed with FORCE_EVAL=1, which deletes each old
# result first, so a failure leaves a visible gap and never the old number.
# Every other task is skipped and keeps its result. Insertion runs only the
# experiment the export reads (knowledge-acquisition), 1/57 of a full run.
#
# Inputs: TASKS (required), DRY_RUN (default 1), TIME_EVAL (default 03:00:00,
# raise it for dos/pe/vm), PE_WORK. Everything that defines the evaluation is
# set explicitly below; whatever the shell held is scrubbed first.
set -u

[ -f internal/uwiki/eval_cell_body.sh ] || { echo "run this from the repo root" >&2; exit 1; }
source internal/uwiki/scrub_env.sh
scrub_inherited_env

PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
OUT_ROOT="$PE/unlearning-pareto-1B"
TAGS="1B-v2-lr3e-06 1B-v2-lr1e-05 1B-v2-lr5e-05 1B-v2-lr1e-03"
DRY="${DRY_RUN:-1}"
TASKS="${TASKS:?set TASKS, e.g. TASKS=\"fk il bm\"}"
TIME_EVAL="${TIME_EVAL:-03:00:00}"

declare -A FLAG=([fk]=SKIP_FK [il]=SKIP_IL [bm]=SKIP_BM [vm]=SKIP_VM [pe]=SKIP_PE
                 [dos]=SKIP_DOS [mia]=SKIP_MIA)
for f in SKIP_PPL SKIP_GW SKIP_FK SKIP_IL SKIP_BM SKIP_VM SKIP_PE SKIP_DOS SKIP_MIA; do
  export "$f=1"
done
for t in $TASKS; do
  [ -n "${FLAG[$t]:-}" ] || { echo "unknown task '$t' (use: fk il bm vm pe dos mia)" >&2; exit 1; }
  export "${FLAG[$t]}=0"
done

# Offline, as the sweep's own evals ran: the datasets, the MIA reference model
# and the DoS judge are in the shared cache, and ~200 jobs starting together
# online is what hit the HF rate limit before. Refuse DoS without the judge.
JUDGE="${HF_HOME:-$PE/hf}/hub/models--meta-llama--Meta-Llama-3-8B-Instruct"
if [ "$SKIP_DOS" = "0" ] && ! ls "$JUDGE"/snapshots/*/*.safetensors >/dev/null 2>&1; then
  echo "no cached DoS judge under $JUDGE, and these evals run offline. Not launching." >&2
  exit 1
fi
NOISE="$PE/noise-vectors/OLMo-2-1B-Exp"
MIA_CACHE="$PE/hf/mia-cache-b1"

echo "re-evaluating $OUT_ROOT at batch 1"
echo "  tasks:   $TASKS"
echo "  dry run: $DRY    walltime per checkpoint: $TIME_EVAL"
env | grep '^SKIP_' | sort | sed 's/^/  /'

for tag in $TAGS; do
  if [ ! -d "$OUT_ROOT/$tag" ]; then
    echo "  [skip] $tag not present"
    continue
  fi
  EVAL_CELL_SCRIPT=internal/lumi/eval_pareto_cell.sh \
  OUTPUT_ROOT="$OUT_ROOT" RUN_TAG="$tag" SKIP_ANCHORS=1 SKIP_EPOCH_CKPTS=1 \
  FORCE_EVAL=1 EVAL_MAX_NUM_SEQS=1 INFERENCE_MAX_NUM_SEQS=1 \
  IL_EXPERIMENT=knowledge-acquisition IL_MAX_TOKENS=1000000 BM_SPLIT=0 \
  NOISE_DIR="$NOISE" NOISE_STD=0.075 \
  MIA_BATCH=1 MIA_CACHE_DIR="$MIA_CACHE" MIA_REF_CACHE_DIR="$MIA_CACHE/ref" \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TIME="$TIME_EVAL" DRY_RUN="$DRY" \
    bash internal/uwiki/launch_pareto_evals.sh || echo "  launch FAILED for $tag" >&2
done
