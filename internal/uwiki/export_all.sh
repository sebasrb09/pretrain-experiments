#!/bin/bash
# export_all.sh -- export every LUMI sweep plus the post-training models.
#
#   bash internal/uwiki/export_all.sh
#
# Then copy $PE_WORK/exports-* down to the analysis repo.
#
# PATHS, checked against results_meta.json of each existing export rather than
# assumed:
#   decayed-root            -> exports-decayed-lumi    (1B cooled-down sweep)
#   unlearning-pareto-2.7B  -> exports-2.7B-lumi
#   unlearning-pareto-1B    -> exports-1B-lumi         (the new sweep)
#
# $DATA does NOT exist on LUMI -- it is an ASC variable (internal/asc/env.sh).
# Everything here lands under $PE_WORK.
#
# audit_configs.py and export_results.py MUST run together: the exporter does
# not write results_configs.csv, and a stale one silently hides cells.
set -u

REPO="${REPO:-$PWD}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 1; }
PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
OUT_BASE="${OUT_BASE:-$PE}"

log () { echo "[$(date '+%F %T')] $*"; }

one () {   # one <root-dir-name> <export-dir-name>
  local root="$PE/$1" out="$OUT_BASE/$2"
  if [ ! -d "$root" ]; then
    log "[skip] $1 -- no such root"; return 0
  fi
  mkdir -p "$out"
  log "=== $1 -> $2"
  python internal/uwiki/audit_configs.py \
      --output-root "$root" --out "$out/results_configs.csv" \
    || log "  audit_configs FAILED for $1"
  python internal/uwiki/export_results.py \
      --output-root "$root" --out "$out" \
    || log "  export_results FAILED for $1"
  if [ -f "$out/results_cells.csv" ]; then
    log "  cells: $(( $(wc -l < "$out/results_cells.csv") - 1 ))"
  fi
}

one decayed-root           exports-decayed-lumi
one unlearning-pareto-2.7B exports-2.7B-lumi
one unlearning-pareto-1B   exports-1B-lumi

# ---- the post-training models ----------------------------------------
# These are single models, not sweep cells, so they do NOT match the cell glob
#   <root>/<tag>/<method>/<cell>/step-*/evals
# They were written flat, as $PE/post-training/<model>/<eval>/results.yaml.
#
# collect_anchors() globs <root>/anchors/**/results.yaml recursively and takes
# the model directory name as the anchor's "point", which is exactly the right
# shape for them. So stage a root whose anchors/ IS the post-training tree.
# Copy rather than symlink: glob's ** does not reliably descend symlinked dirs.
PT="$PE/post-training"
if [ -d "$PT" ] && [ -n "$(ls -A "$PT" 2>/dev/null)" ]; then
  STAGE="$PE/post-training-export"
  log "=== post-training -> exports-post-training"
  rm -rf "$STAGE"
  mkdir -p "$STAGE/anchors"
  cp -r "$PT"/* "$STAGE/anchors/" 2>/dev/null || true
  mkdir -p "$OUT_BASE/exports-post-training"
  python internal/uwiki/export_results.py \
      --output-root "$STAGE" --out "$OUT_BASE/exports-post-training" \
    || log "  export_results FAILED for post-training"
  if [ -f "$OUT_BASE/exports-post-training/results_anchors.csv" ]; then
    log "  anchor rows: $(( $(wc -l < "$OUT_BASE/exports-post-training/results_anchors.csv") - 1 ))"
    log "  (results_cells.csv will be empty -- these are models, not cells)"
  fi
else
  log "[skip] post-training -- $PT is empty or missing"
fi

log "=== done. Copy down with: ==="
log "    rsync -av lumi:$OUT_BASE/exports-decayed-lumi   ./"
log "    rsync -av lumi:$OUT_BASE/exports-2.7B-lumi      ./"
log "    rsync -av lumi:$OUT_BASE/exports-1B-lumi        ./"
log "    rsync -av lumi:$OUT_BASE/exports-post-training  ./"
