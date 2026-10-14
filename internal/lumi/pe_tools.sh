# Shell helpers for following the LUMI runs. Load them into the current shell with
#
#   source internal/lumi/pe_tools.sh        (from the repo root)
#
# or once for every login, by adding that line, with the full path, to ~/.bashrc:
#
#   echo 'source /scratch/project_465003383/unlearning_baselines/pretrain-experiments/internal/lumi/pe_tools.sh' >> ~/.bashrc
#
# Sourcing only defines functions; nothing runs until you call one.
#   pe_report [since]   read-only: what is done, running and waiting
#   news_redo [dry]     the news membership test wherever it is missing or used other labels
#   qa_redo [dry] [ref] PopQA, LAMBADA and SciQ wherever one is missing; ref: only the reference models
#                       (Baseline, Retraining, SGD, smaller, post-training and published models)
# The two *_redo functions submit jobs; run them as
#   setsid nohup bash -c "$(declare -f qa_redo); qa_redo" > "$PE_WORK/qa_redo.log" 2>&1 < /dev/null & disown
# (arguments go after the second qa_redo, e.g. ...; qa_redo ref" > ...)
# Shell variables are not exported on LUMI, so each function finds the work
# directory itself and falls back to /scratch/project_465003383/unlearning_baselines.

pe_report () {   # pe_report [since, default 3 days ago]: read-only. What is done, what runs, what waits.
  local W="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}" since="${1:-$(date -d '3 days ago' +%F)}"
  local want ALL="ppl fk il wm mia bm pe dos math news" XMAX=33 d s g
  want=$(md5sum "$W/pretrain-experiments/resources/train-once-answer-all/muse_news_conditions.jsonl" 2>/dev/null | cut -c1-12)
  _fam () { case $1 in ppl) echo '^c4_perplexity$ 1';; fk) echo '^fictional_knowledge$ 1';; il) echo '^insertion_likelihood$ 1';;
      wm) echo '^gaussian_watermark$ 1';; mia) echo '^mia_(plain|random)_ 12';; bm) echo '^benchmark_contamination_s[0-8]$ 9';;
      pe) echo '^prompt_extraction 2';; dos) echo '^denial_of_service 2';; math) echo '^mathematical_reasoning_ops[135]$ 3';;
      news) echo '^news_memorization$ 1';; esac; }
  _grp () {   # _grp <title> <gate: a C4 value, or -> <task families> <eval dirs...>
    local title="$1" gate="$2" fams="$3"; shift 3
    local n=0 full=0 out=0 part=0 newsok=0 pq=0 qaneed=0 d m f pat exp miss c4
    local -A lack=()
    for d in "$@"; do
      [ -e "$d" ] || [ -d "${d%/evals}" ] || continue
      n=$((n+1))
      m=$(ls "$d" 2>/dev/null | sed -n 's/\.done$//p')
      c4=$(sed -n 's/^perplexity: *//p' "$d/c4_perplexity/results.yaml" 2>/dev/null | head -1)
      # the three benchmarks are wanted wherever the point can be drawn: C4 known and at most XMAX, or no gate at all
      if [ "$gate" = all ] || { [ -n "$c4" ] && awk -v x="$c4" -v g="$XMAX" 'BEGIN{exit !(x <= g)}'; }; then
        qaneed=$((qaneed+1)); [ "$(grep -cxE 'popqa|lambada|sciq' <<< "$m")" = 3 ] && pq=$((pq+1))
      fi
      if [ "$gate" != - ] && [ "$gate" != all ]; then
        if [ -n "$c4" ] && awk -v x="$c4" -v g="$gate" 'BEGIN{exit !(x > g)}'; then out=$((out+1)); continue; fi
      fi
      miss=""
      for f in $fams; do
        read -r pat exp <<< "$(_fam "$f")"
        [ "$(grep -cE "$pat" <<< "$m")" -ge "$exp" ] || { miss="$miss $f"; lack[$f]=$(( ${lack[$f]:-0} + 1 )); }
      done
      if [ -z "$miss" ]; then full=$((full+1)); else part=$((part+1)); fi
      grep -qsE "conditions_md5: *['\"]?$want" "$d/news_memorization/results.yaml" && newsok=$((newsok+1))
    done
    [ "$n" = 0 ] && return
    printf '%-32s %4d' "$title" "$n"
    [ "$gate" != - ] && [ "$gate" != all ] && printf ' | %d beyond C4 %s' "$out" "$gate"
    printf ' | suite complete %d' "$full"
    if [ "$part" -gt 0 ]; then printf ', incomplete %d (missing:' "$part"; for f in $fams; do [ -n "${lack[$f]:-}" ] && printf ' %s %d' "$f" "${lack[$f]}"; done; printf ')'; fi
    case " $fams " in *" news "*) [ "$newsok" != "$((full + part))" ] && printf ' | news on new labels %d' "$newsok";; esac
    printf ' | capability %d/%d\n' "$pq" "$qaneed"
  }
  echo "== RESULTS ON DISK ($(hostname), $(date '+%F %H:%M'))   suite = ppl fk il wm mia bm pe dos math news"
  local cells=() rmu=()
  for d in "$W"/unlearning-pareto-1B/1B-v2-*/*/*/step-*/; do
    [ -d "$d" ] || continue
    case "$d" in */1B-v2-rmu-a0-*) rmu+=("${d}evals");; *) cells+=("${d}evals");; esac
  done
  _grp "1B sweep cells" - "$ALL" "${cells[@]}"
  _grp "1B RMU alpha=0" 31 "$ALL" "${rmu[@]}"
  for s in $(ls -d "$W"/hpo-final/hpo-*-final-* 2>/dev/null | sed 's|.*/||; s/-final-.*//' | sort -u); do
    g=31; [ "$s" = hpo-ce-u-f3 ] && g=-
    _grp "1B finalized ${s#hpo-}" "$g" "$ALL" $(printf '%s/evals ' "$W"/hpo-final/"$s"-final-*/*/*/step-*)
  done
  _grp "1B reference points, batch 1" all "ppl fk il wm mia" "$W"/unlearning-pareto-1B-b1/anchors/*/step-* | sed 's/ | capability.*//'
  _grp "1B reference points, eager" all "bm pe dos math news" "$W"/unlearning-pareto-1B-e8/anchors/*/step-*
  _grp "1B post-training models" all "$ALL" "$W"/post-training-1B/anchors/*
  for d in "$W"/size-reference/anchors/*/ "$W"/published-reference/anchors/*/; do
    [ -d "$d" ] || continue
    printf '%-32s C4 %s | PopQA %s LAMBADA %s SciQ %s | held-out %s | gibberish %s\n' \
      "$(case $d in */size-reference/*) echo smaller;; *) echo published;; esac) model $(basename "$d")" \
      "$(sed -n 's/^perplexity: *//p' "$d/c4_perplexity/results.yaml" 2>/dev/null | head -1 | cut -c1-5)" \
      "$([ -f "$d/popqa.done" ] && echo done || echo -)" "$([ -f "$d/lambada.done" ] && echo done || echo -)" \
      "$([ -f "$d/sciq.done" ] && echo done || echo -)" \
      "$([ -f "$d/benchmark_contamination_s0.done" ] && echo done || echo -)" \
      "$([ -f "$d/denial_of_service.done" ] && echo done || echo -)"
  done
  _grp "2.7B reference points" all "$ALL" "$W"/reference-2.7B/anchors/*/step-*
  for s in $(ls -d "$W"/hpo-final-2.7B/hpo-*-final-* 2>/dev/null | sed 's|.*/||; s/-final-.*//' | sort -u); do
    _grp "2.7B finalized ${s#hpo-}" 33 "$ALL" $(printf '%s/evals ' "$W"/hpo-final-2.7B/"$s"-final-*/*/*/step-*)
    [ -z "$(ls -d "$W"/hpo-final-2.7B/"$s"-final-*/*/*/step-* 2>/dev/null)" ] && printf '%-32s no checkpoint yet (training)\n' "2.7B finalized ${s#hpo-}"
  done
  echo "== SEARCHES (trials with a result / trials started)"
  for s in $(ls -d "$W"/hpo/hpo-*-wm-t[0-9]* 2>/dev/null | sed 's|.*/||; s/-t[0-9]*$//' | sort -u); do
    printf '%-32s %3d / %3d\n' "$s" "$(ls "$W"/hpo/"$s"-t[0-9]*/*/*/hpo_result.json 2>/dev/null | wc -l)" "$(ls -d "$W"/hpo/"$s"-t[0-9]* 2>/dev/null | wc -l)"
  done
  echo "== QUEUE"
  squeue -u "$USER" -h -o "%T %j" | sed -E \
    -e 's/ pe-anchor-2\.7B-.*-(b1|e8|e8math|c4|util)$/ 2.7B reference points, group \1/' \
    -e 's/ pe-(hpo-[A-Za-z0-9.-]+)-final-t[0-9]+-.*-(b1|e8|e8math|c4)$/ finalized \1, evaluation group \2/' \
    -e 's/ (hpo-[A-Za-z0-9.-]+)-final-t[0-9]+$/ finalized \1, TRAINING/' \
    -e 's/ (hpo-[a-z0-9-]+)-t[0-9]+$/ search \1, trial/' \
    -e 's/ pe-1B-v2-rmu-a0-.*-(b1|e8|e8math|c4)$/ 1B RMU alpha=0, evaluation group \1/' \
    -e 's/ pe-model-.*-(b1|e8|e8math|c4|util)$/ whole models, group \1/' \
    -e 's/ pe-.*-qa$/ PopQA backfill/' \
    -e 's/ pe-anchor-.*$/ 1B reference points (news)/' \
    -e 's/ pe-(1B-v2|hpo-ce-u-f3-final)-.*$/ 1B checkpoints (news)/' \
    -e 's/ mid-.*$/ mid-training/' | sort -k2 | uniq -c | awk '{c=$1; st=$2; $1=""; $2=""; sub(/^ +/, ""); printf "  %4d %-8s %s\n", c, st, $0}'
  [ -z "$(squeue -u "$USER" -h)" ] && echo "  (empty)"
  echo "== DETACHED DRIVERS ON THIS LOGIN NODE (they wait, then submit; other login nodes are not visible from here)"
  ps -u "$USER" -o etime=,args= | grep -E "optuna_hpo\.py|qa_redo|news_redo|hpo_bridge\.sh|launch_pareto" | grep -v "grep\|sed -E" | sed -E \
    -e 's/^ *([0-9:-]+) .*optuna_hpo\.py(.*)$/  \1  driver:\2/' -e 's/^ *([0-9:-]+) .*hpo_bridge\.sh.*$/  \1  the bridge that submits for the drivers/' \
    -e 's/^ *([0-9:-]+) .*(qa_redo|news_redo)[^;]*;? ?(qa_redo|news_redo).*$/  \1  \3 (backfill loop)/' \
    -e 's/ --(poll|storage|submit-cap|budget-tokens|max-batch|max-parallel|eval-minutes) [^ ]+//g' | cut -c1-170
  local HB="$W/hpo/bridge/.heartbeat"
  [ -f "$HB" ] && echo "  bridge heartbeat $(( $(date +%s) - $(stat -c %Y "$HB") )) s ago (fine below 300)"
  echo "== JOBS SINCE $since"
  sacct -u "$USER" -X -S "$since" -n -o State%20 | awk '{print $1}' | sort | uniq -c | awk '{printf "%s %s   ", $1, $2} END{print ""}'
  echo "-- ended badly (cancelled ones not counted):"
  sacct -u "$USER" -X -S "$since" -n -o JobName%70,State%20,Elapsed | awk '$2 ~ /FAILED|TIMEOUT|OUT_OF_ME|NODE_FAIL/' | tail -20
}

news_redo () {   # news_redo [dry]: the news membership test wherever it is missing or used another conditions file
  local dry=0; [ "${1:-}" = dry ] && dry=1
  local W="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
  local F=resources/train-once-answer-all/muse_news_conditions.jsonl
  [ -f internal/lumi/eval_pareto_cell.sh ] && [ -s "$F" ] || { echo "run from the repo root, with the conditions file built"; return 1; }
  grep -q '"config": "train"' "$F" || { echo "REFUSING: $F is the old conditions file; rebuild it first"; return 1; }
  local want queued p busy=0
  want=$(md5sum "$F" | cut -c1-12)
  queued=$(squeue -u "$USER" -h -o %j) || { echo "squeue failed"; return 1; }
  # a news launcher that is still submitting would submit the same checkpoints again
  for p in $(pgrep -u "$USER" -f 'launch_pareto[_]evals.sh'); do
    case "$(readlink "/proc/$p/fd/1" 2>/dev/null)" in *news*) busy=1; echo "news launcher still running: pid $p -> $(readlink "/proc/$p/fd/1")";; esac
  done
  if [ "$busy" = 1 ] && [ "$dry" = 0 ]; then echo "REFUSING to submit until it has finished (the listing works: news_redo dry)"; return 1; fi
  ( source internal/uwiki/scrub_env.sh && scrub_inherited_env > /dev/null
    # exactly the launcher's REEVAL="news" EVAL_BATCH=8 job
    for f in SKIP_PPL SKIP_GW SKIP_FK SKIP_IL SKIP_BM SKIP_VM SKIP_PE SKIP_DOS SKIP_MIA SKIP_NEWS SKIP_MATH; do export $f=1; done
    export SKIP_NEWS=0 NEWS_N=0 FORCE_EVAL=1 EVAL_MAX_NUM_SEQS=1 INFERENCE_MAX_NUM_SEQS=8 INFERENCE_ATTN_IMPL=eager \
           ALLOW_ROCM_PADDED_BATCHES=1
    n=0; ok=0; inq=0
    one () {   # one <job name> <result dir> <exports> <offline 0|1>
      local have; have=$(grep -hs conditions_md5 "$2/news_memorization/results.yaml" | awk '{print $2}' | tr -d "'\"")
      if [ "$have" = "$want" ]; then ok=$((ok+1)); return; fi
      if grep -qxF "$1" <<< "$queued"; then inq=$((inq+1)); return; fi
      n=$((n+1))
      if [ "$dry" = 1 ]; then echo "  [dry] $1   (now: ${have:-no result})"; return; fi
      while [ "$(squeue -u "$USER" -h | wc -l)" -ge 195 ]; do sleep 60; done
      HF_HUB_OFFLINE=$4 HF_DATASETS_OFFLINE=$4 sbatch -J "$1" -t 02:00:00 --export=ALL,"$3" internal/lumi/eval_pareto_cell.sh
    }
    # the sweep cells and the CE-U F3 finalize test, offline like the launcher's jobs. The RMU
    # alpha = 0 cells and the other finalized cells get news from their own suite jobs.
    for ckpt in "$W"/unlearning-pareto-1B/1B-v2-*/*/*/step-* "$W"/hpo-final/hpo-ce-u-f3-final-*/*/*/step-*; do
      [ -d "$ckpt" ] || continue
      case "$ckpt" in */1B-v2-rmu-a0-*) continue;; esac
      cell=$(dirname "$ckpt"); method=$(basename "$(dirname "$cell")"); tag=$(basename "$(dirname "$(dirname "$cell")")")
      one "pe-$tag-$method-$(basename "$cell")-$(basename "$ckpt")" "$ckpt/evals" "CELL_DIR=$cell,CKPT=$ckpt,EVAL_OUT=$ckpt/evals" 1
    done
    # Baseline, Retraining and the SGD control: steps 100000-110000 of the two repos, online like their other evals
    A="$W/unlearning-pareto-1B-e8/anchors"
    for abs in 100000 101000 102000 103000 104000 105000 106000 107000 108000 109000 110000; do
      case $abs in 100000) rev=stage1-step100000-tokens210B;; 110000) rev=stage1-step110000-tokens231B;; *) rev=stage1-step$abs;; esac
      rel=$((abs - 100000)); point=unlearn-baseline; [ "$rel" = 0 ] && point=baseline
      one "pe-anchor-$point-$abs" "$A/$point/step-$rel" "MODEL=sbordt/OLMo-2-1B-Exp-Unlearning,REVISION=$rev,EVAL_OUT=$A/$point/step-$rel" 0
      one "pe-anchor-deep-ignorance-$abs" "$A/deep-ignorance/step-$rel" "MODEL=sbordt/OLMo-2-1B-Unlearning,REVISION=$rev,EVAL_OUT=$A/deep-ignorance/step-$rel" 0
    done
    echo "news_redo: conditions file $want; $ok done with it, $inq waiting in the queue, $n $([ "$dry" = 1 ] && echo 'to submit' || echo submitted)" )
}

qa_redo () {   # qa_redo [dry] [ref]: the capability benchmarks (PopQA, LAMBADA, SciQ) on every 1B result that lacks one
  # (until 2026-10-14 ARC-Challenge, CommonsenseQA and OpenBookQA; dropped: every
  # benchmark of the TOAA paper was contaminated in the experiment models, per its authors)
  local dry=0 ref=0 a; for a in "$@"; do case $a in dry) dry=1;; ref) ref=1;; *) echo "qa_redo [dry] [ref]"; return 1;; esac; done
  local W="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}" XMAX=33 TASKS="popqa lambada sciq" t
  for t in $TASKS; do
    [ -s "resources/train-once-answer-all/capability_$t.json" ] || { echo "run from the repo root, after pulling the benchmark task files"; return 1; }
  done
  [ -f internal/lumi/eval_pareto_cell.sh ] || { echo "run from the repo root, after pulling the benchmark task files"; return 1; }
  grep -q 'SKIP_CAP' internal/uwiki/eval_cell_body.sh || { echo "REFUSING: this checkout's eval body has no SKIP_CAP block; git pull first"; return 1; }
  local queued; queued=$(squeue -u "$USER" -h -o %j) || { echo "squeue failed"; return 1; }
  ( source internal/uwiki/scrub_env.sh && scrub_inherited_env > /dev/null
    # the capability block only, with the settings of the suite's eager group (batch 8, eager
    # attention); a task already done in a result directory is skipped by its .done marker
    for f in SKIP_PPL SKIP_GW SKIP_FK SKIP_IL SKIP_BM SKIP_VM SKIP_PE SKIP_DOS SKIP_MIA SKIP_NEWS SKIP_MATH SKIP_QA SKIP_CAP; do export $f=1; done
    export SKIP_CAP=0 CAP_TASKS="$TASKS" FORCE_EVAL=0 EVAL_MAX_NUM_SEQS=1 INFERENCE_MAX_NUM_SEQS=8 INFERENCE_ATTN_IMPL=eager ALLOW_ROCM_PADDED_BATCHES=1
    n=0; ok=0; inq=0; far=0
    one () {   # one <job name> <result dir> <exports> <offline 0|1> [further .done markers wanted]
      local t; for t in $TASKS ${5:-}; do [ -f "$2/$t.done" ] || break; done
      if [ -f "$2/$t.done" ]; then ok=$((ok+1)); return; fi
      if grep -qxF "$1" <<< "$queued"; then inq=$((inq+1)); return; fi
      n=$((n+1))
      if [ "$dry" = 1 ]; then [ "$n" -le 12 ] && echo "  [dry] $1"; return; fi
      while [ "$(squeue -u "$USER" -h | wc -l)" -ge 195 ]; do sleep 60; done
      # 3 h until timed (PopQA: 14,267 short generations at batch 8; LAMBADA and SciQ take minutes)
      HF_HUB_OFFLINE=$4 HF_DATASETS_OFFLINE=$4 sbatch -J "$1" -t 03:00:00 --export=ALL,"$3" internal/lumi/eval_pareto_cell.sh
    }
    # outside references: the decayed experiment pair (with / without the inserted data) and AI2's
    # OLMo-2-0425-1B (4T tokens, should score highest); then the smaller models. The pair also gets
    # the held-out questions (contamination split 0), whose accuracy TOAA's Table 3 reports for
    # exactly these two models (55.15% OLMo-2-1B-Exp, 55.51% OLMo-2-1B): a check of our scoring
    for spec in OLMo-2-1B-Exp:sbordt/OLMo-2-1B-Exp:main OLMo-2-1B:sbordt/OLMo-2-1B:main \
                OLMo-2-0425-1B:allenai/OLMo-2-0425-1B:main; do
      IFS=: read -r name repo rev <<< "$spec"
      one "pe-model-$name-qa" "$W/published-reference/anchors/$name" \
          "MODEL=$repo,REVISION=$rev,EVAL_OUT=$W/published-reference/anchors/$name,SKIP_BM=0,BM_SPLITS=0" 0 \
          benchmark_contamination_s0
    done
    for size in 546M 179M; do
      one "pe-model-OLMo-2-$size-qa" "$W/size-reference/anchors/OLMo-2-$size" \
          "MODEL=sbordt/OLMo-2-$size-Unlearning,REVISION=stage1-step100000-tokens210B,EVAL_OUT=$W/size-reference/anchors/OLMo-2-$size" 0
    done
    # trained checkpoints inside the figure (C4 perplexity at most $XMAX): the sweep, RMU alpha = 0, the finalized cells
    [ "$ref" = 1 ] || for ckpt in "$W"/unlearning-pareto-1B/1B-v2-*/*/*/step-* "$W"/hpo-final/hpo-*-final-*/*/*/step-*; do
      [ -d "$ckpt" ] || continue
      c4=$(sed -n 's/^perplexity: *//p' "$ckpt/evals/c4_perplexity/results.yaml" 2>/dev/null | head -1)
      if [ -z "$c4" ] || awk -v x="$c4" -v g="$XMAX" 'BEGIN{exit !(x > g)}'; then far=$((far+1)); continue; fi
      cell=$(dirname "$ckpt"); method=$(basename "$(dirname "$cell")"); tag=$(basename "$(dirname "$(dirname "$cell")")")
      one "pe-$tag-$method-$(basename "$cell")-$(basename "$ckpt")-qa" "$ckpt/evals" "CELL_DIR=$cell,CKPT=$ckpt,EVAL_OUT=$ckpt/evals" 1
    done
    # Baseline, Retraining and the SGD control (steps 100000-110000), online like their other evaluations
    A="$W/unlearning-pareto-1B-e8/anchors"
    for abs in 100000 101000 102000 103000 104000 105000 106000 107000 108000 109000 110000; do
      case $abs in 100000) rev=stage1-step100000-tokens210B;; 110000) rev=stage1-step110000-tokens231B;; *) rev=stage1-step$abs;; esac
      rel=$((abs - 100000)); point=unlearn-baseline; [ "$rel" = 0 ] && point=baseline
      one "pe-anchor-$point-$abs-qa" "$A/$point/step-$rel" "MODEL=sbordt/OLMo-2-1B-Exp-Unlearning,REVISION=$rev,EVAL_OUT=$A/$point/step-$rel" 0
      one "pe-anchor-deep-ignorance-$abs-qa" "$A/deep-ignorance/step-$rel" "MODEL=sbordt/OLMo-2-1B-Unlearning,REVISION=$rev,EVAL_OUT=$A/deep-ignorance/step-$rel" 0
    done
    # the six post-training models, from their staged folders
    for d in "$W"/post-training-1B/anchors/*/; do
      [ -d "$d" ] || continue
      name=$(basename "$d")
      [ -f "$W/models/$name/config.json" ] || { echo "  no staged model for $name, skipped"; continue; }
      one "pe-model-$name-qa" "${d%/}" "MODEL=$W/models/$name,EVAL_OUT=${d%/}" 1
    done
    [ "$dry" = 1 ] && [ "$n" -gt 12 ] && echo "  [dry] ... and $((n - 12)) more"
    echo "qa_redo: $ok done, $inq waiting in the queue, $n $([ "$dry" = 1 ] && echo 'to submit' || echo submitted); $far checkpoints beyond perplexity $XMAX or without one are left out" )
}
