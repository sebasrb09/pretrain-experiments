#!/bin/bash
# hpo_bridge.sh -- runs SLURM commands on behalf of optuna_hpo.py.
#
#   setsid nohup bash internal/uwiki/hpo/hpo_bridge.sh \
#       > "$PE_WORK/hpo/bridge.log" 2>&1 < /dev/null &
#   disown
#
# WHY. The HPO driver needs optuna and torch, which live in the PyTorch
# Singularity container, so it runs there. SLURM exists only on the host: there
# is no sbatch or squeue inside the container. The driver therefore writes each
# SLURM command as a small request script into BRIDGE_DIR, and this loop, run
# on the HOST from a normal login shell, executes it and writes back stdout,
# stderr and the exit code.
#
# SAFETY. A directory whose scripts get executed is a privilege boundary, and
# project scratch is shared with the whole team. So the directory is private
# (mode 700) and a request is only run if it is owned by the user running this
# loop. Each request script also unsets every experiment variable in
# internal/uwiki/scrub_env.sh before exporting the ones the job needs, so
# nothing from this shell (env.sh's 2.7B OUTPUT_ROOT, say) reaches a job.
#
# One bridge serves any number of drivers. It exits after IDLE_HOURS without a
# request, or at once if BRIDGE_DIR/.stop exists:
#   touch "$PE_WORK/hpo/bridge/.stop"
set -u

PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
DIR="${BRIDGE_DIR:-$PE/hpo/bridge}"
IDLE_HOURS="${IDLE_HOURS:-48}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not on PATH. Run this on the LUMI host, not in the container."
  exit 1
fi
if [ -n "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then
  echo "Refusing: this shell is inside a container. Run the bridge on the host."
  exit 1
fi

mkdir -p "$DIR" && chmod 700 "$DIR" || { echo "cannot create $DIR"; exit 1; }
rm -f "$DIR/.stop"
echo "[$(date '+%F %T')] bridge on $(hostname), pid $$, serving $DIR"

last=$(date +%s)
shopt -s nullglob
while :; do
  touch "$DIR/.heartbeat"
  if [ -f "$DIR/.stop" ]; then
    echo "[$(date '+%F %T')] stop file found, exiting"; rm -f "$DIR/.stop"; exit 0
  fi
  for req in "$DIR"/*.req; do
    base="${req%.req}"
    # Claim it atomically, so two bridges never run the same request.
    mv "$req" "$base.run" 2>/dev/null || continue
    if [ ! -O "$base.run" ]; then
      echo "[$(date '+%F %T')] REFUSED $(basename "$req"): not owned by $(id -un)"
      rm -f "$base.run"; continue
    fi
    bash "$base.run" > "$base.out" 2> "$base.err"
    rc=$?
    # The exit code is written LAST, by rename: its appearance is what tells
    # the driver that .out and .err are complete.
    echo "$rc" > "$base.rc.tmp" && mv "$base.rc.tmp" "$base.rc"
    echo "[$(date '+%F %T')] rc=$rc $(tail -1 "$base.run" | cut -c1-120)"
    rm -f "$base.run"
    last=$(date +%s)
  done
  if [ $(( $(date +%s) - last )) -gt $(( IDLE_HOURS * 3600 )) ]; then
    echo "[$(date '+%F %T')] idle for ${IDLE_HOURS}h, exiting"; exit 0
  fi
  sleep 2
done
