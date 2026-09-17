"""Delete the WEIGHTS of checkpoints that have already been evaluated.

WHY THIS EXISTS
---------------
`prune_trainer_state.py` reclaims optimizer state. That is no longer where the
space is: on a full 9.8 TB volume the breakdown was

    8.43 TB  *.safetensors      <- checkpoint weights
    1.62 TB  trainer_state.pt
    0.19 TB  optim.pt

A step-N checkpoint's weights are needed for exactly three things: running the
evals, resuming training, and re-running an eval later. Once every eval in the
suite has written its `.done` marker, the numbers the analysis reads live in
`evals/*/results.yaml`, and the ~6 GB of weights beside them is dead storage.

WHAT IT REMOVES, AND WHAT IT KEEPS
----------------------------------
Removed from a qualifying checkpoint directory:

    *.safetensors, *.bin          the weights, which are the whole point
    trainer_state.pt              a weightless checkpoint is not resumable, and
                                  leaving it would let find_latest_checkpoint
                                  pick a directory that cannot be resumed from

Kept, always:

    evals/                        every results.yaml and .done marker
    config.json, tokenizer*, etc  kilobytes, and they document the cell
    metrics.jsonl, *_config.json  in the cell directory, never touched

SAFETY GUARDS  (all on by default; a checkpoint must pass EVERY one)
--------------------------------------------------------------------
  1. fully evaluated   -- every eval in --require has a .done marker
  2. not the newest    -- the highest-numbered checkpoint in a cell is always
                          kept, so a cell can still be resumed or extended
  3. tag not running   -- --exclude-running asks squeue, as the state pruner
                          does; a chained run resumes from its own checkpoints
  4. old enough        -- --min-age-hours, default 24
  5. --apply           -- without it this only reports

The first guard is the important one: this is irreversible, and the only thing
standing between "reclaim dead storage" and "throw away a run" is whether the
evals actually finished. A checkpoint whose eval crashed half way has no .done
marker and is therefore never touched.

USAGE
-----
    # report only
    python internal/uwiki/prune_checkpoint_weights.py \\
        --root $DATA/unlearning-pareto --root $DATA/decayed-root \\
        --exclude-running

    # then, having read the report
    python internal/uwiki/prune_checkpoint_weights.py \\
        --root $DATA/unlearning-pareto --root $DATA/decayed-root \\
        --exclude-running --apply

    # keep the two newest per cell rather than one
    ... --keep-newest 2
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict

WEIGHT_SUFFIXES = (".safetensors", ".bin")
STATE_FILE = "trainer_state.pt"

# The evals whose markers must all be present. These are the ones
# export_results.py reads; gaussian_watermark and the MIA are deliberately NOT
# required, because they are legitimately skipped on many cells (missing noise
# vectors, missing holdout) and requiring them would protect everything.
DEFAULT_REQUIRE = ("c4_perplexity", "fictional_knowledge", "insertion_likelihood")


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def squeue_tags():
    """Run-tag-ish tokens from every queued or running job name."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        sys.exit(f"ERROR: --exclude-running could not run squeue: {e}\n"
                 "       Refusing to continue: without it this would delete "
                 "checkpoints belonging to running jobs.")
    if out.returncode != 0:
        sys.exit(f"ERROR: squeue failed ({out.returncode}): {out.stderr.strip()}\n"
                 "       Refusing to continue for the same reason.")
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def tag_in_jobs(tag, jobs):
    """Is `tag` named by a job? Requires a boundary so 1B-tag != 1B-tagB."""
    for j in jobs:
        i = j.find(tag)
        while i >= 0:
            end = i + len(tag)
            if end == len(j) or j[end] in "-_./":
                return True
            i = j.find(tag, i + 1)
    return False


def step_number(name):
    for prefix in ("step-", "epoch-"):
        if name.startswith(prefix):
            try:
                return int(name[len(prefix):])
            except ValueError:
                return None
    return None


def find_checkpoints(root):
    """<root>/<tag>/<method>/<knob>-<value>/{step,epoch}-N/"""
    for tag in sorted(os.listdir(root)):
        tag_dir = os.path.join(root, tag)
        if tag == "anchors" or not os.path.isdir(tag_dir):
            continue
        for method in sorted(os.listdir(tag_dir)):
            m_dir = os.path.join(tag_dir, method)
            if not os.path.isdir(m_dir):
                continue
            for cell in sorted(os.listdir(m_dir)):
                c_dir = os.path.join(m_dir, cell)
                if not os.path.isdir(c_dir):
                    continue
                ckpts = []
                for entry in sorted(os.listdir(c_dir)):
                    n = step_number(entry)
                    if n is not None and os.path.isdir(os.path.join(c_dir, entry)):
                        ckpts.append((n, os.path.join(c_dir, entry)))
                if ckpts:
                    yield tag, method, cell, sorted(ckpts)


def weights_in(ckpt):
    out = []
    for name in os.listdir(ckpt):
        p = os.path.join(ckpt, name)
        if os.path.isfile(p) and (name.endswith(WEIGHT_SUFFIXES) or name == STATE_FILE):
            out.append(p)
    return out


def evaluated(ckpt, require):
    """Every required eval has written its .done marker beside evals/."""
    evals = os.path.join(ckpt, "evals")
    if not os.path.isdir(evals):
        return False, "no evals/ directory"
    missing = [e for e in require
               if not os.path.isfile(os.path.join(evals, f"{e}.done"))]
    if missing:
        return False, "missing marker(s): " + ", ".join(missing)
    return True, ""


def main():
    data = os.environ.get("PE_DATA") or os.environ.get("DATA") or os.path.expanduser("~")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="tree to scan; repeatable "
                         f"(default: {os.path.join(data, 'unlearning-pareto')})")
    ap.add_argument("--exclude", action="append", default=[],
                    help="run tag to protect; repeatable")
    ap.add_argument("--exclude-running", action="store_true",
                    help="protect every tag named by a queued or running job")
    ap.add_argument("--require", action="append", default=None,
                    help=f"eval that must have a .done marker; repeatable "
                         f"(default: {', '.join(DEFAULT_REQUIRE)})")
    ap.add_argument("--keep-newest", type=int, default=1,
                    help="checkpoints per cell to keep regardless (default: 1)")
    ap.add_argument("--min-age-hours", type=float, default=24.0,
                    help="never touch anything younger (default: 24)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without it this only reports")
    a = ap.parse_args()

    roots = a.root or [os.path.join(data, "unlearning-pareto")]
    roots = [r for r in roots if os.path.isdir(r)]
    if not roots:
        sys.exit("ERROR: no readable --root given")
    require = tuple(a.require) if a.require else DEFAULT_REQUIRE

    jobs = squeue_tags() if a.exclude_running else []
    if a.exclude_running:
        print(f"squeue: {len(jobs)} queued/running job(s) to protect against")
    print(f"roots   : {', '.join(roots)}")
    print(f"require : {', '.join(require)}")
    print(f"keep    : newest {a.keep_newest} per cell, nothing under "
          f"{a.min_age_hours:g}h\n")

    import time
    now = time.time()
    free_by_tag, kept_by_reason = defaultdict(lambda: [0, 0]), defaultdict(int)
    doomed = []

    for root in roots:
        for tag, method, cell, ckpts in find_checkpoints(root):
            if any(tag == e or tag.startswith(e.rstrip("*")) for e in a.exclude):
                kept_by_reason["explicitly excluded"] += len(ckpts)
                continue
            if jobs and tag_in_jobs(tag, jobs):
                kept_by_reason["run tag is in a queued/running job"] += len(ckpts)
                continue
            protected = {p for _, p in ckpts[-a.keep_newest:]} if a.keep_newest else set()
            for n, ckpt in ckpts:
                if ckpt in protected:
                    kept_by_reason[f"newest {a.keep_newest} in cell"] += 1
                    continue
                files = weights_in(ckpt)
                if not files:
                    kept_by_reason["no weight files (already pruned)"] += 1
                    continue
                age_h = (now - max(os.path.getmtime(f) for f in files)) / 3600
                if age_h < a.min_age_hours:
                    kept_by_reason[f"younger than {a.min_age_hours:g}h"] += 1
                    continue
                ok, why = evaluated(ckpt, require)
                if not ok:
                    kept_by_reason[f"not fully evaluated ({why})"] += 1
                    continue
                size = sum(os.path.getsize(f) for f in files)
                doomed.append((ckpt, files, size))
                free_by_tag[tag][0] += len(files)
                free_by_tag[tag][1] += size

    if free_by_tag:
        print(f"{'run tag':<42}{'files':>7}{'size':>14}")
        for tag in sorted(free_by_tag):
            cnt, size = free_by_tag[tag]
            print(f"{tag:<42}{cnt:>7}{human(size):>14}")
    total = sum(s for _, _, s in doomed)
    print(f"\n{'WOULD FREE' if not a.apply else 'FREED':<12}: {human(total)}"
          f"   ({len(doomed)} checkpoint(s))")
    for reason, n in sorted(kept_by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>6} kept because: {reason}")

    if not a.apply:
        print("\nReport only. Re-run with --apply to delete.")
        print("This is IRREVERSIBLE: regenerating a checkpoint means re-training "
              "the cell.")
        return 0

    removed = 0
    for ckpt, files, _ in doomed:
        for f in files:
            try:
                os.remove(f)
                removed += 1
            except OSError as e:
                print(f"  !! {f}: {e}", file=sys.stderr)
    print(f"\nremoved {removed} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
