#!/usr/bin/env python3
"""Reclaim disk by deleting trainer_state.pt from cells that no longer need it.

WHY
---
Adam's two moment buffers for 1.48B fp32 parameters are ~11-18 GB, four to six
times the ~3 GB of weights beside them. On MUSICA they measured 4 505 GB across
414 files -- about half of a 9.8 TB filesystem, which hit 100% full.

NOTHING in the eval or aggregation path reads them: the eval scripts load the HF
checkpoint, export_results.py takes the step from the directory name, and
aggregate_pareto.py falls back to that same name (which for a step-N checkpoint
IS the optimizer step). Their only purpose is `--auto-resume`.

So a trainer_state.pt is dead weight unless its cell might still be resumed.

WHAT IT KEEPS
-------------
Deleting the wrong one costs a chained multi-day run its restart point, so the
default policy is deliberately conservative:

  * keep the newest checkpoint's state in every cell -- that is the only one
    find_latest_checkpoint would ever resume from, so keeping it preserves
    resumability everywhere while still reclaiming the bulk
  * never touch a file younger than --min-age-hours (default 24), which is the
    proxy for "a job is writing here right now"
  * never touch a cell whose run tag matches --exclude
  * with --exclude-running, never touch a cell whose run tag appears in any
    queued or running job name

--exclude-running closes a gap the age guard does NOT cover. A chained run
resumes from its LAST checkpoint, which may have been written days ago: the
satimp 200->500 run restarts from a step-200 trainer_state.pt that is old
enough to look prunable while being precisely the file the next chain link
needs. Age tells you what is being written, not what is about to be read.

--all drops the keep-newest rule for cells you know are finished. Use it with
--exclude for anything still chaining.

USAGE
-----
    # look first; this is the default and it writes nothing
    python internal/uwiki/prune_trainer_state.py

    # a second tree, and protect the run that is still chaining
    python internal/uwiki/prune_trainer_state.py \
        --root $DATA/unlearning-pareto --root $DATA/decayed-root \
        --exclude '1B-p2-satimp-rt*'

    python internal/uwiki/prune_trainer_state.py --apply
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import time

STATE = "trainer_state.pt"
STEP_RE = re.compile(r"^step-(\d+)$")


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} PB"


def find_states(roots):
    """Yield (root, run_tag, cell_dir, ckpt_name, path, size, mtime)."""
    for root in roots:
        if not os.path.isdir(root):
            print(f"note: no such root, skipping: {root}", file=sys.stderr)
            continue
        # <root>/<run_tag>/<method>/<knob-value>/<ckpt>/trainer_state.pt
        for run_tag in sorted(os.listdir(root)):
            tag_dir = os.path.join(root, run_tag)
            if not os.path.isdir(tag_dir) or run_tag == "anchors":
                continue
            for method in sorted(os.listdir(tag_dir)):
                m_dir = os.path.join(tag_dir, method)
                if not os.path.isdir(m_dir):
                    continue
                for cell in sorted(os.listdir(m_dir)):
                    cell_dir = os.path.join(m_dir, cell)
                    if not os.path.isdir(cell_dir):
                        continue
                    for ckpt in sorted(os.listdir(cell_dir)):
                        p = os.path.join(cell_dir, ckpt, STATE)
                        if os.path.isfile(p):
                            st = os.stat(p)
                            yield (root, run_tag, cell_dir, ckpt, p,
                                   st.st_size, st.st_mtime)


def newest_ckpt(names):
    """The checkpoint a resume would pick: highest step-N.

    An epoch-N alongside step-N checkpoints is an end-of-run duplicate, but its
    ordering against them is not readable from the name, so any cell containing
    one keeps that state too rather than risk removing a live resume point.
    """
    steps = [(int(m.group(1)), n) for n in names
             if (m := STEP_RE.match(n)) is not None]
    keep = set()
    if steps:
        keep.add(max(steps)[1])
    keep.update(n for n in names if not STEP_RE.match(n))
    return keep


_BOUNDARY = "-_./"


def tag_in_jobs(tag: str, jobs: list[str]) -> bool:
    """Does `tag` appear in any job name as a whole component?

    Plain substring containment would spare a FINISHED sweep because a longer
    tag containing it happens to be running -- 1B-tag protected by 1B-tagB --
    which costs reclaimed space for no safety gain.

    The distinguishing character always follows the tag: both launchers embed
    it as `<RUN_TAG>-<method>-...`, so a genuine hit is followed by '-' or by
    end-of-string, while 1B-tag inside 1B-tagB is followed by 'B'. Requiring
    that end boundary removes the false positive and cannot narrow a real
    match. No boundary is required BEFORE the tag, because prefixes vary
    ("pe-" on eval jobs, none on training jobs) and demanding one there would
    trade a safe over-match for an unsafe under-match.
    """
    for job in jobs:
        start = 0
        while True:
            i = job.find(tag, start)
            if i < 0:
                break
            end = i + len(tag)
            if end == len(job) or job[end] in _BOUNDARY:
                return True
            start = i + 1
    return False


def queued_job_names() -> list[str]:
    """Job names of everything this user has queued or running, via squeue.

    Matching is substring containment of a run tag in a job name, so it works
    whatever prefix the launchers use (training and eval jobs name themselves
    differently). That over-matches rather than under-matches, which is the
    right direction for a tool whose mistakes are irreversible.

    Any failure is fatal rather than silent: an empty list would mean "nothing
    is running", which is the most dangerous possible wrong answer here.
    """
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    cmd = ["squeue", "-h", "-o", "%j"] + (["-u", user] if user else [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(
            f"ERROR: --exclude-running needs squeue and it failed: {exc}\n"
            "       Run this on a login node, or drop the flag and name the "
            "live runs with --exclude."
        )
    if out.returncode != 0:
        raise SystemExit(
            f"ERROR: squeue exited {out.returncode}: {out.stderr.strip()}\n"
            "       Refusing to continue: I cannot tell which runs are live."
        )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    data = os.environ.get("PE_DATA") or os.environ.get("DATA") or os.path.expanduser("~")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="tree to scan; repeatable "
                         f"(default: {os.path.join(data, 'unlearning-pareto')})")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob on run tag; repeatable. Protects chaining runs.")
    ap.add_argument("--exclude-running", action="store_true",
                    help="also protect every run tag that appears in a queued "
                         "or running job name (asks squeue). Covers chained "
                         "runs whose resume checkpoint is older than "
                         "--min-age-hours.")
    ap.add_argument("--all", action="store_true",
                    help="also delete the newest state in each cell "
                         "(only for cells you know are finished)")
    ap.add_argument("--min-age-hours", type=float, default=24.0,
                    help="never touch a file younger than this (default: 24)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without it this only reports")
    args = ap.parse_args()

    roots = args.root or [os.path.join(data, "unlearning-pareto")]
    cutoff = time.time() - args.min_age_hours * 3600.0

    live_jobs: list[str] = []
    if args.exclude_running:
        live_jobs = queued_job_names()
        print(f"squeue: {len(live_jobs)} queued/running jobs to protect against")

    by_cell: dict[str, list] = {}
    for rec in find_states(roots):
        by_cell.setdefault(rec[2], []).append(rec)

    if not by_cell:
        print("no trainer_state.pt found under: " + ", ".join(roots))
        return 0

    to_delete, kept = [], []
    reasons: dict[str, int] = {}

    def hold(why):
        reasons[why] = reasons.get(why, 0) + 1

    for cell_dir, recs in sorted(by_cell.items()):
        names = [r[3] for r in recs]
        keep_names = set() if args.all else newest_ckpt(names)
        for (_root, run_tag, _cd, ckpt, path, size, mtime) in recs:
            if any(fnmatch.fnmatch(run_tag, pat) for pat in args.exclude):
                kept.append((path, size)); hold("excluded by --exclude"); continue
            if tag_in_jobs(run_tag, live_jobs):
                kept.append((path, size)); hold("run tag is in a queued/running job"); continue
            if mtime > cutoff:
                kept.append((path, size)); hold(f"younger than {args.min_age_hours} h"); continue
            if ckpt in keep_names:
                kept.append((path, size)); hold("newest in its cell (resume point)"); continue
            to_delete.append((path, size))

    del_bytes = sum(s for _, s in to_delete)
    keep_bytes = sum(s for _, s in kept)

    per_tag: dict[str, list] = {}
    for path, size in to_delete:
        rel = path
        for root in roots:
            if path.startswith(root):
                rel = os.path.relpath(path, root)
                break
        per_tag.setdefault(rel.split(os.sep)[0], [0, 0])
        per_tag[rel.split(os.sep)[0]][0] += 1
        per_tag[rel.split(os.sep)[0]][1] += size

    print(f"roots   : {', '.join(roots)}")
    print(f"cells   : {len(by_cell)}")
    print(f"files   : {len(to_delete) + len(kept)}  "
          f"({len(to_delete)} to delete, {len(kept)} kept)")
    print()
    if per_tag:
        print(f"{'run tag':<40} {'files':>6} {'size':>12}")
        for tag, (n, b) in sorted(per_tag.items(), key=lambda kv: -kv[1][1]):
            print(f"{tag:<40} {n:>6} {human(b):>12}")
        print()
    print(f"WOULD FREE : {human(del_bytes)}")
    print(f"kept       : {human(keep_bytes)}")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>4} kept because: {why}")

    if not to_delete:
        return 0
    if not args.apply:
        print("\ndry run -- nothing deleted. Re-run with --apply.")
        return 0

    freed, failed = 0, 0
    for path, size in to_delete:
        try:
            os.unlink(path)
            freed += size
        except OSError as exc:
            print(f"  ! {path}: {exc}", file=sys.stderr)
            failed += 1
    print(f"\ndeleted {len(to_delete) - failed} files, freed {human(freed)}")
    if failed:
        print(f"{failed} could not be removed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
