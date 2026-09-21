"""Which checkpoints have a Gaussian-watermark result, and do the failures
correlate with the model having diverged?

The watermark differentiates the loss through the model, so a checkpoint whose
weights or logits have blown up yields NaN gradients -- and NaN is the correct
answer there, not a bug. If every failure sits at a high perplexity, nothing
needs fixing: those cells are past the utility cap and enter no figure. If
failures appear at healthy perplexities, something else is wrong.

Run on LUMI from the repo root:
    python internal/lumi/survey_watermark.py
    python internal/lumi/survey_watermark.py --root <dir>   # one root only
"""
import argparse
import glob
import os

W = "/scratch/project_465003383/unlearning_baselines"


def c4_of(step_dir):
    """Raw c4 perplexity for a checkpoint, or None if it was never measured."""
    y = os.path.join(step_dir, "evals", "c4_perplexity", "results.yaml")
    if not os.path.isfile(y):
        return None
    try:
        import yaml
        d = yaml.safe_load(open(y)) or {}
    except Exception:
        return None
    for k in ("perplexity", "ppl", "c4_perplexity"):
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    return None


def survey(root):
    rows = []
    for step_dir in sorted(glob.glob(os.path.join(root, "*", "*", "*", "step-*"))):
        ev = os.path.join(step_dir, "evals")
        gw = os.path.join(ev, "gaussian_watermark")
        marker = os.path.join(ev, "gaussian_watermark.done")
        pts = glob.glob(os.path.join(gw, "*.pt"))
        rel = os.path.relpath(step_dir, root)
        rows.append({
            "cell": rel,
            "tag": rel.split(os.sep)[0],
            "has_pt": bool(pts),
            "marked": os.path.isfile(marker),
            "c4": c4_of(step_dir),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", dest="roots")
    a = ap.parse_args()
    roots = a.roots or [f"{W}/decayed-root", f"{W}/unlearning-pareto-2.7B"]

    for root in roots:
        rows = survey(root)
        if not rows:
            print(f"=== {root}: nothing found\n")
            continue
        ok = [r for r in rows if r["has_pt"]]
        bad = [r for r in rows if not r["has_pt"]]
        print(f"=== {os.path.basename(root)}: {len(ok)}/{len(rows)} have watermark results")

        by_tag = {}
        for r in rows:
            t = by_tag.setdefault(r["tag"], [0, 0])
            t[0] += 1
            t[1] += 1 if r["has_pt"] else 0
        for t, (n, k) in sorted(by_tag.items()):
            flag = "" if k == n else "   <-- incomplete"
            print(f"    {t:34s} {k:3d}/{n:3d}{flag}")

        # the question that matters: are the failures the diverged checkpoints?
        okc = [r["c4"] for r in ok if r["c4"] is not None]
        badc = [r["c4"] for r in bad if r["c4"] is not None]
        if okc and badc:
            okc.sort(); badc.sort()
            med = lambda v: v[len(v) // 2]
            print()
            print(f"    c4 perplexity, succeeded : n={len(okc):3d} "
                  f"min {min(okc):9.2f}  median {med(okc):9.2f}  max {max(okc):12.2f}")
            print(f"    c4 perplexity, failed    : n={len(badc):3d} "
                  f"min {min(badc):9.2f}  median {med(badc):9.2f}  max {max(badc):12.2f}")
            healthy = [r for r in bad if r["c4"] is not None and r["c4"] < 25]
            if healthy:
                print(f"    {len(healthy)} FAILED at a healthy perplexity (<25) "
                      f"-- these are NOT explained by divergence:")
                for r in healthy[:10]:
                    print(f"        {r['cell']:58s} c4={r['c4']:.2f}")
            else:
                print("    every failure is at a high perplexity: consistent with "
                      "the model having diverged, which makes NaN the correct answer.")
        marked_no_pt = [r for r in rows if r["marked"] and not r["has_pt"]]
        if marked_no_pt:
            print(f"    {len(marked_no_pt)} marked done but wrote nothing "
                  f"(pre-guard markers -- delete and re-run)")
        print()


if __name__ == "__main__":
    main()
