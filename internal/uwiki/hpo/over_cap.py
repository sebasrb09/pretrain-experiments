"""Print 1 if a rung's C4 perplexity is over the utility cap, else 0.

Used by hpo_trial.sh between rungs to decide whether to abandon the rest of the
ladder. A separate file rather than an inline heredoc because the trial script
already nests two scripts and a heredoc inside a heredoc is how backslashes get
eaten.

Reads both perplexities through export_results.py's own reader: the rung's, and
the baseline anchor's under ANCHOR_ROOT, which was measured on the same C4 file.
Any failure to read prints 0: evaluating one rung too many costs minutes,
stopping a good trial by mistake costs the trial.
"""
import importlib.util
import os
import sys


def main():
    if len(sys.argv) != 2:
        print(0)
        return 0
    eval_dir = sys.argv[1]
    try:
        exporter = os.path.join(os.environ["REPO"], "internal", "uwiki", "export_results.py")
        spec = importlib.util.spec_from_file_location("_ex", exporter)
        ex = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ex)
        c4 = ex.scalar(eval_dir, "c4_perplexity", "perplexity")
        base = ex.scalar(os.path.join(os.environ["ANCHOR_ROOT"], "baseline", "step-0"),
                         "c4_perplexity", "perplexity")
        cap = float(os.environ["UTIL_CAP_PCT"])
        c4, base = float(c4), float(base)
    except BaseException:
        print(0)
        return 0
    delta = 100.0 * (c4 - base) / base
    print(1 if delta > cap else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
