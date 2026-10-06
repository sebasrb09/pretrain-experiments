"""Print 1 if a rung's C4 perplexity is over the utility cap, else 0.

Used by hpo_trial.sh between rungs to decide whether to abandon the rest of the
ladder. A separate file rather than an inline heredoc because the trial script
already nests two scripts and a heredoc inside a heredoc is how backslashes get
eaten.

Reads the perplexity through export_results.py's own reader, so the number the
early stop acts on is the number the exporter would record.
"""
import importlib.util
import os
import sys


def main():
    if len(sys.argv) != 2:
        print(0)
        return 0
    eval_dir = sys.argv[1]
    repo = os.environ.get("REPO", "")
    exporter = os.path.join(repo, "internal", "uwiki", "export_results.py")
    if not os.path.exists(exporter):
        # Cannot read the result, so do not stop. Evaluating one rung too many
        # costs an hour; stopping a good trial by mistake costs the trial.
        print(0)
        return 0

    # export_results.py exits hard if PyYAML is absent, and any import error
    # here must not propagate: failing to read means "do not stop", never
    # "stop". The caller also has `|| over=0`, so this is belt and braces.
    try:
        spec = importlib.util.spec_from_file_location("_ex", exporter)
        ex = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ex)
        c4 = ex.scalar(eval_dir, "c4_perplexity", "perplexity")
    except BaseException:
        print(0)
        return 0
    if c4 is None:
        # The rung did not land. That is not a reason to stop: summarize_trial.py
        # will simply leave it out of the trajectory.
        print(0)
        return 0

    # Baseline anchor, not the 19.71 cap; the driver passes it explicitly.
    base = float(os.environ.get("BASE_C4_PPL", "18.7734"))
    cap = float(os.environ.get("UTIL_CAP_PCT", "5.0"))
    delta = 100.0 * (float(c4) - base) / base
    print(1 if delta > cap else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
