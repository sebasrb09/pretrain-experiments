"""The search objective: forgetting on three tasks, each measured against its own anchors.

For each task t, with measurement x_t at a checkpoint:

    p_t = clip( (s_t(x_t) - s_t(B_t)) / (s_t(C_t) - s_t(B_t)), 0, 1 )
    F   = (p_fk + p_il + p_wm) / 3

    B_t   the BASELINE anchor: the model every method starts from
    C_t   the COUNTERFACTUAL anchor (deep-ignorance): same corpus, targets removed
    s_t   the scale the task is compared on
            fk  log10 of the fictional-knowledge probability  (spans ~5 decades)
            il  insertion-likelihood perplexity, linear       (spans ~3x)
            wm  signed Gaussian-watermark Q4 score, linear    (a test statistic)

p_t = 0 means nothing forgotten and 1 means the counterfactual was reached, so
every task is in the same unit: the fraction of the way a perfect unlearner
travels. These are export_results.py's fk_forgot, il_forgot and wm_removed,
clipped to [0, 1].

Why each choice, because each one decides how much a task counts:

  * Anchor-gap scaling, not z-scores across trials. Standardising by the spread
    of the trials would hand the watermark, which barely moves, the same weight
    as the tasks that do, and would amplify its noise. The anchor gap is fixed
    before the search starts, so a task's weight cannot depend on the trials.
  * log10 for fk. Linear would score a 10x drop in probability as 90% forgotten
    while the model still sits 10,000x above the counterfactual, and that one
    task would carry ~4x the weight of insertion. In log space the two carry
    comparable weight (sd of their terms across the 1B sweep: 0.016 vs 0.021).
  * Clipping. Past the counterfactual (a broken model drives fk to ~1e-78) earns
    nothing extra, so no task can be gamed by destroying the model, and moving
    away from the counterfactual cannot cancel progress on another task.
  * Equal weights. Each task counts once, as in the paper's seven-task figure.

The other tasks are held out and never seen by the search: verbatim,
contamination, prompt extraction and DoS have anchor gaps too small (or of the
wrong sign) to normalise against at 1B, and nothing moves MIA within budget.

ANCHORS ARE MEASURED WITH THE TRIALS' EXACT EVALUATION SETTINGS. An anchor from
another site or another batch size is a different measurement: the 1B LUMI
export carried an ASC baseline for insertion (3.603) while every untouched LUMI
checkpoint reads ~12.9, which would credit every trial with 29% insertion
forgetting before a single step. settings.json pins what the anchors were
measured with, and check_settings() refuses a trial that differs.

    python forget_score.py show  --anchor-root DIR   print the anchors and gaps
    python forget_score.py check --anchor-root DIR   exit 1 unless the current
                                                     environment matches them

THE SEARCH FRONTIER (2026-10-14). A wm search keeps the checkpoints on its
frontier, C4 perplexity against Gaussian-poison detectability, so the full suite
can run on them later without retraining; the others are deleted as before.

    python forget_score.py keep  --output-root R --run-tag T --method M --cell-dir D --step N
        (hpo_trial.sh, after scoring a rung) exit 0: keep step-N, 3: delete it
    python forget_score.py prune --output-root R --study S --method M [--apply]
        delete the kept checkpoints of finished trials that are dominated now

THE FULL-SUITE PLAN (2026-10-15). Which search checkpoints get the full suite:

    python forget_score.py frontier --export E [--sweep DIR[:ANCHORS],...] --tol PTS --out PLAN.csv
        from a pe_export folder; optuna_hpo.py --frontier PLAN.csv evaluates them
"""
import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from statistics import NormalDist

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORTER = os.path.join(HERE, os.pardir, "export_results.py")

TASKS = ("fk", "il", "wm")
# What a search maximises, by name. Every study records its objective, and the
# driver refuses to mix them.
OBJECTIVES = {
    # mean progress on knowledge, insertion and the Gaussian poison (the f3
    # searches of 2026-10-07/08)
    "f3": ("fk", "il", "wm"),
    # the Gaussian poison alone (decided 2026-10-09): the one task whose removal
    # is a calibrated statistical test, so unlearning is told apart from noise
    "wm": ("wm",),
    # the Gaussian poison over the figures' whole perplexity range (2026-10-15):
    # the trial's value is trajectory_area(), not its best point, and its rungs
    # are scored up to C4 33 instead of +5%
    "wmarea": ("wm",),
}
# The utility cap each objective's trials stop at, in % over the Baseline's C4
# on the search's slice. wmarea's is the figures' x range, C4 33 on the reporting
# slice (Baseline 18.77 there): 33 / 18.77 - 1 = 75.8%.
CAP_PCT = {"f3": "5.0", "wm": "5.0", "wmarea": "75.8"}
POINTS = ("baseline", "deep-ignorance")

# The variables that decide what an evaluation measures. The anchors and every
# trial must agree on all of them, plus the content of the C4 file.
EVAL_KEYS = ("NOISE_DIR", "NOISE_STD", "EVAL_MAX_NUM_SEQS", "INFERENCE_MAX_NUM_SEQS",
             "C4_TASK_FILE", "IL_EXPERIMENT", "IL_MAX_TOKENS")

# Minimum anchor separation per task, in the task's own scale. Below these the
# anchors do not separate and p_t would be noise divided by noise.
MIN_GAP = {"fk": 1.0, "il": 1.0, "wm": 0.5}


class AnchorError(RuntimeError):
    pass


def load_exporter():
    spec = importlib.util.spec_from_file_location("_export_results", EXPORTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _num(v):
    """A float, or None for anything missing or NaN. +inf is kept: it is how a
    destroyed model's perplexity overflows, and it is still a measurement."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def measure(eval_dir, il_experiment, ex=None):
    """The four raw numbers a rung contributes, read with the exporter's readers."""
    ex = ex or load_exporter()
    _, wm_q4 = ex.watermark(eval_dir)
    return {
        "c4": _num(ex.scalar(eval_dir, "c4_perplexity", "perplexity")),
        "fk": _num(ex.scalar(eval_dir, "fictional_knowledge", "probability")),
        "il": _num(ex.insertion_likelihood(eval_dir, il_experiment)),
        "wm": _num(wm_q4),
    }


def _scale(task, x):
    if task == "fk":
        return math.log10(x) if x > 0 else -math.inf
    return x


def progress(task, x, anchors):
    """p_t for one measurement, clipped to [0, 1]."""
    b = _scale(task, anchors["baseline"][task])
    c = _scale(task, anchors["deep-ignorance"][task])
    s = _scale(task, x)
    if math.isinf(s):
        # fk probability of exactly 0, or an overflowed perplexity: further than
        # the counterfactual in the direction of forgetting.
        v = 1.0 if (s - b) * (c - b) > 0 else 0.0
    else:
        v = (s - b) / (c - b)
    return min(1.0, max(0.0, v))


def score(m, anchors, objective="f3"):
    """p_<task> for every measured task, and F, the mean over the objective's
    tasks; None if one of the objective's tasks is missing."""
    tasks = OBJECTIVES[objective]
    if any(m.get(t) is None for t in tasks):
        return None
    out = {f"p_{t}": progress(t, m[t], anchors) for t in TASKS if m.get(t) is not None}
    out["F"] = sum(out[f"p_{t}"] for t in tasks) / len(tasks)
    return out


def fingerprint(env):
    """What the anchors were, or a trial is, measured with."""
    missing = [k for k in EVAL_KEYS if not env.get(k)]
    if missing:
        raise AnchorError("evaluation settings not set: " + " ".join(missing))
    fp = {k: env[k] for k in EVAL_KEYS}
    with open(env["C4_TASK_FILE"], "rb") as fh:
        fp["c4_task_file_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    return fp


def write_settings(root, env, models):
    os.makedirs(root, exist_ok=True)
    blob = {"eval": fingerprint(env), "models": models}
    path = os.path.join(root, "settings.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2, sort_keys=True)
    return path


def read_settings(root):
    path = os.path.join(root, "settings.json")
    if not os.path.exists(path):
        raise AnchorError(f"no {path}: the anchors were never submitted (optuna_hpo.py --anchors)")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_settings(root, env):
    """Raise unless env evaluates exactly the way the anchors were evaluated."""
    want = read_settings(root)["eval"]
    have = fingerprint(env)
    diff = {k: (want.get(k), have.get(k)) for k in sorted(set(want) | set(have))
            if want.get(k) != have.get(k)}
    if diff:
        lines = [f"  {k}: anchors {a!r}, this run {b!r}" for k, (a, b) in diff.items()]
        raise AnchorError("evaluation settings differ from the anchors':\n" + "\n".join(lines))


def load_anchors(root, il_experiment, ex=None):
    """Both anchors' measurements, validated. Raises AnchorError with the reason."""
    try:
        import torch   # noqa: F401  the exporter reads the watermark .pt files with it
    except ImportError:
        # export_results.watermark() returns None without torch, which would
        # surface as "no finite wm" and hide the real cause.
        raise AnchorError("torch is not importable in this Python, so the watermark scores "
                          "cannot be read. Run inside the container: source internal/lumi/env.sh")
    ex = ex or load_exporter()
    out = {}
    for pt in POINTS:
        d = os.path.join(root, pt, "step-0")
        m = measure(d, il_experiment, ex)
        bad = [k for k, v in m.items() if v is None or math.isinf(v)]
        if bad:
            raise AnchorError(f"anchor {pt} has no finite {', '.join(bad)} under {d}")
        out[pt] = m
    b, c = out["baseline"], out["deep-ignorance"]
    if b["fk"] <= 0 or c["fk"] <= 0:
        raise AnchorError(f"fk anchors must be positive (baseline {b['fk']}, counterfactual {c['fk']})")
    gaps = {
        "fk": math.log10(b["fk"]) - math.log10(c["fk"]),   # baseline knows more
        "il": c["il"] - b["il"],                           # counterfactual is more surprised
        "wm": c["wm"] - b["wm"],                           # baseline is more negative
    }
    for t, g in gaps.items():
        if g < MIN_GAP[t]:
            raise AnchorError(f"anchors do not separate on {t}: gap {g:.4g} in its own scale, "
                              f"need at least {MIN_GAP[t]}")
    out["gaps"] = gaps
    return out


# ------------------------------------------------------------ the search frontier
# The points are (C4 perplexity, poison TPR at 1% FPR). A point is dominated when
# another is no worse on both and better on one; the frontier is what is left.
# A rung is kept unless a point known when it is scored dominates it, so every
# point of the final frontier is kept (nothing dominates it at any time); points
# that only later become dominated are removed by prune. The comparison is within
# the study: the grid sweep measured C4 on another file, so its points are not
# on the same axis, and the search's own frontier contains every search point of
# the combined one.
KEEP_MAX_C4 = 33.0      # the figures' x range; beyond it nothing is drawn or evaluated
Z99 = NormalDist().inv_cdf(0.99)


def poison_tpr(wm_q4):
    """Detectability of the Gaussian poison: TPR at 1% FPR of the unit-variance
    watermark score shifted by wm_q4 (notebooks/pareto_tasks.py tpr_at_1pct_fpr)."""
    return 1.0 - NormalDist().cdf(Z99 - abs(wm_q4))


def trajectory_area(points, anchors, cap_pct):
    """The wmarea objective: how far one trial's trajectory lowers the poison
    TPR across the whole perplexity range, in [0, 1].

    The attainment curve f(c) is the lowest TPR among the trial's points with a
    C4 increase of at most c (the Baseline's TPR before its first point). The
    value is the mean over c in [0, cap_pct] of (TPR_baseline - f(c)) /
    (TPR_baseline - TPR_counterfactual), clipped to [0, 1]: the area between
    the Baseline and the trial's curve in the plotted plane, normalised so that
    1 is the counterfactual's TPR from the first step on. It rewards a
    configuration that lowers the TPR early and keeps lowering it, which is
    what the frontier in the figures is made of, instead of one point at +5%.
    points: dicts with c4_delta_pct and wm_q4 (summarize_trial's points)."""
    tb = poison_tpr(anchors["baseline"]["wm"])
    tc = poison_tpr(anchors["deep-ignorance"]["wm"])
    pts = sorted((max(0.0, p["c4_delta_pct"]), poison_tpr(p["wm_q4"])) for p in points
                 if p.get("c4_delta_pct") is not None and p.get("wm_q4") is not None
                 and p["c4_delta_pct"] <= cap_pct)
    area, best, at = 0.0, tb, 0.0
    for c, t in pts:
        area += (c - at) * (tb - best)
        at, best = c, min(best, t)
    area += (cap_pct - at) * (tb - best)
    return min(1.0, max(0.0, area / (cap_pct * (tb - tc))))


def dominated(p, others, tol=0.0):
    """tol (TPR, a fraction): a point with C4 no higher whose TPR is at most tol
    higher also removes p, so only points that lower the TPR by more than tol
    survive. 0 is plain Pareto dominance. Two points never remove each other."""
    c, t = p
    return any(c2 <= c and t2 <= t + tol and (c2 < c or t2 < t) for c2, t2 in others)


def study_name(run_tag):
    """hpo-npo-wm-t0012 -> hpo-npo-wm"""
    return re.sub(r"-t\d+$", "", run_tag)


def _result_points(path, max_c4):
    """{step: (c4, tpr)} of one finished trial's hpo_result.json."""
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    out = {}
    for p in res.get("points", []):
        c4, wm = _num(p.get("c4_ppl")), _num(p.get("wm_q4"))
        if c4 is not None and wm is not None and c4 <= max_c4:
            out[int(p["step"])] = (c4, poison_tpr(wm))
    return out


def finished_trials(output_root, study, method):
    """{cell dir: hpo_result.json path} of every finished trial of the study."""
    return {os.path.dirname(p): p for p in
            glob.glob(os.path.join(output_root, f"{study}-t[0-9]*", method, "*", "hpo_result.json"))}


def _c4_tpr(eval_dir, ex):
    """(c4, tpr) of a rung scored in this job, or None. Reads the watermark .pt (torch)."""
    c4 = _num(ex.scalar(eval_dir, "c4_perplexity", "perplexity"))
    _, wm = ex.watermark(eval_dir)
    wm = _num(wm)
    return None if c4 is None or wm is None else (c4, poison_tpr(wm))


def keep_decision(output_root, run_tag, method, cell_dir, step, max_c4=KEEP_MAX_C4, ex=None, tol=0.0):
    """(keep, why) for step-<step> of a running trial, against the finished
    trials of its study and this trial's own earlier rungs."""
    ex = ex or load_exporter()
    me = _c4_tpr(os.path.join(cell_dir, "evals", f"step-{step}"), ex)
    if me is None:
        return False, "no C4 or watermark result"
    if me[0] > max_c4:
        return False, f"C4 {me[0]:.2f} beyond {max_c4:g}"
    others = []
    for cell, path in finished_trials(output_root, study_name(run_tag), method).items():
        if os.path.abspath(cell) != os.path.abspath(cell_dir):
            others += _result_points(path, max_c4).values()
    for d in glob.glob(os.path.join(cell_dir, "evals", "step-*")):
        s = os.path.basename(d)[len("step-"):]
        if s.isdigit() and int(s) != int(step):
            p = _c4_tpr(d, ex)
            if p is not None and p[0] <= max_c4:
                others.append(p)
    why = f"C4 {me[0]:.3f}, poison TPR {100 * me[1]:.2f}%, against {len(others)} points"
    return (not dominated(me, others, tol)), why


def prune(output_root, study, method, apply=False, max_c4=KEEP_MAX_C4, log=print, tol=0.0):
    """Delete (apply=True) the kept checkpoints of finished trials that the
    study's finished points now dominate, or that were never scored. Trials still
    running are left alone. Returns (kept, deleted) as lists of directories."""
    trials = finished_trials(output_root, study, method)
    pts = {cell: _result_points(path, max_c4) for cell, path in trials.items()}
    allp = [(cell, s, p) for cell, d in pts.items() for s, p in d.items()]
    kept, gone = [], []
    for cell in sorted(trials):
        for d in sorted(glob.glob(os.path.join(cell, "step-*"))):
            s = os.path.basename(d)[len("step-"):]
            if not (s.isdigit() and os.path.isdir(d)):
                continue
            p = pts[cell].get(int(s))
            if p is not None and not dominated(p, [q for c, t, q in allp if (c, t) != (cell, int(s))], tol):
                kept.append(d)
                continue
            gone.append(d)
            log(f"  {'delete' if apply else 'would delete'} {d}: "
                + ("not scored or beyond the C4 range" if p is None else
                   f"dominated (C4 {p[0]:.3f}, TPR {100 * p[1]:.2f}%)"))
            if apply:
                shutil.rmtree(d, ignore_errors=True)
    return kept, gone


# The full suite costs ~18 GCD-hours a checkpoint, so it does not run on every
# point of every search frontier (2026-10-15, user and supervisor): only on the
# search checkpoints that beat the grid sweep, thinned so that each lowers the
# poison TPR by more than `tol` points below every point, search or sweep, with
# lower perplexity. C4 is compared as the increase over each side's own
# Baseline: the search measured it on another slice of C4 (Baseline 18.81
# there, 18.77 on the reporting slice). Read from a pe_export folder, so it runs
# off the cluster; optuna_hpo.py --frontier evaluates exactly the rows.
PLAN_FIELDS = ["method", "study", "trial", "step", "c4_rel_pct", "tpr_pct", "on_disk"]


def _csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _baseline_c4(anchors_csv):
    for r in _csv(anchors_csv):
        if r.get("point") == "baseline" and _num(r.get("c4_ppl")):
            return _num(r["c4_ppl"])
    raise SystemExit(f"no Baseline C4 in {anchors_csv}")


def thin(points, eps, start):
    """Greedy along perplexity: a point stays when its TPR is more than eps below
    the best TPR kept so far, starting from the Baseline's. eps 0: the Pareto
    frontier. Points are (C4 increase, TPR, ...), both in percent."""
    out, best = [], start
    for p in sorted(points, key=lambda p: (p[0], p[1])):
        if p[1] < best - eps:
            out.append(p)
            best = p[1]
    return out


def frontier_trials(output_root, study, method, tol=0.25, max_c4=KEEP_MAX_C4, n=12):
    """Trial numbers of a finished study with a point on its own frontier (C4
    against poison TPR, thinned at tol TPR points), at most n of them spread
    along it in perplexity: the warm start of a later study (optuna_hpo.py
    --warm-start). Read from the trials' hpo_result.json on the cluster."""
    pts, start = [], None
    for cell, path in finished_trials(output_root, study, method).items():
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
        m = re.search(r"-t(\d+)$", os.path.basename(os.path.dirname(os.path.dirname(cell))))   # <tag>/<method>/<cell>
        if not m or not res.get("anchors"):
            continue
        start = 100 * poison_tpr(res["anchors"]["baseline"]["wm"])
        for step, (c4, t) in _result_points(path, max_c4).items():
            pts.append((c4, 100 * t, int(m.group(1))))
    if start is None:
        return []
    order = []
    for _, _, no in thin(pts, tol, start):
        if no not in order:
            order.append(no)
    if len(order) > n:
        order = [order[round(i * (len(order) - 1) / (n - 1))] for i in range(n)]
    return order


class _Recorder:
    """Stands in for an Optuna trial to read a space's bounds off spaces.suggest."""
    def __init__(self):
        self.d = {}

    def suggest_float(self, name, lo, hi, log=False):
        if name.endswith("_logit"):
            sig = lambda u: float(f"{1.0 / (1.0 + math.exp(-u)):.12g}")   # 0.999, not 0.99899999...
            self.d[name[:-len("_logit")]] = ("logit", sig(lo), sig(hi))
        else:
            self.d[name] = ("log" if log else "lin", lo, hi)
        return lo

    def suggest_categorical(self, name, choices):
        self.d[name] = ("cat", list(choices))
        return choices[0]


# Stage 2's space, read off stage 1 (2026-10-15, user: "we already learned a lot
# from stage 1 ... as long as we move through the same Pareto front"). See
# derive_space. Where the stage-1 frontier presses on an edge, the range may
# open up to these outer limits (user's choice, 2026-10-15); everywhere else it
# stays inside the stage-1 space.
SPACE_OUTER = {("wga", "beta1"): (None, 500.0), ("ce-u", "min_forget_ce"): (None, 0.1)}
SMALL_BATCHES = [32, 64]


def derive_space(study_dir, top=0.25, min_trials=8, margin=0.15, cap_pct=75.8):
    """The stage-2 space of one method from its stage-1 export (report.py --out folder).

    Every stage-1 trial is scored with the stage-2 objective, trajectory_area over
    what it measured (its curve held flat past its last rung, so a lower bound),
    and the best `top` fraction of them, at least min_trials, are the evidence:
    a point "near the frontier" would not do, since every trial starts next to the
    Baseline, where the frontier starts too. Each hyperparameter is bounded by
    what those trials span, widened by `margin` of the stage-1 width on each side,
    in the scale the search samples (log, logit or linear), and clipped to the
    stage-1 space, except:
      - the learning rate keeps the stage-1 ceiling: stage 2 must reach C4 33,
        further than the +5% stage 1 was scored on, and step size is what gets there
      - a range that reaches an edge with an outer limit (SPACE_OUTER) opens to it
      - batch: the sizes those trials used, plus 32 and 64 (user's choice)
    Returns {"params": {name: [lo, hi]}, "batch_choices": [...], "best_trials": [...]}."""
    import spaces
    with open(os.path.join(study_dir, "study.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    method = meta["method"]
    rec = _Recorder()
    spaces.suggest(rec, method, 55, 512, "v1")
    base = _num(meta.get("base_c4_ppl")) or 18.81
    by = {}
    for r in _csv(os.path.join(study_dir, "rungs.csv")):
        c4, wm = _num(r.get("c4_ppl")), _num(r.get("wm_q4"))
        if c4 is not None and wm is not None:
            by.setdefault(int(r["trial"]), []).append({"c4_delta_pct": 100 * (c4 / base - 1), "wm_q4": wm})
    area = {no: trajectory_area(p, meta["anchors"], cap_pct) for no, p in by.items()}
    trials = {int(r["trial"]): r for r in _csv(os.path.join(study_dir, "trials.csv"))}
    # One vote per CONFIGURATION: the GP re-proposes its best point, and CE-U's
    # top 18 trials were one configuration, which would make a range of a point.
    seen, ranked = set(), []
    for no in sorted(area, key=lambda no: -area[no]):
        sig = tuple(f"{_num(trials[no].get(k)):.3g}" for k in sorted(trials.get(no, {}))
                    if k.startswith("hp_") and _num(trials[no].get(k)) is not None) if no in trials else (no,)
        if sig not in seen:
            seen.add(sig)
            ranked.append(no)
    chosen = ranked[:max(min_trials, math.ceil(top * len(ranked)))]
    fwd = {"log": math.log, "lin": lambda v: v, "logit": lambda v: math.log(v / (1 - v))}
    inv = {"log": math.exp, "lin": lambda u: u, "logit": lambda u: 1.0 / (1.0 + math.exp(-u))}
    out = {"params": {}, "batch_choices": None, "best_trials": chosen,
           "areas": {no: round(area[no], 5) for no in chosen}}
    for name, spec in rec.d.items():
        vals = [_num(trials[n].get(f"hp_{name}")) for n in chosen if n in trials]
        vals = [v for v in vals if v is not None]
        if spec[0] == "cat":
            used = sorted({int(v) for v in vals}) or spec[1]
            out["batch_choices"] = sorted(set(SMALL_BATCHES) | {b for b in spec[1] if used[0] <= b <= used[-1]})
            continue
        kind, lo, hi = spec
        f, g = fwd[kind], inv[kind]
        if not vals:
            out["params"][name] = [lo, hi]
            continue
        w = margin * (f(hi) - f(lo))
        a, b = min(f(v) for v in vals) - w, max(f(v) for v in vals) + w
        # a bound within float round-off of the stage-1 edge (logit and log round trips) is that edge
        new_lo = lo if a <= f(lo) + 1e-9 else g(a)
        new_hi = hi if b >= f(hi) - 1e-9 else g(b)
        if name == "learning_rate":
            new_hi = hi
        outer = SPACE_OUTER.get((method, name))
        if outer and outer[1] is not None and b >= f(hi):
            new_hi = outer[1]
        if outer and outer[0] is not None and a <= f(lo):
            new_lo = outer[0]
        # 6 significant digits, never outside the bounds just set (a v1 edge stays exact)
        out["params"][name] = [max(new_lo, float(f"{new_lo:.6g}")), min(new_hi, float(f"{new_hi:.6g}"))]
    return out


def frontier_plan(export, sweeps, tol, max_c4=KEEP_MAX_C4):
    """The plan rows (PLAN_FIELDS) for every search exported under `export`
    (hpo-*-wm/ and hpo-*-wmarea/; the searches of one method form one frontier).
    sweeps: [(dir with results_cells.csv, dir with results_anchors.csv)], the
    grid sweep(s) to beat; max_c4 is on the first sweep's (reporting) C4 axis."""
    lim = max_c4 / _baseline_c4(os.path.join(sweeps[0][1], "results_anchors.csv")) - 1
    swept = {}
    for cells, anchors in sweeps:
        b = _baseline_c4(os.path.join(anchors, "results_anchors.csv"))
        for r in _csv(os.path.join(cells, "results_cells.csv")):
            c4, wm = _num(r.get("c4_ppl")), _num(r.get("wm_q4"))
            if c4 is not None and wm is not None and c4 / b - 1 <= lim:
                swept.setdefault(r["method"], []).append((100 * (c4 / b - 1), 100 * poison_tpr(wm), None))
    pts, start, kept = {}, {}, {}
    for d in sorted(glob.glob(os.path.join(export, "hpo-*-wm")) + glob.glob(os.path.join(export, "hpo-*-wmarea"))):
        with open(os.path.join(d, "study.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        method, study = meta["method"], meta["study"]
        b = _num(meta.get("base_c4_ppl")) or 18.81
        start[method] = 100 * poison_tpr(meta["anchors"]["baseline"]["wm"])
        for r in _csv(os.path.join(d, "trials.csv")):
            kept[(study, int(r["trial"]))] = {int(s) for s in (r.get("kept_steps") or "").split()}
        for r in _csv(os.path.join(d, "rungs.csv")):
            c4, wm = _num(r.get("c4_ppl")), _num(r.get("wm_q4"))
            if c4 is not None and wm is not None and c4 / b - 1 <= lim:
                pts.setdefault(method, []).append(
                    (100 * (c4 / b - 1), 100 * poison_tpr(wm), (study, int(r["trial"]), int(r["step"]))))
    rows = []
    for method in sorted(pts):
        for c, t, key in thin(pts[method] + swept.get(method, []), tol, start[method]):
            if key is not None:
                study, trial, step = key
                rows.append({"method": method, "study": study, "trial": trial, "step": step,
                             "c4_rel_pct": round(c, 4), "tpr_pct": round(t, 4),
                             "on_disk": int(step in kept.get((study, trial), set()))})
    return sorted(rows, key=lambda r: (r["method"], r["study"], r["trial"], r["step"]))


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "check", "keep", "prune", "frontier", "space"])
    ap.add_argument("--export", default=None, help="frontier: a pe_export folder (hpo-*-wm/, cells-1B-v2/, anchors-b1/)")
    ap.add_argument("--sweep", default=None,
                    help="frontier: the grid sweeps to beat, comma separated, each CELLS_DIR[=ANCHORS_DIR] "
                         "(results_cells.csv, and results_anchors.csv for its Baseline C4; default "
                         "<export>/cells-1B-v2=<export>/anchors-b1). The first sets the C4 axis of --max-c4.")
    ap.add_argument("--out", default=None, help="frontier: the plan CSV to write")
    ap.add_argument("--anchor-root", default=None)
    ap.add_argument("--il-experiment", default=None,
                    help="default: IL_EXPERIMENT from settings.json")
    ap.add_argument("--output-root", default=None, help="keep / prune: where the trials' cells live")
    ap.add_argument("--run-tag", default=None, help="keep: the trial's tag (<study>-tNNNN)")
    ap.add_argument("--study", default=None, help="prune: the study, e.g. hpo-npo-wm")
    ap.add_argument("--method", default=None)
    ap.add_argument("--cell-dir", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--max-c4", type=float, default=KEEP_MAX_C4)
    ap.add_argument("--apply", action="store_true", help="prune: delete (default: only list)")
    ap.add_argument("--tol", type=float, default=float(os.environ.get("KEEP_TPR_TOL", "0") or 0),
                    help="keep / prune / frontier: TPR tolerance in percentage points (default KEEP_TPR_TOL, else 0 = "
                         "the Pareto frontier): a point must lower the TPR by more than this to stay")
    args = ap.parse_args()
    if args.cmd == "space":
        # stage 2's space from stage 1 (derive_space): resources/hpo/space-v2.json, read by spaces.py
        if not (args.export and args.out):
            ap.error("space needs --export and --out")
        import spaces
        methods = {}
        for d in sorted(glob.glob(os.path.join(args.export, "hpo-*-wm"))):
            with open(os.path.join(d, "study.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            s = derive_space(d)
            s["study"] = meta["study"]
            methods[meta["method"]] = s
            rec = _Recorder()
            spaces.suggest(rec, meta["method"], 55, 512, "v1")
            print(f"== {meta['method']}: from the best {len(s['best_trials'])} configurations of {meta['study']} "
                  f"(area {min(s['areas'].values()):.3f}-{max(s['areas'].values()):.3f}); batch {s['batch_choices']}")
            for k, (lo, hi) in s["params"].items():
                if [lo, hi] != [rec.d[k][1], rec.d[k][2]]:
                    print(f"   {k:14s} [{rec.d[k][1]:.3g}, {rec.d[k][2]:.3g}] -> [{lo:.4g}, {hi:.4g}]")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"derived_from": os.path.basename(os.path.abspath(args.export)),
                       "rule": "forget_score.derive_space: best quarter (at least 8) of the stage-1 configurations "
                               "by trajectory_area, each range widened by 15% of the stage-1 width, learning-rate "
                               "ceiling kept, SPACE_OUTER where an edge is reached, batches 32 and 64 added",
                       "methods": methods}, fh, indent=1)
        print(f"wrote {args.out}")
        return 0
    if args.cmd == "frontier":
        if not (args.export and args.out):
            ap.error("frontier needs --export and --out")
        sweeps = []
        for s in (args.sweep or f"{args.export}/cells-1B-v2={args.export}/anchors-b1").split(","):
            cells, _, anchors = s.partition("=")
            sweeps.append((cells, anchors or cells))
        rows = frontier_plan(args.export, sweeps, args.tol, args.max_c4)
        with open(args.out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=PLAN_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"tolerance {args.tol:g} TPR points, C4 up to {args.max_c4:g}, beating "
              + ", ".join(c for c, _ in sweeps))
        print(f"  {'method':16s} {'checkpoints':>11s} {'on disk':>8s} {'not on disk (trials)':>21s}")
        for m in sorted({r["method"] for r in rows}):
            mine = [r for r in rows if r["method"] == m]
            gone = [r for r in mine if not r["on_disk"]]
            print(f"  {m:16s} {len(mine):11d} {len(mine) - len(gone):8d} "
                  f"{len(gone):12d} ({len({(r['study'], r['trial']) for r in gone})})")
        gone = [r for r in rows if not r["on_disk"]]
        print(f"  {'total':16s} {len(rows):11d} {len(rows) - len(gone):8d} "
              f"{len(gone):12d} ({len({(r['study'], r['trial']) for r in gone})})")
        print(f"wrote {args.out}; optuna_hpo.py --frontier {args.out} --dry-run says what it would train and run")
        return 0
    if args.cmd == "keep":
        keep, why = keep_decision(args.output_root, args.run_tag, args.method, args.cell_dir, args.step, args.max_c4,
                                  tol=args.tol / 100.0)
        print(f"    step-{args.step}: {'on the frontier so far, kept' if keep else 'dominated, deleted'} ({why})"
              if "against" in why else f"    step-{args.step}: deleted ({why})")
        return 0 if keep else 3
    if args.cmd == "prune":
        kept, gone = prune(args.output_root, args.study, args.method, args.apply, args.max_c4, tol=args.tol / 100.0)
        print(f"{args.study}: {len(kept)} frontier checkpoint(s) kept, {len(gone)} "
              f"{'deleted' if args.apply else 'to delete (rerun with --apply)'}")
        for d in kept:
            print(f"  keep {d}")
        return 0
    if not args.anchor_root:
        ap.error("--anchor-root is required for show and check")
    try:
        st = read_settings(args.anchor_root)
        il_exp = args.il_experiment or st["eval"]["IL_EXPERIMENT"]
        if args.cmd == "check":
            check_settings(args.anchor_root, os.environ)
        a = load_anchors(args.anchor_root, il_exp)
    except (AnchorError, OSError, KeyError) as e:
        print(f"ANCHORS NOT USABLE: {e}", file=sys.stderr)
        return 1
    print(f"anchors under {args.anchor_root} (il experiment {il_exp})")
    for k, v in sorted(st["eval"].items()):
        print(f"  {k} = {v}")
    print(f"  {'':16s} {'c4':>10s} {'fk':>12s} {'il':>10s} {'wm_q4':>10s}")
    for pt in POINTS:
        m = a[pt]
        print(f"  {pt:16s} {m['c4']:10.4f} {m['fk']:12.4e} {m['il']:10.4f} {m['wm']:+10.4f}")
    g = a["gaps"]
    print(f"  gaps: fk {g['fk']:.3f} decades, il {g['il']:.3f}, wm {g['wm']:.4f}")
    # The batch-invariance diagnostic that optuna_hpo.py --anchors also submits.
    bc = os.path.join(args.anchor_root.rstrip("/") + "-batchcheck", "baseline-mns8", "step-0")
    if os.path.isdir(bc):
        ex = load_exporter()
        fk8 = _num(ex.scalar(bc, "fictional_knowledge", "probability"))
        il8 = _num(ex.insertion_likelihood(bc, il_exp))
        b = a["baseline"]
        rel = lambda x, y: "-" if x is None else f"{100.0 * (x - y) / y:+.3f}%"
        print(f"  batch check, baseline at INFERENCE_MAX_NUM_SEQS=8 vs the anchor's setting:")
        print(f"    fk {fk8 if fk8 is not None else 'missing'} vs {b['fk']:.6g}  ({rel(fk8, b['fk'])})")
        print(f"    il {il8 if il8 is not None else 'missing'} vs {b['il']:.6g}  ({rel(il8, b['il'])})")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
