"""Hyperparameter heatmaps over the exported sweep tables.

WHY THIS EXISTS
---------------
`export_results.py` writes one row per evaluated checkpoint. Reading a
hyperparameter plane off that by hand means grouping on two columns, reducing a
whole trajectory to one number, and deciding what to do about cells that were
never run. This does those three things explicitly and lets every one of them
be set from the command line, so a figure can be re-cut without editing code.

WHY THE PLANES ARE SPARSE  (read this before concluding the plot is broken)
--------------------------------------------------------------------------
The sweeps were designed one-factor-at-a-time, not factorially. For most
methods there is:

    a KNOB sweep at one fixed learning rate     1B-p3-satimp-b{1,2,5,10}  @ 5e-5
    an LR sweep at one fixed knob value         1B-p1-satimp-rt-lr{1e-5,5e-5} @ b5

which traces a CROSS through a single point, not a plane. Roughly 5 of 8 cells
in a 4x2 are therefore empty by construction. `grad-diff` is the one complete
grid, because its knob sweep was run at both learning rates.

That design was deliberate -- a curve family whose rate moves with the knob
cannot be read as a family -- so the holes are missing EXPERIMENTS, not missing
plotting. `--coverage` shows the cross shape directly and `--missing` prints the
pairs that would have to run to close it.

A heatmap with most cells empty is a weak figure. Decide from `--missing`
whether to complete the grid or to keep the one-factor line panels instead.

USAGE
-----
    # the plane behind section 11's satimp panel
    python internal/uwiki/heatmap.py --method satimp --x lr --y knob_value \
        --metric il --relative --agg best --cap 5

    # raw perplexity instead of a normalised fraction
    python internal/uwiki/heatmap.py --method satimp --x lr --y knob_value \
        --metric c4 --raw --agg best

    # rmu's c x alpha grid: alpha exists only inside the run tag
    python internal/uwiki/heatmap.py --method rmu \
        --x-regex 'a(\\d+)$' --x-label alpha \
        --y-regex '-c(\\d+)-' --y-label c \
        --metric il --relative --tags '1B-rmu-grid-*'

    # why is it sparse
    python internal/uwiki/heatmap.py --method npo --x lr --y knob_value --coverage
    python internal/uwiki/heatmap.py --method npo --x lr --y knob_value --missing

    # the decayed tree, written to a file
    python internal/uwiki/heatmap.py --exports exports-decayed --method satimp \
        --x lr --y knob_value --metric il --relative --out satimp-decayed.png

METRICS
-------
`--metric` takes a short name plus `--relative` (default) or `--raw`:

    short   relative column   raw column    better
    fk      fk_forgot         fk_prob       higher / lower
    il      il_forgot         il_ppl        higher / higher
    wm      wm_removed        wm_q4         higher / lower
    c4      c4_delta_pct      c4_ppl        lower  / lower

Any literal column name from results_cells.csv also works, in which case
`--relative/--raw` is ignored. "Relative" here means a fraction of the distance
from the un-unlearned baseline to the deep-ignorance floor, as defined in
export_results.py -- log-space for fk, linear for il and wm.

AGGREGATION
-----------
Each (x, y) cell holds a whole trajectory, so one number has to be chosen:

    best   the extreme in the metric's good direction, subject to --cap
    worst  the extreme in the other direction
    last   the highest step
    at N   the checkpoint at step N exactly
    count  how many checkpoints exist  (what --coverage plots)

`--cap` drops checkpoints whose c4_delta_pct exceeds it before aggregating,
which is what makes "best" mean "best at an affordable utility cost" rather
than "furthest along a destroyed model". Pass `--cap none` to disable.
"""

import argparse
import csv
import fnmatch
import glob
import json
import math
import os
import re
import sys

# short name -> (relative column, raw column, higher_is_better_relative,
#                higher_is_better_raw)
METRICS = {
    "fk": ("fk_forgot", "fk_prob", True, False),
    "il": ("il_forgot", "il_ppl", True, True),
    "wm": ("wm_removed", "wm_q4", True, False),
    "c4": ("c4_delta_pct", "c4_ppl", False, False),
}

NUMERIC = ("lr", "knob_value", "step", "fk_prob", "fk_forgot", "il_ppl",
           "il_forgot", "c4_ppl", "c4_delta_pct", "wm_full", "wm_q4",
           "wm_removed")

# Config fields merged in from results_configs.csv. These become usable as axes
# and as filters, which is the only way to plot a method on the plane its own
# two hyperparameters span -- satimp's beta1 x retain_loss_weight, say, rather
# than knob x learning rate.
CONFIG_NUMERIC = ("retain_loss_weight", "beta1", "beta2", "beta", "gamma",
                  "steering_coefficient", "alpha", "learning_rate",
                  "max_steps", "epochs", "seed")


def fnum(s):
    """Float, or None. '' and 'nan' are missing, not zero."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load_cells(exports_dir, tags=None, methods=None, variant=None):
    path = os.path.join(exports_dir, "results_cells.csv")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: no results_cells.csv under {exports_dir!r}. "
                 f"Run export_results.py --out {exports_dir} first.")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if methods and r.get("method") not in methods:
                continue
            if tags and not any(fnmatch.fnmatch(r.get("run_tag", ""), t)
                                for t in tags):
                continue
            if variant is not None and (r.get("variant") or "") != variant:
                continue
            for k in NUMERIC:
                r["_" + k] = fnum(r.get(k))
            rows.append(r)
    return rows


def load_configs(exports_dir):
    """(run_tag, method, knob_value) -> that cell's config record.

    Merged into every row so the CONFIGURED parameters -- retain_loss_weight,
    beta1, beta2, gamma, alpha, steering_coefficient -- can be used as axes and
    filters. results_cells.csv carries only the swept knob and the learning
    rate, which is why the first version of this tool could plot knob x lr and
    nothing else, and why a two-hyperparameter method like satimp could not be
    shown on the plane its two hyperparameters actually span.

    Method names are normalised because drivers record their MODULE name
    ("ce_u", "grad_diff") while the sweep tables use the dispatch name
    ("ce-u", "grad-diff"), so a naive join silently misses those two.
    """
    path = os.path.join(exports_dir, "results_configs.csv")
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = (r.get("run_tag", ""),
                   (r.get("method") or "").replace("_", "-"),
                   (r.get("knob_value") or ""))
            out[key] = r
    return out


def load_retain(exports_dir):
    """(run_tag, method) -> retain_loss_weight, from audit_configs.py's CSV.

    The retain weight is NOT in results_cells.csv, and it is the field that
    decides whether two cells may share a plane at all. Reading it from the
    cells' own config records is the only reliable route: tag names do not
    carry it (`1B-p3-satimp-b10.0` has no -fo suffix and is forget-only), and
    one tag can hold different weights for different methods.

    Method names are normalised because drivers record their MODULE name
    ("ce_u", "grad_diff") while the sweep tables use the dispatch name
    ("ce-u", "grad-diff"), so a naive join silently misses those two.
    """
    path = os.path.join(exports_dir, "results_configs.csv")
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = (r.get("run_tag", ""), (r.get("method") or "").replace("_", "-"))
            w = (r.get("retain_loss_weight") or "").strip()
            if w:
                out.setdefault(key, set()).add(w)
    return {k: sorted(v) for k, v in out.items()}


def load_endpoints(exports_dir):
    p = os.path.join(exports_dir, "results_meta.json")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return (json.load(f) or {}).get("endpoints", {}) or {}


def axis_getter(column, regex, label):
    """Return (fn, label). A regex reads the axis out of run_tag.

    rmu's alpha is only ever recorded in the run tag (1B-rmu-grid-c5-a100), so
    a column-only tool could not plot that plane at all.
    """
    if regex:
        rx = re.compile(regex)
        def get(r):
            m = rx.search(r.get("run_tag", ""))
            return m.group(1) if m else None
        return get, (label or f"run_tag ~ {regex}")
    def get(r):
        v = r.get(column)
        return None if v in ("", None) else str(v)
    return get, (label or column)


def sort_key(v):
    """Numeric where possible, so 1e-5 sorts below 5e-5 and 10 above 9."""
    try:
        return (0, float(v), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(v))


def aggregate(rows, col, higher_better, how, at_step):
    vals = [r for r in rows if r.get("_" + col) is not None]
    if how == "count":
        # An empty cell must stay None so it renders as "not run". Returning
        # 0.0 here printed never-run cells as a measured zero and counted them
        # as present -- exactly the confusion the dashed-cell convention and
        # this whole tool exist to prevent.
        return (float(len(rows)) if rows else None), len(rows)
    if not vals:
        return None, len(rows)
    if how == "last":
        r = max(vals, key=lambda r: (r["_step"] if r["_step"] is not None else -1))
    elif how == "at":
        hit = [r for r in vals if r["_step"] == at_step]
        if not hit:
            return None, len(rows)
        r = hit[0]
    else:
        want_max = higher_better if how == "best" else (not higher_better)
        r = (max if want_max else min)(vals, key=lambda r: r["_" + col])
    return r["_" + col], len(rows)


def newest_bundle(prefix="iclr"):
    """The highest-year tueplots bundle whose name starts with `prefix`.

    Hardcoding `bundles.iclr2024` means the figures silently keep 2024's
    geometry after tueplots ships a newer one, and the submission target moves
    every year. Resolving it at call time makes `pip install -U tueplots` the
    whole upgrade. Falls back to the newest neurips bundle, whose column widths
    are close enough to be a sane default, then to None (matplotlib defaults).
    """
    from tueplots import bundles
    import re
    for pre in (prefix, "neurips"):
        cand = []
        for name in dir(bundles):
            m = re.fullmatch(rf"{pre}(\d{{4}})", name)
            if m:
                cand.append((int(m.group(1)), name))
        if cand:
            return max(cand)[1]
    return None


def apply_paper_style(plt, ncols=1, usetex=False, prefix="iclr"):
    """Set rcParams from the newest ICLR bundle. Returns the bundle name used.

    usetex=False by default: the bundles ask for LaTeX, and a machine without a
    TeX install fails deep inside the first draw with an error that does not
    mention fonts. Pass usetex=True once you have confirmed `latex` is on PATH --
    the glyphs then match the template exactly.
    """
    try:
        from tueplots import bundles
    except ImportError:
        return "none (tueplots not installed -- pip install tueplots)"
    name = newest_bundle(prefix)
    if name is None:
        return "none (no matching bundle)"
    plt.rcParams.update(getattr(bundles, name)(ncols=ncols, usetex=usetex))
    return f"{name} (ncols={ncols}, usetex={usetex})"


def main():
    ap = argparse.ArgumentParser(
        description="Hyperparameter heatmaps over exported sweep tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exports", default="exports",
                    help="directory holding results_cells.csv (default: exports)")
    ap.add_argument("--method", action="append", dest="methods",
                    help="restrict to a method; repeatable")
    ap.add_argument("--tags", action="append",
                    help="glob on run_tag; repeatable")
    ap.add_argument("--variant", help="exact match on the variant column")
    ap.add_argument("--retain", default="any",
                    help="filter on retain_loss_weight: 'any', 'on' (1.0), "
                         "'off' (0.0), or an EXACT value such as 0.01. "
                         "Intermediate weights are real -- SatImp's alpha sweep "
                         "used 0.01 and 0.1 -- and an on/off filter drops them "
                         "silently, which is why this is not a fixed choice list.")

    ap.add_argument("--x", default="lr", help="column for the x axis")
    ap.add_argument("--y", default="knob_value", help="column for the y axis")
    ap.add_argument("--x-regex", help="instead of --x, capture group 1 from run_tag")
    ap.add_argument("--y-regex", help="instead of --y, capture group 1 from run_tag")
    ap.add_argument("--x-label"); ap.add_argument("--y-label")

    ap.add_argument("--metric", default="il",
                    help="fk | il | wm | c4, or a literal column name")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--relative", action="store_true", default=None,
                   help="fraction of baseline -> deep-ignorance (default)")
    g.add_argument("--raw", dest="relative", action="store_false",
                   help="the measured value instead")

    ap.add_argument("--agg", default="best",
                    choices=["best", "worst", "last", "at", "count"])
    ap.add_argument("--at-step", type=float, help="step for --agg at")
    ap.add_argument("--cap", default=None,
                    help="drop checkpoints above this c4_delta_pct "
                         "(default 5, or none under --coverage; 'none' disables)")
    ap.add_argument("--max-ppl", type=float, default=None,
                    help="also drop checkpoints whose il_ppl exceeds this")

    ap.add_argument("--coverage", action="store_true",
                    help="plot checkpoint COUNTS -- shows the sweep shape")
    ap.add_argument("--missing", action="store_true",
                    help="list the (x, y) pairs that were never run, and stop")
    ap.add_argument("--out", help="write a PNG/SVG/PDF here (needs matplotlib)")
    ap.add_argument("--json", dest="json_out", help="write the matrix as JSON")
    ap.add_argument("--style", default="iclr", choices=("iclr", "neurips", "none"),
                    help="tueplots bundle family for --out; the newest year that "
                         "tueplots ships is resolved at run time (default: iclr)")
    ap.add_argument("--ncols", type=int, default=1, choices=(1, 2),
                    help="1 for a full-width figure, 2 for a single column")
    ap.add_argument("--usetex", action="store_true",
                    help="render text with LaTeX -- matches the template exactly, "
                         "but fails obscurely without a TeX install")
    ap.add_argument("--vmax", type=float, help="fix the colour scale maximum")
    ap.add_argument("--cmap", default="Blues")
    ap.add_argument("--decimals", type=int, default=4)
    a = ap.parse_args()

    if a.relative is None:
        a.relative = True
    if a.cap is None:
        # --coverage and --missing both ask "was this pair ever run", which the
        # utility cap distorts: a cell whose every checkpoint blew the cap still
        # ran. Capping here would report it as never run and send someone off to
        # relaunch an experiment that already exists.
        a.cap = "none" if (a.coverage or a.missing) else "5"
    if a.agg == "at" and a.at_step is None:
        sys.exit("ERROR: --agg at requires --at-step")

    # ---- metric -----------------------------------------------------------
    if a.metric in METRICS:
        rel_col, raw_col, rel_hi, raw_hi = METRICS[a.metric]
        col = rel_col if a.relative else raw_col
        higher_better = rel_hi if a.relative else raw_hi
    else:
        col, higher_better = a.metric, True
        if col.startswith("c4") or col in ("fk_prob", "wm_q4", "wm_full"):
            higher_better = False

    rows = load_cells(a.exports, a.tags, set(a.methods) if a.methods else None,
                      a.variant)
    if not rows:
        sys.exit("ERROR: no rows matched. Check --exports / --method / --tags.")

    # --- merge the configured parameters onto every row ---------------------
    # results_cells.csv records only the swept knob and the learning rate; the
    # rest of the configuration lives in results_configs.csv. Merging makes
    # retain_loss_weight, beta1, beta2, gamma and alpha available as AXES, so a
    # method with two hyperparameters can be plotted on the plane they span.
    cmap = load_configs(a.exports)
    if cmap:
        merged = 0
        for r in rows:
            c = cmap.get((r.get("run_tag", ""),
                          (r.get("method") or "").replace("_", "-"),
                          r.get("knob_value", "")))
            if not c:
                continue
            merged += 1
            for k, v in c.items():
                # Never let a config field shadow the cell's own measurements,
                # or the identity columns the join was keyed on.
                if k in r:
                    continue
                r[k] = v
                if k in CONFIG_NUMERIC:
                    r["_" + k] = fnum(v)
        if merged < len(rows):
            print(f"note: {len(rows) - merged} of {len(rows)} row(s) have no "
                  "config record; config-derived axes and --retain drop them.",
                  file=sys.stderr)

    # --- retain filter ------------------------------------------------------
    if a.retain != "any":
        if not cmap:
            sys.exit("ERROR: --retain needs results_configs.csv beside "
                     f"results_cells.csv in {a.exports!r}. Build it with:\n"
                     "       python internal/uwiki/audit_configs.py "
                     "--output-root <sweep root> --out <exports dir>")
        want = {"on": "1.0", "off": "0.0"}.get(a.retain, a.retain)
        wantf = fnum(want)
        if wantf is None:
            sys.exit(f"ERROR: --retain {a.retain!r} must be 'any', 'on', 'off', "
                     "or an exact numeric weight such as 0.01")
        # Compared as FLOATS: the CSV may hold 0.01, 0.010 or 1e-2 for one
        # value, and a string match would silently return an empty plane.
        rows = [r for r in rows
                if r.get("_retain_loss_weight") is not None
                and abs(r["_retain_loss_weight"] - wantf) < 1e-12]
        if not rows:
            sys.exit(f"ERROR: no cells with retain_loss_weight == {wantf:g}. "
                     "Methods whose published form has no retain term (ce-u, "
                     "gradient-ascent, wga) are always empty for a nonzero "
                     "value. Values actually present are listed by:\n"
                     "       python internal/uwiki/audit_configs.py --print")
    if col not in rows[0]:
        sys.exit(f"ERROR: no column {col!r}. Available: "
                 + ", ".join(k for k in rows[0] if not k.startswith('_')))

    # ---- utility cap ------------------------------------------------------
    # Applied BEFORE aggregation: "best" must mean best at an affordable cost,
    # not furthest along an already-destroyed model.
    n_before = len(rows)
    if a.cap.lower() not in ("none", "off", ""):
        cap = float(a.cap)
        rows = [r for r in rows
                if r["_c4_delta_pct"] is not None and r["_c4_delta_pct"] <= cap]
    if a.max_ppl is not None:
        rows = [r for r in rows
                if r["_il_ppl"] is not None and r["_il_ppl"] <= a.max_ppl]
    n_capped = n_before - len(rows)

    xget, xlab = axis_getter(a.x, a.x_regex, a.x_label)
    yget, ylab = axis_getter(a.y, a.y_regex, a.y_label)

    cells = {}
    for r in rows:
        xv, yv = xget(r), yget(r)
        if xv is None or yv is None:
            continue
        cells.setdefault((xv, yv), []).append(r)
    if not cells:
        sys.exit("ERROR: no rows carried both axes. For an axis that only "
                 "exists inside the tag, use --x-regex / --y-regex.")

    xs = sorted({x for x, _ in cells}, key=sort_key)
    ys = sorted({y for _, y in cells}, key=sort_key)

    how = "count" if a.coverage else a.agg
    matrix, counts, tags_at = [], [], {}
    for y in ys:
        mrow, crow = [], []
        for x in xs:
            rs = cells.get((x, y), [])
            v, n = aggregate(rs, col, higher_better, how, a.at_step)
            mrow.append(v); crow.append(n)
            if rs:
                tags_at[(x, y)] = sorted({r["run_tag"] for r in rs})
        matrix.append(mrow); counts.append(crow)

    filled = sum(1 for r in matrix for v in r if v is not None)
    total = len(xs) * len(ys)

    # ---- missing-cell report ---------------------------------------------
    if a.missing:
        print(f"{xlab} x {ylab} for method(s) "
              f"{','.join(a.methods) if a.methods else 'all'}: "
              f"{filled}/{total} cells present\n")
        print("MISSING (never run):")
        for y in ys:
            for x in xs:
                if (x, y) not in cells:
                    print(f"  {ylab}={y:<10} {xlab}={x}")
        print("\nPRESENT:")
        for (x, y), tg in sorted(tags_at.items(), key=lambda kv: sort_key(kv[0][0])):
            print(f"  {ylab}={y:<10} {xlab}={x:<10} {len(cells[(x,y)]):>3} ckpts  "
                  f"{', '.join(tg)}")
        return

    # ---- text table (always; the tool must work headless) -----------------
    what = "checkpoint count" if a.coverage else (
        f"{col}  ({'relative' if a.relative and a.metric in METRICS else 'raw'}, "
        f"agg={how}" + (f" @{a.at_step:g}" if how == "at" else "") + ")")
    print(f"\n{what}")
    print(f"rows={ylab}  cols={xlab}  "
          f"method={','.join(a.methods) if a.methods else 'all'}  "
          f"exports={a.exports}")
    if n_capped:
        print(f"cap: dropped {n_capped} checkpoint(s) over c4_delta_pct {a.cap}")
    print(f"coverage: {filled}/{total} cells")

    # Two run tags landing in one cell is not automatically wrong -- a rerun or
    # a denser resample belongs there -- but tags also differ in retain weight,
    # model and schedule, and silently maxing over a retain and a forget-only
    # arm is an analysis error, not a plot. Name them and let the caller decide.
    pooled = {k: v for k, v in tags_at.items() if len(v) > 1}
    if pooled:
        print(f"WARNING: {len(pooled)} cell(s) aggregate more than one run_tag.")
        print("         Tags can differ in retain weight, model or schedule, so "
              "these may mix")
        print("         configurations. Narrow with --tags if so:")
        for (x, y), tg in sorted(pooled.items(), key=lambda kv: sort_key(kv[0][0])):
            print(f"           {ylab}={y}, {xlab}={x}:  {', '.join(tg)}")
    print()

    w = max(12, a.decimals + 8)
    print(" " * 12 + "".join(f"{x:>{w}}" for x in xs))
    for y, mrow in zip(ys, matrix):
        line = f"{y:>10}  "
        for v in mrow:
            line += (f"{'.':>{w}}" if v is None
                     else (f"{int(v):>{w}}" if a.coverage
                           else f"{v:>{w}.{a.decimals}f}"))
        print(line)
    print("\n'.' = never run, not a measured zero.")
    if filled < total:
        print(f"{total - filled} empty cell(s). See --missing, and the module "
              f"docstring for why these planes are sparse.")

    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump({"x": xs, "y": ys, "x_label": xlab, "y_label": ylab,
                       "metric": col, "relative": bool(a.relative),
                       "agg": how, "cap": a.cap, "m": matrix,
                       "counts": counts, "filled": filled, "total": total},
                      f, indent=1)
        print(f"wrote {a.json_out}")

    # ---- figure -----------------------------------------------------------
    if a.out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            sys.exit("ERROR: --out needs matplotlib. The table above is "
                     "complete; re-run without --out, or pip install matplotlib.")
        if a.style != "none":
            applied = apply_paper_style(plt, ncols=a.ncols, usetex=a.usetex)
            print(f"style: {applied}")
        arr = np.array([[np.nan if v is None else v for v in r] for r in matrix],
                       dtype=float)
        if a.style == "none":
            figsize = (1.5 + 1.25 * len(xs), 1.4 + 0.85 * len(ys))
        else:
            # The bundle fixes WIDTH to the template's text block -- that is the
            # point of using it, so never override it. Height is ours: the
            # bundle's default assumes a line plot, and a plane with eleven rows
            # needs more. Scale from the bundle width so cells stay near-square,
            # and cap it so a tall plane cannot run off the page.
            w = plt.rcParams["figure.figsize"][0]
            figsize = (w, min(0.90 * w * len(ys) / max(len(xs), 1) + 0.7, 8.0))
        fig, ax = plt.subplots(figsize=figsize)
        cmap = matplotlib.colormaps[a.cmap].copy()
        cmap.set_bad("none")          # never-run cells are hollow, not tinted
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, origin="lower",
                       aspect="auto", vmin=0 if higher_better else None,
                       vmax=a.vmax)
        ax.set_xticks(range(len(xs)), xs)
        ax.set_yticks(range(len(ys)), ys)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(what, fontsize=10)
        finite = arr[np.isfinite(arr)]
        mid = (finite.max() + finite.min()) / 2 if finite.size else 0
        for i in range(len(ys)):
            for j in range(len(xs)):
                v = arr[i, j]
                if np.isnan(v):
                    # Dashed outline + en dash: "not run" must never be
                    # mistakable for a light-coloured measured value.
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               ec="0.6", ls="--", lw=.8))
                    ax.text(j, i, "--", ha="center", va="center",
                            color="0.6", fontsize=8)
                else:
                    ax.text(j, i, f"{int(v)}" if a.coverage
                            else f"{v:.{min(a.decimals,3)}f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if v > mid else "black")
        fig.colorbar(im, ax=ax, shrink=.85)
        # The tueplots bundles switch on constrained_layout, and calling
        # tight_layout() after a colorbar exists then raises
        # "Colorbar layout of new layout engine not compatible with old engine".
        # Constrained layout already does the job, so only lay out by hand when
        # no bundle is active.
        if a.style == "none":
            fig.tight_layout()
        fig.savefig(a.out, dpi=200)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
