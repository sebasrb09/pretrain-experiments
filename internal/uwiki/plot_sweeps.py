# -*- coding: utf-8 -*-
"""The hyperparameter-sweep figure: every method, every knob value, one plane.

    from internal.uwiki.plot_sweeps import sweep_figure
    fig = sweep_figure("/data/fs201378/sr44833/exports")
    fig.savefig("sweeps.pdf", bbox_inches="tight")

Reads results_cells.csv / results_anchors.csv, so re-running after a fresh
export picks up new runs with no edits here.

DESIGN NOTES
------------
Small multiples, not one panel. Eight methods x 2-4 knob values is ~25 series;
no single plane separates that many. One panel per method, one line per knob
value, and the control redrawn in every panel so each comparison is local.

Knob values are ORDERED (b1 = 0.5 < 1 < 2 < 5), so they take a one-hue
sequential ramp rather than categorical hues -- the ramp encodes the ordering
that categorical colors would throw away. Both ramps below pass the ordinal
checks from the dataviz skill (monotone lightness, adjacent dL >= 0.06,
light-end contrast >= 2:1, hue spread <= 40 deg).

X is insertion likelihood, not fk_prob: it is the better-calibrated of the two
(deep-ignorance scatter +-1% against 3.4x) and, unlike fk_prob, its
normalisation does not move when the deep-ignorance floor is re-estimated.

Y stops at PPL_MAX. Past that every model is destroyed and the differences
stop carrying information, while the compression hides the transition -- which
is the part the figure is about.
"""
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

PPL_MAX = 42.0
BASELINE_PPL = 18.773

RAMP = {
    "light": ["#7FAAC9", "#4E86AC", "#2A6187", "#0F3A55"],
    "dark":  ["#B3D2E8", "#7FAECF", "#4E86AC", "#2A6187"],
}
INK = {"light": ("#101820", "#5B6675", "#8994A3", "#DCE2E8", "#F5F7F9"),
       "dark":  ("#E4E9EF", "#94A0AF", "#6C7889", "#253039", "#151C24")}
CONTROL = {"light": "#2F6D4F", "dark": "#74C29A"}

# method -> (panel title, knob label, [(value label, run_tag), ...])
# One learning rate per panel: a curve family is only interpretable if the
# rate is held fixed while the knob varies (confirmed by audit_cell_lr.sh).
PANELS = [
    ("ce-u",            "lr",   [("1e-5", "1B-dense-ceu-lr1e-5"),
                                 ("5e-5", "1B-dense-ceu-lr5e-5")]),
    ("gradient-ascent", "lr",   [("1e-5", "1B-p1-ga-fo-lr1e-5"),
                                 ("5e-5", "1B-p2-ga-fo")]),
    ("wga",             r"$\beta_1$",
                                [("0.5", "1B-beta-wga-b0.5"), ("1.0", "1B-dense-wga-lr1e-5"),
                                 ("2.0", "1B-beta-wga-b2.0"), ("5.0", "1B-beta-wga-b5.0")]),
    ("satimp",          r"$\beta_1$",
                                [("1.0", "1B-p3-satimp-b1.0"), ("5.0", "1B-p2-satimp-fo"),
                                 ("10.0", "1B-p3-satimp-b10.0")]),
    ("grad-diff",       r"$\lambda$",
                                [("0.5", "1B-p3-grad-diff-l0.5"), ("1.0", "1B-p3-grad-diff-l1.0"),
                                 ("2.0", "1B-p3-grad-diff-l2.0"), ("5.0", "1B-p3-grad-diff-l5.0")]),
    ("simnpo",          r"$\beta$",
                                [("0.1", "1B-p2-simnpo-fo"), ("0.5", "1B-p3-simnpo-b0.5"),
                                 ("2.5", "1B-p3-simnpo-b2.5")]),
    ("npo",             r"$\beta$",
                                [("1e-3", "1B-p3-npo-b1e-3"), ("1e-2", "1B-p3-npo-b1e-2"),
                                 ("1e-1", "1B-frontier-npo-lr1e-5")]),
    ("rmu",             "c",    [("6.5", "1B-rmu-noresume")]),
]

# Prefer a dense re-run where one exists: 1B-p3d-* supersedes the sparse curve
# for the same (method, value). Applied automatically so this file needs no
# edit when the dense grid lands.
def _prefer_dense(rows, method, value, tag):
    dense = f"1B-p3d-{method}-b{value}"
    return dense if any(r["run_tag"] == dense for r in rows) else tag


def _load(exports_dir):
    with open(os.path.join(exports_dir, "results_cells.csv"), encoding="utf-8") as f:
        cells = list(csv.DictReader(f))
    with open(os.path.join(exports_dir, "results_anchors.csv"), encoding="utf-8") as f:
        anchors = list(csv.DictReader(f))
    return cells, anchors


def _num(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


def _traj(cells, tag):
    pts = [(_num(r, "il_forgot"), _num(r, "c4_ppl")) for r in cells
           if r["run_tag"] == tag]
    pts = [(x * 100, y) for x, y in pts
           if x is not None and y is not None and y <= PPL_MAX and x <= 1.05]
    return sorted(pts)


def sweep_figure(exports_dir, mode="light", ncols=4):
    cells, anchors = _load(exports_dir)
    ink, muted, faint, rule, surface = INK[mode]
    ramp = RAMP[mode]

    control = sorted(
        (_num(a, "il_forgot") * 100, _num(a, "c4_ppl")) for a in anchors
        if a["point"] == "unlearn-baseline" and a["step"]
        and _num(a, "il_forgot") is not None)

    nrows = -(-len(PANELS) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.5 * nrows),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(surface)
    axes = axes.ravel()

    for ax, (method, knob, series) in zip(axes, PANELS):
        ax.set_facecolor(surface)
        # baseline first, so every other mark sits above it
        ax.axhline(BASELINE_PPL, color=faint, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.plot([p[0] for p in control], [p[1] for p in control],
                color=CONTROL[mode], lw=2.2, zorder=3,
                solid_capstyle="round", label="control")

        drawn, ends = 0, []
        for i, (value, tag) in enumerate(series):
            tag = _prefer_dense(cells, method, value, tag)
            pts = _traj(cells, tag)
            if len(pts) < 2:
                continue
            color = ramp[min(i, len(ramp) - 1)]
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=color, lw=2, marker="o", ms=3.4, mew=0,
                    solid_capstyle="round", zorder=4 + i)
            ends.append([pts[-1][0], pts[-1][1], f"{knob}={value}", color])
            drawn += 1

        # Direct labels at each curve's end -- <=4 series per panel, so every one
        # is named and identity never rests on color alone. Curves that stop in
        # the same region (wga's beta1=0.5/2.0/5.0 all die below 50% forgotten)
        # would otherwise stack their labels on top of each other, so nudge them
        # apart vertically first, in data units, preserving their order.
        MIN_GAP = (PPL_MAX - BASELINE_PPL) * 0.062
        ends.sort(key=lambda e: e[1])
        for j in range(1, len(ends)):
            if ends[j][1] - ends[j - 1][1] < MIN_GAP:
                ends[j][1] = ends[j - 1][1] + MIN_GAP
        for x, y, text, color in ends:
            ax.annotate(text, xy=(x, y), xytext=(4, 0),
                        textcoords="offset points", fontsize=7.5,
                        color=color, va="center", ha="left", zorder=9,
                        annotation_clip=False)

        ax.set_title(method, loc="left", fontsize=11, color=ink, pad=7)
        if drawn == 0:
            ax.text(0.5, 0.5, "no usable trajectory", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=faint)
        ax.grid(True, color=rule, lw=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(rule)
        ax.tick_params(colors=faint, labelsize=8, length=3)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        # Right margin leaves room for the direct labels, which sit outside the
        # last data point and are drawn unclipped.
        ax.set_xlim(-4, 124)
        ax.set_ylim(BASELINE_PPL - 0.9, PPL_MAX)

    for ax in axes[len(PANELS):]:
        ax.set_visible(False)

    fig.supxlabel("forgotten  (% of baseline→deep-ignorance, insertion likelihood)",
                  fontsize=9.5, color=muted, y=0.02)
    fig.supylabel("c4 perplexity", fontsize=9.5, color=muted, x=0.012)
    fig.suptitle("Every method, every hyperparameter value, against the control",
                 fontsize=13.5, color=ink, x=0.055, ha="left", y=0.985)

    # One legend for the two things that repeat in every panel. Knob values are
    # direct-labelled per panel, so they are deliberately absent here.
    handles = [plt.Line2D([], [], color=CONTROL[mode], lw=2.2),
               plt.Line2D([], [], color=faint, lw=1, ls=(0, (2, 3)))]
    fig.legend(handles, ["continued pretraining (the control)",
                         f"baseline perplexity ({BASELINE_PPL:.2f})"],
               loc="upper right", frameon=False, fontsize=9,
               labelcolor=muted, bbox_to_anchor=(0.98, 0.995))

    fig.tight_layout(rect=(0.03, 0.035, 1, 0.955))
    return fig


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get("DATA", os.path.expanduser("~")), "exports")
    f = sweep_figure(d, mode=sys.argv[2] if len(sys.argv) > 2 else "light")
    out = "sweep_figure.pdf"
    f.savefig(out, bbox_inches="tight", facecolor=f.get_facecolor())
    print("wrote", out)
