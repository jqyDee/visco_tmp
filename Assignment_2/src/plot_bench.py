# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot Exercise 5B: correct matches + precision vs rotation and vs scale.

usage: uv run src/plot_bench.py out/features/bench.csv
writes 5_bench_rotation.png and 5_bench_scale.png next to the CSV.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical order (validated: CVD dE >= 9, normal dE >= 27).
# Aqua is < 3:1 on white -> every series also gets a marker shape + direct label.
STYLE = {
    "ORB": {"color": "#2a78d6", "marker": "o"},
    "AKAZE": {"color": "#eb6834", "marker": "s"},
    "BRISK": {"color": "#1baf7a", "marker": "^"},
}
INK = "#2b2b2a"
MUTED = "#8a8985"
GRID = "#e6e5e1"


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_kind(df, kind, xlabel, out_path, log_x=False):
    sub = df[df["kind"] == kind].sort_values("value")
    if sub.empty:
        print(f"no rows for kind={kind}, skipping")
        return
    xs = sorted(sub["value"].unique())

    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    ends = []  # (y, x, name) of each "correct" line's last point
    for det, st in STYLE.items():
        d = sub[sub["detector"] == det]
        if d.empty:
            continue
        for ax, col in ((ax_c, "correct"), (ax_p, "precision")):
            ax.plot(d["value"], d[col], color=st["color"], marker=st["marker"],
                    markersize=7, linewidth=2, label=det,
                    markeredgecolor="white", markeredgewidth=1.5)
        last = d.iloc[-1]
        ends.append((last["correct"], last["value"], det))

    # direct labels at line ends, pushed apart so they never overlap
    ax_c.set_ylim(bottom=0)
    min_gap = 0.07 * ax_c.get_ylim()[1]
    placed = []
    for y, x, det in sorted(ends):
        y_label = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_label)
        ax_c.annotate(det, (x, y_label), xytext=(8, 0),
                      textcoords="offset points", va="center", fontsize=9,
                      color=INK, annotation_clip=False)

    for ax in (ax_c, ax_p):
        style_axes(ax)
        if log_x:
            ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.minorticks_off()
        ax.set_xlabel(xlabel, color=INK, fontsize=10)

    ax_c.set_ylabel("correct matches (< 3 px)", color=INK, fontsize=10)
    ax_c.margins(x=0.15)
    ax_p.set_ylabel("precision (correct / kept)", color=INK, fontsize=10)
    ax_p.set_ylim(0, 1.02)
    ax_p.legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle(f"Feature detectors vs {xlabel.split(' ')[0].lower()}",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    csv = Path(sys.argv[1])
    df = pd.read_csv(csv)
    plot_kind(df, "rotation", "rotation angle (°)", csv.parent / "5_bench_rotation.png")
    plot_kind(df, "scale", "scale factor", csv.parent / "5_bench_scale.png", log_x=True)


if __name__ == "__main__":
    main()
