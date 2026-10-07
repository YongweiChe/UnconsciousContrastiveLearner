"""Fig. 13: success rate and SPL of Direct vs LogSumExp navigation, from results/rl/<maze>/evaluation.json.

    python -m rl.report
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"direct": ("Direct", "#2a78d6", ""), "lse": ("LogSumExp", "#eb6834", "//")}
INK, MUTED = "#2b2b2b", "#8a8a85"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/rl")
    p.add_argument("--figures", default="figures/rl")
    args = p.parse_args()
    evals = {f.parent.name: json.loads(f.read_text()) for f in sorted(Path(args.results).glob("*/evaluation.json"))}
    if not evals:
        raise SystemExit(f"no evaluation.json under {args.results}; run python -m rl.evaluate first")
    mazes = list(evals)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (metric, title) in zip(axes, (("success", "Success rate"), ("spl", "SPL"))):
        x = np.arange(len(mazes))
        for j, (method, (label, color, hatch)) in enumerate(STYLE.items()):
            mean = np.array([evals[m]["summary"][method][metric]["mean"] for m in mazes])
            sem = np.array([evals[m]["summary"][method][metric]["sem"] for m in mazes])
            bars = ax.bar(x + (j - 0.5) * 0.38, mean, width=0.36, yerr=sem, capsize=3, label=label, color=color,
                          hatch=hatch, edgecolor="white", linewidth=0, error_kw=dict(ecolor=INK, linewidth=1))
            for b, v in zip(bars, mean):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, mazes)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_ylim(0, 1.08)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK)
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
    n = next(iter(evals.values()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Navigating to row/column labels ({len(n['seeds'])} seeds x {n['n_trials']} episodes per maze, ± s.e.)",
                 color=INK, y=1.1)
    out = Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fig13_language_navigation.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
