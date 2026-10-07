"""Figures for the pretrained-model experiments, from results/real_world/*.json.

    python -m real_world.report

comparison.png  Direct vs Monte Carlo (largest M) vs a model trained on that pair, R@1 and R@10
scaling.png     Monte Carlo recall@1 vs the number of bridge samples M, against Direct
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Same categorical order as didactic.report: blue, orange, aqua; texture via hatching as the secondary encoding.
STYLE = {
    "Direct": dict(color="#2a78d6", hatch=""),
    "Monte Carlo": dict(color="#eb6834", hatch="//"),
    "Model trained on the pair": dict(color="#1baf7a", hatch=".."),
}
INK, MUTED = "#2b2b2b", "#8a8a85"

COMPARISONS = [  # (experiment, label, direct baseline, model trained on the A-C pair)
    ("clip_clap_via_caption_pool", "CLIP + CLAP\nimage ↔ audio\nvia text", "CLIP image . CLAP audio", None),
    ("languagebind_via_caption_pool", "LanguageBind\nimage ↔ audio\nvia text", "LanguageBind direct", None),
    ("imagebind_clap_via_audio", "ImageBind + CLAP\nimage ↔ text\nvia audio", None, "OpenCLIP ViT-H (paired, in ImageBind)"),
    ("imagebind_clip_via_image", "ImageBind + CLIP\naudio ↔ text\nvia images", "ImageBind direct", "CLAP (paired)"),
]
SCALING = [  # (experiment, title, direct baseline)
    ("clip_clap_via_caption_pool", "CLIP + CLAP, caption bridge", "CLIP image . CLAP audio"),
    ("languagebind_via_caption_pool", "LanguageBind, caption bridge", "LanguageBind direct"),
    ("imagebind_via_image", "ImageBind, image bridge", "ImageBind direct"),
]


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)


def monte_carlo_by_m(summary):
    """{M: summary} for the Monte Carlo entries of an experiment."""
    return {int(k.split("M=")[1].rstrip(")")): v for k, v in summary.items() if k.startswith("Monte Carlo (M=")}


def plot_comparison(results, path, n_candidates):
    rows = [(label, results[e], d, t) for e, label, d, t in COMPARISONS if e in results]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, metric in zip(axes, ("R@1", "R@10")):
        x = np.arange(len(rows))
        for j, method in enumerate(STYLE):
            vals = []
            for _, summary, direct, trained in rows:
                key = {"Direct": direct, "Model trained on the pair": trained}.get(method)
                entry = summary.get(key) if key else (max(monte_carlo_by_m(summary).items())[1] if method == "Monte Carlo" else None)
                vals.append((entry[metric]["mean"], entry[metric]["ci95"]) if entry else (np.nan, 0))
            mean, ci = np.array(vals).T
            bars = ax.bar(x + (j - 1) * 0.27, mean, width=0.25, yerr=ci, capsize=3, label=method, color=STYLE[method]["color"],
                          hatch=STYLE[method]["hatch"], edgecolor="white", linewidth=0, error_kw=dict(ecolor=INK, linewidth=1))
            for b, m in zip(bars, mean):
                if np.isfinite(m):
                    ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.2f}", ha="center", fontsize=7, color=INK)
        k = int(metric[2:])
        ax.axhline(k / n_candidates, color=MUTED, linewidth=0.8, linestyle=":")
        ax.text(-0.45, k / n_candidates + 0.012, "chance", color=MUTED, fontsize=8, ha="left")
        ax.set_xticks(x, [r[0] for r in rows], fontsize=8.5)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_title(f"Recall@{k} ({n_candidates} candidates)", color=INK, fontsize=11)
        ax.set_ylim(0, 1.05)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def plot_scaling(results, path):
    rows = [(results[e], title, d) for e, title, d in SCALING if e in results]
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.6), sharey=True, squeeze=False)
    for ax, (summary, title, direct) in zip(axes[0], rows):
        by_m = sorted(monte_carlo_by_m(summary).items())
        m = np.array([k for k, _ in by_m])
        mean = np.array([v["R@1"]["mean"] for _, v in by_m])
        ci = np.array([v["R@1"]["ci95"] for _, v in by_m])
        ax.plot(m, mean, color=STYLE["Monte Carlo"]["color"], linewidth=1.5, marker="o", markersize=5, label="Monte Carlo")
        ax.fill_between(m, mean - ci, mean + ci, color=STYLE["Monte Carlo"]["color"], alpha=0.15, linewidth=0)
        d = summary[direct]["R@1"]
        ax.axhline(d["mean"], color=STYLE["Direct"]["color"], linewidth=1.5, label="Direct")
        ax.axhspan(d["mean"] - d["ci95"], d["mean"] + d["ci95"], color=STYLE["Direct"]["color"], alpha=0.12, linewidth=0)
        ax.set_xscale("log")
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel("bridge samples M", color=INK)
        ax.set_ylim(0, 0.6)
        style_axes(ax)
    axes[0][0].set_ylabel("Recall@1 (25 candidates)", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/real_world")
    p.add_argument("--figures", default="figures/real_world")
    args = p.parse_args()
    raw = {f.stem: json.loads(f.read_text()) for f in Path(args.results).glob("*.json")}
    if not raw:
        raise SystemExit(f"no results in {args.results}")
    results = {name: r["summary"] for name, r in raw.items()}
    n_candidates = next(iter(raw.values()))["n_candidates"]
    out = Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, out / "comparison.png", n_candidates)
    plot_scaling(results, out / "scaling.png")


if __name__ == "__main__":
    main()
