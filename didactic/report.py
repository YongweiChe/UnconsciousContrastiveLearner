"""Summarizes didactic.run results: a table of final recall@1 and the paper figures.

    python -m didactic.report --experiment main          # Fig. 2, Fig. 3
    python -m didactic.report --experiment ci_ablation   # Fig. 8 (bridge bypass)
    python -m didactic.report --experiment embed_2d      # Fig. 7
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CRITIC_TITLES = {
    "cosine": r"$f = \phi_1^\top\phi_2 / \|\phi_1\|\|\phi_2\|$",
    "dot": r"$f = \phi_1^\top\phi_2$",
    "neg_sq_l2": r"$f = -\frac{1}{2}\|\phi_1 - \phi_2\|^2$",
}
# Fixed categorical order (validated: blue, orange, aqua); line style is the secondary encoding.
METHOD_STYLE = {
    "Direct": dict(color="#2a78d6", linestyle="-"),
    "Monte Carlo": dict(color="#eb6834", linestyle="--"),
    "Ground Truth": dict(color="#1baf7a", linestyle=":"),
}
INK, MUTED = "#2b2b2b", "#8a8a85"


def load(results_dir):
    records = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    if not records:
        raise SystemExit(f"no results in {results_dir}")
    return records


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)


def method_curves(records, kind, critic, corr=None, bypass=None):
    """{method: [n_seeds, n_epochs] array} for one model kind and critic."""
    out = defaultdict(list)
    for r in records:
        if (corr is not None and r["corr_noise"] != corr) or (bypass is not None and r.get("bypass", 0.0) != bypass):
            continue
        c = r["curves"].get(f"{kind}/{critic}")
        if c is not None:
            out["Direct"].append(c["direct"])
            if c["mc"][0] is not None:
                out["Monte Carlo"].append(c["mc"])
        gt = r["curves"].get(f"ground_truth/{critic}")
        if gt is not None:
            out["Ground Truth"].append(gt["direct"])
    return {k: np.array(v) for k, v in out.items()}


def plot_training_curves(records, kind, critics, path, title):
    fig, axes = plt.subplots(1, len(critics), figsize=(4.2 * len(critics), 3.4), sharey=True, squeeze=False)
    for ax, critic in zip(axes[0], critics):
        for method, arr in method_curves(records, kind, critic, corr=0.0, bypass=0.0).items():
            mean, std = arr.mean(0), arr.std(0)
            epochs = np.arange(len(mean))
            ax.plot(epochs, mean, linewidth=1.5, label=method, **METHOD_STYLE[method])
            ax.fill_between(epochs, mean - std, mean + std, color=METHOD_STYLE[method]["color"], alpha=0.15, linewidth=0)
        ax.set_title(CRITIC_TITLES[critic], color=INK, fontsize=11)
        ax.set_xlabel("Epoch", color=INK)
        ax.set_ylim(0, 1)
        style_axes(ax)
    axes[0][0].set_ylabel("Recall@1 (32 candidates)", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, color=INK, y=1.09)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


REFERENCE_STYLE = {
    "Bayes-optimal": dict(color="#2b2b2b", linestyle="-", marker="s"),
    "Exact Lemma 1": dict(color="#8a8a85", linestyle="-.", marker="^"),
}


def plot_bypass(records, critics, path):
    """Fig. 8: final recall vs the fraction rho of the A-C signal that bypasses the bridge B."""
    rhos = sorted({r["bypass"] for r in records})
    m_large = records[0]["config"]["final_bridge_pool_size"]
    by_rho = {rho: [r for r in records if r["bypass"] == rho] for rho in rhos}
    refs = {
        "Bayes-optimal": [[r["exact"]["bayes"] for r in by_rho[rho]] for rho in rhos],
        "Exact Lemma 1": [[r["exact"]["lemma1"] for r in by_rho[rho]] for rho in rhos],
    }
    fig, axes = plt.subplots(1, len(critics), figsize=(4.2 * len(critics), 3.6), sharey=True, squeeze=False)
    for ax, critic in zip(axes[0], critics):
        series = {
            "Direct": [[r["curves"][f"unconscious/{critic}"]["direct"][-1] for r in by_rho[rho]] for rho in rhos],
            "Monte Carlo": [[r["curves"][f"unconscious/{critic}"]["mc_final_large"] for r in by_rho[rho]] for rho in rhos],
            "Ground Truth": [[r["curves"][f"ground_truth/{critic}"]["direct"][-1] for r in by_rho[rho]] for rho in rhos],
        }
        for name, vals in {**refs, **series}.items():
            style = REFERENCE_STYLE.get(name) or {**METHOD_STYLE[name], "marker": "o"}
            mean, std = np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals])
            ax.errorbar(rhos, mean, yerr=std, linewidth=1.5 if name in METHOD_STYLE else 1.0, markersize=5,
                        capsize=3, label=name, **style)
        ax.axhline(1 / 32, color=MUTED, linewidth=0.8, linestyle=":")
        ax.text(rhos[0], 1 / 32 + 0.015, "chance", color=MUTED, fontsize=8)
        ax.set_title(CRITIC_TITLES[critic], color=INK, fontsize=11)
        ax.set_xlabel(r"fraction of A$-$C signal bypassing B ($\rho$)", color=INK)
        ax.set_xticks(rhos)
        ax.set_ylim(0, 1.02)
        style_axes(ax)
    axes[0][0].set_ylabel("Final recall@1 (32 candidates)", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Violating conditional independence (Assumption 1); Monte Carlo and Lemma 1 at M = {m_large:,}",
                 color=INK, y=1.10)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def plot_embeddings_2d(results_dir, critics, path, seed=0):
    emb = np.load(results_dir / f"seed{seed}_corr0.npz")
    fig, axes = plt.subplots(1, len(critics), figsize=(3.6 * len(critics), 3.6), squeeze=False)
    for ax, critic in zip(axes[0], critics):
        z = emb[f"unconscious/{critic}/B"]
        if critic == "cosine":  # the critic only sees directions
            z = z / np.linalg.norm(z, axis=1, keepdims=True)
        ax.scatter(z[:, 0], z[:, 1], s=8, color=METHOD_STYLE["Direct"]["color"], alpha=0.6, linewidths=0)
        ax.set_title(CRITIC_TITLES[critic], color=INK, fontsize=11)
        ax.set_aspect("equal", adjustable="datalim")
        style_axes(ax)
    fig.suptitle(r"$\phi_B$ of held-out samples (seed %d)" % seed, color=INK, y=1.06)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def print_table(records):
    rows = defaultdict(lambda: defaultdict(list))
    for r in records:
        for name, c in r["curves"].items():
            key = (r["corr_noise"], r.get("bypass", 0.0), name)
            rows[key]["direct"].append(c["direct"][-1])
            if c["mc"][-1] is not None:
                rows[key]["mc"].append(c["mc"][-1])
            if c.get("mc_final_large") is not None:
                rows[key]["mc_large"].append(c["mc_final_large"])
            for pair, s in r["scales"][name].items():
                rows[key][f"scale {pair}"].append(s)
    fmt = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f}" if xs else "-"
    m_large = records[0]["config"].get("final_bridge_pool_size")
    large = f"MC @ M={m_large}" if m_large else ""
    print(f"\n{'corr':>5} {'rho':>5}  {'model':28s} {'direct':>15} {'monte carlo':>15} {large:>15}   learned logit scales  (n={len(records)} runs)")
    for (corr, rho, name), v in sorted(rows.items()):
        scales = "  ".join(f"{k[6:]}={np.mean(x):.2f}" for k, x in v.items() if k.startswith("scale"))
        print(f"{corr:5g} {rho:5g}  {name:28s} {fmt(v['direct']):>15} {fmt(v.get('mc', [])):>15} {fmt(v.get('mc_large', [])):>15}   {scales}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="main")
    p.add_argument("--results", default="results/didactic")
    p.add_argument("--figures", default="figures/didactic")
    args = p.parse_args()

    results_dir = Path(args.results) / args.experiment
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    records = load(results_dir)
    critics = records[0]["config"]["critics"]
    kinds = records[0]["config"]["kinds"]
    print_table(records)

    if args.experiment == "ci_ablation":
        plot_bypass(records, critics, fig_dir / "fig8_ci_bypass.png")
        exact = [r for r in records if r.get("exact")]
        for rho in sorted({r["bypass"] for r in exact}):
            rs = [r["exact"] for r in exact if r["bypass"] == rho]
            print(f"rho {rho:g}: Bayes-optimal {np.mean([x['bayes'] for x in rs]):.3f}  exact Lemma 1 {np.mean([x['lemma1'] for x in rs]):.3f}")
    elif args.experiment == "embed_2d":
        plot_embeddings_2d(results_dir, critics, fig_dir / "embeddings_2d.png")
    else:
        if "unconscious" in kinds:
            plot_training_curves(records, "unconscious", critics, fig_dir / f"{args.experiment}_unconscious.png",
                                 "Shared bridge encoder (Fig. 2)")
        if "disparate" in kinds:
            plot_training_curves(records, "disparate", critics, fig_dir / f"{args.experiment}_disparate.png",
                                 "Independent A-B and B-C models (Fig. 3)")


if __name__ == "__main__":
    main()
