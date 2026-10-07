"""Runs the synthetic experiments of Sec. 6.1, 6.2 and Appendix C.3/C.4.

    python -m didactic.run --experiment main --seeds 0-19
    python -m didactic.run --experiment ci_ablation --seeds 0-4
    python -m didactic.run --experiment embed_2d --seeds 0-4

Each (seed, corr_noise, bypass) writes results/didactic/<experiment>/seed<seed>_corr<corr>[_bypass<rho>].json
(config, per-epoch recall curves for every model, learned logit scales, optional exact
references) and a matching .npz with embeddings of held-out samples for didactic.report.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import BridgeDistribution
from .exact import ExactReferences
from .models import ContrastiveModel
from .train import evaluate, train

ALL_CRITICS = ["cosine", "dot", "neg_sq_l2"]
ALL_KINDS = ["unconscious", "disparate", "ground_truth"]

BASE = dict(
    dims=(32, 16, 32),
    noise=(2.0, 2.0),
    corr_noises=[0.0],
    bypasses=[0.0],
    exact_reference=False,  # also record Bayes-optimal and exact-Lemma-1 recall (didactic.exact)
    critics=ALL_CRITICS,
    kinds=ALL_KINDS,
    embed_dim=32,
    hidden=128,  # MLP width
    temperature=1.0,
    epochs=64,
    batch_size=256,
    lr=0.01,
    lr_step=60,
    lr_gamma=0.1,
    norm_penalty=0.0,
    train_size=20000,  # per pair
    val_size=1000,
    bridge_pool_size=5000,  # Monte Carlo samples for the per-epoch curves
    final_bridge_pool_size=100000,  # and for one extra evaluation of the final model
    n_saved_embeddings=500,
)

PRESETS = {
    # Fig. 2 (all critics, Direct vs Monte Carlo vs Ground Truth) and Fig. 3 (disparate).
    "main": {},
    # Fig. 8: violate Assumption 1 by routing a fraction rho of the A-C signal around B.
    "ci_ablation": dict(bypasses=[0.0, 0.25, 0.5, 0.75, 1.0], kinds=["unconscious", "ground_truth"], exact_reference=True),
    # Fig. 7: 2-d embeddings to visualize the representation distribution.
    "embed_2d": dict(embed_dim=2, kinds=["unconscious"]),
}


def run_one(cfg, seed, corr_noise, bypass, device):
    rng = np.random.default_rng(seed)
    dist = BridgeDistribution.random(rng, cfg["dims"], cfg["noise"])  # shared across corr / bypass levels
    dist.corr_noise, dist.bypass = corr_noise, bypass
    data_rng = np.random.default_rng([seed, int(corr_noise * 1000)] + ([int(bypass * 1000)] if bypass else []))
    to_t = lambda arrs: tuple(torch.from_numpy(x).to(device) for x in arrs)
    train_data = to_t(dist.sample(2 * cfg["train_size"], data_rng))
    val = to_t(dist.sample(cfg["val_size"], data_rng))
    big_pool = to_t(dist.sample(max(cfg["bridge_pool_size"], cfg["final_bridge_pool_size"]), data_rng))[1]
    bridge_pool = big_pool[: cfg["bridge_pool_size"]]
    exact = None
    if cfg["exact_reference"]:
        ref = ExactReferences(dist, val[0].cpu().numpy(), val[2].cpu().numpy())
        exact = {"bayes": ref.bayes(), "lemma1": ref.add_bridge(big_pool[: cfg["final_bridge_pool_size"]].cpu().numpy())}
        print(f"  exact: Bayes {exact['bayes']:.3f}  Lemma 1 @ M={cfg['final_bridge_pool_size']} {exact['lemma1']:.3f}", flush=True)

    curves, scales, embeddings = {}, {}, {}
    for critic in cfg["critics"]:
        for kind in cfg["kinds"]:
            name = f"{kind}/{critic}"
            torch.manual_seed(seed)
            model = ContrastiveModel(kind, dist.dims, cfg["embed_dim"], critic, cfg["temperature"], cfg["hidden"]).to(device)
            t0 = time.time()
            curves[name] = train(
                model,
                train_data,
                val,
                bridge_pool,
                pair_size=cfg["train_size"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                lr=cfg["lr"],
                lr_step=cfg["lr_step"],
                lr_gamma=cfg["lr_gamma"],
                norm_penalty=cfg["norm_penalty"],
                generator=torch.Generator().manual_seed(seed),
            )
            curves[name]["mc_final_large"] = evaluate(model, val, big_pool[: cfg["final_bridge_pool_size"]])["mc"]
            scales[name] = {k: v.exp().item() for k, v in model.log_scales.items()}
            model.eval()
            with torch.no_grad():
                k = cfg["n_saved_embeddings"]
                index = {"A": 0, "B": 1, "B1": 1, "B2": 1, "C": 2}
                for enc in model.encoders:
                    embeddings[f"{name}/{enc}"] = model.embed(enc, val[index[enc]][:k]).cpu().numpy()
            c = curves[name]
            mc = f"{c['mc'][-1]:.3f}  mc@{cfg['final_bridge_pool_size']} {c['mc_final_large']:.3f}" if c["mc"][-1] is not None else "  -  "
            print(f"  {name:28s} direct {c['direct'][-1]:.3f}  mc {mc}  ({time.time() - t0:.1f}s)", flush=True)
    return dist, curves, scales, embeddings, exact


def parse_seeds(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=sorted(PRESETS), default="main")
    p.add_argument("--seeds", default="0-19", help="e.g. 0-19 or 0,3,7")
    p.add_argument("--out", default="results/didactic")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--critics", nargs="+", choices=ALL_CRITICS)
    p.add_argument("--kinds", nargs="+", choices=ALL_KINDS)
    p.add_argument("--corr-noises", nargs="+", type=float)
    p.add_argument("--bypasses", nargs="+", type=float)
    for key in ["embed_dim", "hidden", "epochs", "lr_step", "batch_size", "train_size", "val_size",
                "bridge_pool_size", "final_bridge_pool_size"]:
        p.add_argument(f"--{key.replace('_', '-')}", type=int)
    for key in ["temperature", "lr", "norm_penalty"]:
        p.add_argument(f"--{key.replace('_', '-')}", type=float)
    args = p.parse_args()

    cfg = {**BASE, **PRESETS[args.experiment]}
    for key in list(cfg):
        if getattr(args, key, None) is not None:
            cfg[key] = getattr(args, key)
    for key in ("corr_noises", "bypasses"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    out_dir = Path(args.out) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed, corr, bypass in itertools.product(parse_seeds(args.seeds), cfg["corr_noises"], cfg["bypasses"]):
        print(f"seed {seed}  corr_noise {corr}  bypass {bypass}", flush=True)
        dist, curves, scales, embeddings, exact = run_one(cfg, seed, corr, bypass, args.device)
        stem = out_dir / (f"seed{seed}_corr{corr:g}" + (f"_bypass{bypass:g}" if bypass else ""))
        record = dict(experiment=args.experiment, seed=seed, corr_noise=corr, bypass=bypass, config=cfg,
                      curves=curves, scales=scales, exact=exact)
        # Append extensions rather than Path.with_suffix, which would treat ".5" in "bypass0.5" as one.
        Path(f"{stem}.json").write_text(json.dumps(record, indent=1))
        np.savez_compressed(f"{stem}.npz", **embeddings)


if __name__ == "__main__":
    main()
