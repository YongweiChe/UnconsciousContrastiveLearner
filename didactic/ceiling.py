"""Reference accuracies for the synthetic task, computed from the true Gaussian densities.

    python -m didactic.ceiling --seeds 0-2 --sizes 5000 100000 1000000

Bayes-optimal: rank candidates by the exact log p(c | a) / p(c) (an upper bound for any method).
Exact Lemma 1: the Monte Carlo estimator with the exact log p(b | a)/p(b) and log p(c | b)/p(c)
in place of learned critics, i.e. what Monte Carlo achieves with perfect contrastive
representations; the gap to a trained model is error from Assumption 2. When Assumption 1 is
violated (bypass > 0) this estimator no longer targets p(c | a) / p(c), and falls below Bayes.
Uses the same data stream as didactic.run, so the validation triplets match.
"""

import argparse

import numpy as np

from .data import BridgeDistribution
from .exact import ExactReferences
from .run import BASE, parse_seeds


def ceiling(seed, cfg, sizes, bypass=0.0):
    dist = BridgeDistribution.random(np.random.default_rng(seed), cfg["dims"], cfg["noise"])
    dist.bypass = bypass
    data_rng = np.random.default_rng([seed, 0] + ([int(bypass * 1000)] if bypass else []))
    dist.sample(2 * cfg["train_size"], data_rng)  # advance past the training data, as in didactic.run
    a, _, c = dist.sample(cfg["val_size"], data_rng)
    ref = ExactReferences(dist, a, c)
    out = {"bayes": ref.bayes()}
    pool_rng = np.random.default_rng([seed, 99])
    for target in sorted(sizes):
        out[target] = ref.add_bridge(dist.sample(target - ref.seen, pool_rng)[1])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--sizes", type=int, nargs="+", default=[5000, 100000])
    p.add_argument("--bypass", type=float, default=0.0)
    p.add_argument("--train-size", type=int, default=BASE["train_size"], help="must match didactic.run to share val data")
    args = p.parse_args()
    cfg = {**BASE, "train_size": args.train_size}
    rows = [ceiling(seed, cfg, args.sizes, args.bypass) for seed in parse_seeds(args.seeds)]
    print(f"Bayes-optimal recall@1: {np.mean([r['bayes'] for r in rows]):.3f}")
    for m in sorted(args.sizes):
        print(f"exact Lemma 1 Monte Carlo, M = {m:>9,d}: {np.mean([r[m] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
