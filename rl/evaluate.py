"""Navigates to row/column labels with Direct vs LogSumExp action selection (Sec. 6.3, Fig. 13).

    python -m rl.evaluate --maze fork --seeds 0 1 2

At each step the agent scores 8 directions a:
    Direct:      f(phi_A(s, a), phi_C(label))
    LogSumExp:   log mean_m exp(f(phi_A(s, a), phi_B(s_m)) + f(phi_B(s_m), phi_C(label))),
                 with s_m uniform over the free area of the maze (Lemma 1)
and moves in the best direction plus Gaussian angle noise. Both policies see the same
starts, labels and noise. Success: reaching the label within max_steps. SPL: success
weighted by shortest-path length / path length.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from ucl import direct_scores, mc_scores

from .maze import MAZES, STEP_SIZE, ContinuousMaze, bfs_path, label_cells, sample_free_points, satisfies
from .models import Agent

ANGLES = np.linspace(0, 2 * np.pi, 8, endpoint=False)
DIRECTIONS = np.stack([np.sin(ANGLES), np.cos(ANGLES)], 1).astype(np.float32)


def load_agent(path):
    blob = torch.load(path, weights_only=False)
    grid = MAZES[blob["maze"]]()
    cfg = blob["config"]
    agent = Agent(sum(grid.shape), cfg["embed_dim"], cfg["temperature"])
    agent.load_state_dict(blob["state_dict"])
    return agent.eval(), grid


@torch.no_grad()
def run_episode(agent, grid, start, label, method, bridge, noise, max_steps):
    env = ContinuousMaze(grid, start)
    z_label = agent.label(torch.tensor([label]))
    z_bridge = agent.state(bridge) if method == "lse" else None
    for step in range(max_steps):
        if satisfies(grid, env.position, label):
            return True, step
        sa = torch.from_numpy(np.concatenate([np.repeat(env.position[None].astype(np.float32), 8, 0), DIRECTIONS], 1))
        z_sa = agent.state_action(sa)
        if method == "direct":
            scores = direct_scores(z_sa, z_label, "neg_sq_l2", agent.scale)
        else:
            scores = mc_scores(z_sa, z_label, z_bridge, z_bridge, "neg_sq_l2", "neg_sq_l2", agent.scale, agent.scale)
        angle = ANGLES[int(scores[:, 0].argmax())] + noise[step]
        env.move(np.array([np.sin(angle), np.cos(angle)]))
    return satisfies(grid, env.position, label), max_steps


def evaluate(agent, grid, n_trials, n_bridge, noise_deg, max_steps, seed):
    rng = np.random.default_rng(seed)
    labels = [l for l in range(sum(grid.shape)) if label_cells(grid, l)]
    results = {"direct": {"success": [], "spl": []}, "lse": {"success": [], "spl": []}}
    while len(results["direct"]["success"]) < n_trials:
        start = sample_free_points(grid, 1, rng)[0]
        label = int(rng.choice(labels))
        if satisfies(grid, start, label):
            continue  # trivially solved
        shortest = len(bfs_path(grid, tuple(start.astype(int)), label_cells(grid, label))) - 1
        if shortest < 0:
            continue  # unreachable
        noise = np.deg2rad(noise_deg) * rng.standard_normal(max_steps)
        bridge = torch.from_numpy(sample_free_points(grid, n_bridge, rng).astype(np.float32))
        for method in ("direct", "lse"):
            ok, steps = run_episode(agent, grid, start, label, method, bridge, noise, max_steps)
            path = steps * STEP_SIZE
            results[method]["success"].append(float(ok))
            results[method]["spl"].append(shortest / max(path, shortest, 1e-9) if ok else 0.0)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maze", choices=sorted(MAZES), default="fork")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--results", default="results/rl")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--n-bridge", type=int, default=5000, help="bridge states s_m for LogSumExp")
    p.add_argument("--noise-deg", type=float, default=20.0)
    p.add_argument("--max-steps", type=int, default=100)
    args = p.parse_args()

    pooled = {"direct": {"success": [], "spl": []}, "lse": {"success": [], "spl": []}}
    for seed in args.seeds:
        agent, grid = load_agent(Path(args.results) / args.maze / f"seed{seed}.pt")
        r = evaluate(agent, grid, args.n_trials, args.n_bridge, args.noise_deg, args.max_steps, seed=1000 + seed)
        for m in r:
            for k in r[m]:
                pooled[m][k].extend(r[m][k])
        print(f"seed {seed}: " + "   ".join(f"{m} success {np.mean(r[m]['success']):.3f} SPL {np.mean(r[m]['spl']):.3f}" for m in r))

    summary = {m: {k: {"mean": float(np.mean(v)), "sem": float(stats.sem(v))} for k, v in r.items()} for m, r in pooled.items()}
    print(f"\n{args.maze}: {len(args.seeds)} seeds x {args.n_trials} episodes")
    for m, name in (("direct", "Direct"), ("lse", "LogSumExp")):
        s = summary[m]
        print(f"  {name:10s} success {s['success']['mean']:.3f} ± {s['success']['sem']:.3f}   SPL {s['spl']['mean']:.3f} ± {s['spl']['sem']:.3f}")
    out = Path(args.results) / args.maze / "evaluation.json"
    out.write_text(json.dumps(dict(vars(args), summary=summary), indent=1))


if __name__ == "__main__":
    main()
