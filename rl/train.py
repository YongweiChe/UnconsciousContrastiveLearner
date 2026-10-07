"""Trains the language-conditioned contrastive RL agent of Sec. 6.3 / App. D.

    python -m rl.train --maze fork --seed 0

Learns phi_A(s, a) <-> phi_B(s_f) (contrastive RL: future states of the same expert
trajectory) and phi_B(s) <-> phi_C(label) (row/column labels of a state), sharing phi_B.
(s, a) and labels are never paired. Saves results/rl/<maze>/seed<seed>.pt.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .data import TrajectoryBuffer, sample_labels
from .maze import MAZES
from .models import Agent, info_nce

DEFAULTS = dict(n_trajectories=20000, steps=20000, batch_size=256, n_negatives=64, gamma=0.1, embed_dim=8,
                temperature=0.5, lr=0.01)


def train(maze_name, seed, cfg, device="cpu", log_every=1000):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    grid = MAZES[maze_name]()
    t0 = time.time()
    buffer = TrajectoryBuffer(grid, cfg["n_trajectories"], rng)
    print(f"{maze_name}: {cfg['n_trajectories']} trajectories, {len(buffer)} steps ({time.time() - t0:.0f}s)", flush=True)

    agent = Agent(sum(grid.shape), cfg["embed_dim"], cfg["temperature"]).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg["lr"])
    t = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
    for step in range(1, cfg["steps"] + 1):
        sa, fut, neg_fut, neg_act = (t(x) for x in buffer.sample(cfg["batch_size"], cfg["gamma"], cfg["n_negatives"], rng))
        z_sa, z_fut = agent.state_action(sa), agent.state(fut)
        rl_loss = info_nce(agent.scale, z_sa, z_fut, agent.state(neg_fut)) + info_nce(agent.scale, z_fut, z_sa, agent.state_action(neg_act))

        lab, st, neg_st, neg_lab = sample_labels(grid, cfg["batch_size"], cfg["n_negatives"], rng)
        z_lab, z_st = agent.label(t(lab, torch.long)), agent.state(t(st))
        label_loss = info_nce(agent.scale, z_lab, z_st, agent.state(t(neg_st))) + info_nce(agent.scale, z_st, z_lab, agent.label(t(neg_lab, torch.long)))

        loss = rl_loss + label_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(f"  step {step:6d}  rl loss {rl_loss.item():.3f}  label loss {label_loss.item():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return agent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maze", choices=sorted(MAZES), default="fork")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/rl")
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=1000)
    for key, value in DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = p.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    agent = train(args.maze, args.seed, cfg, args.device, args.log_every)
    path = Path(args.out) / args.maze / f"seed{args.seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"maze": args.maze, "seed": args.seed, "config": cfg, "state_dict": agent.state_dict()}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
