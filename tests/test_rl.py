import numpy as np

from rl.data import TrajectoryBuffer, sample_labels
from rl.evaluate import evaluate
from rl.maze import MAZES, ContinuousMaze, bfs_path, generate_trajectory, label_cells, satisfies
from rl.train import DEFAULTS, train


def test_bfs_routes_around_walls():
    grid = MAZES["fork"]()
    path = bfs_path(grid, (0, 0), [(2, 0)])  # row 1 is walled for columns 0-9
    assert path[0] == (0, 0) and path[-1] == (2, 0)
    assert all(grid[c] == 0 for c in path)
    assert len(path) - 1 == 2 * 10 + 2  # out to column 10, down two, back
    assert bfs_path(grid, (0, 0), [(1, 0)]) == []  # a wall is unreachable


def test_labels():
    grid = MAZES["fork"]()
    h = grid.shape[0]
    assert len(label_cells(grid, 1)) == 5  # row 1 is free only at columns 10-14
    assert satisfies(grid, (3.2, 7.9), 3) and satisfies(grid, (3.2, 7.9), h + 7)
    assert not satisfies(grid, (3.2, 7.9), 4)


def test_walls_block_movement():
    env = ContinuousMaze(MAZES["fork"](), (0.5, 0.5))
    for _ in range(5):
        env.move((1.0, 0.0))  # straight down into the wall of row 1
    assert int(env.position[0]) == 0


def test_trajectory_reaches_goal():
    rng = np.random.default_rng(0)
    grid = MAZES["fork"]()
    pos, act = generate_trajectory(grid, (0, 0), (14, 0), rng)
    assert tuple(pos[-1].astype(int)) == (14, 0)
    assert np.allclose(np.linalg.norm(act, axis=1), 1)


def test_samplers():
    rng = np.random.default_rng(0)
    grid = MAZES["island"]()
    buf = TrajectoryBuffer(grid, 50, rng)
    sa, fut, neg_fut, neg_act = buf.sample(32, 0.1, 8, rng)
    assert sa.shape == (32, 4) and fut.shape == (32, 2) and neg_fut.shape == (32, 8, 2) and neg_act.shape == (32, 8, 4)
    assert np.allclose(neg_act[..., :2], sa[:, None, :2])  # negative actions keep the state

    labels, states, neg_states, neg_labels = sample_labels(grid, 64, 8, rng)
    h = grid.shape[0]
    rows, cols = states[:, 0].astype(int), states[:, 1].astype(int) + h
    assert np.all((labels == rows) | (labels == cols))
    assert not np.any((neg_labels == rows[:, None]) | (neg_labels == cols[:, None]))
    assert np.all((0 <= neg_labels) & (neg_labels < sum(grid.shape)))


def test_train_and_evaluate_smoke():
    cfg = {**DEFAULTS, "n_trajectories": 100, "steps": 20, "batch_size": 32, "n_negatives": 8}
    agent = train("island", 0, cfg, log_every=10**9).eval()
    r = evaluate(agent, MAZES["island"](), n_trials=4, n_bridge=100, noise_deg=20.0, max_steps=20, seed=0)
    for method in ("direct", "lse"):
        assert len(r[method]["success"]) == 4
        assert all(0.0 <= s <= 1.0 for s in r[method]["spl"])
