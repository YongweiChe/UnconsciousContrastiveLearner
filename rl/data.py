"""Vectorized sampling of contrastive training batches from expert trajectories.

Two paired datasets, as in Sec. 6.3:
    (s, a) <-> s_f   state-action and a future state of the same trajectory
    s      <-> l     a state and one of its row/column labels
"""

import numpy as np

from .maze import generate_trajectory


class TrajectoryBuffer:
    def __init__(self, grid, n_trajectories, rng):
        cells = [tuple(c) for c in np.argwhere(grid == 0)]
        positions, actions, traj_end, starts, lengths = [], [], [], [], []
        n = 0
        while len(starts) < n_trajectories:
            start, goal = (cells[i] for i in rng.integers(len(cells), size=2))
            pos, act = generate_trajectory(grid, start, goal, rng)
            positions.append(pos)
            actions.append(act)
            starts.append(n)
            lengths.append(len(pos))
            n += len(pos)
            traj_end.extend([n - 1] * len(pos))
        self.positions = np.concatenate(positions).astype(np.float32)
        self.actions = np.concatenate(actions).astype(np.float32)
        self.traj_end = np.array(traj_end)  # index of the last step of each step's trajectory
        self.starts, self.lengths = np.array(starts), np.array(lengths)

    def __len__(self):
        return len(self.positions)

    def sample(self, batch_size, gamma, n_negatives, rng):
        """Returns (state_action [B, 4], future [B, 2], negative futures [B, K, 2], negative actions [B, K, 4])."""
        traj = rng.integers(len(self.starts), size=batch_size)  # uniform over trajectories, then steps
        idx = self.starts[traj] + (rng.uniform(size=batch_size) * self.lengths[traj]).astype(int)
        future = np.minimum(idx + rng.geometric(gamma, size=batch_size), self.traj_end[idx])
        state_action = np.concatenate([self.positions[idx], self.actions[idx]], 1)
        neg_future = self.positions[rng.integers(len(self), size=(batch_size, n_negatives))]
        # Negative actions: directions within +-135 degrees of the opposite of the taken action.
        angle = np.arctan2(self.actions[idx, 0], self.actions[idx, 1])
        neg_angle = angle[:, None] + np.pi + (rng.uniform(size=(batch_size, n_negatives)) - 0.5) * 1.5 * np.pi
        neg_dirs = np.stack([np.sin(neg_angle), np.cos(neg_angle)], -1)
        neg_actions = np.concatenate([np.repeat(self.positions[idx][:, None], n_negatives, 1), neg_dirs], -1)
        return state_action, self.positions[future], neg_future, neg_actions.astype(np.float32)


def sample_labels(grid, batch_size, n_negatives, rng):
    """Returns (labels [B], states [B, 2], negative states [B, K, 2], negative labels [B, K]).

    Each state is uniform in a random grid cell and paired with that cell's row or column
    label; negative states have a different label, negative labels are neither of the
    state's two labels.
    """
    h, w = grid.shape
    cells = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), -1).reshape(-1, 2)

    def cell_points(n):
        c = cells[rng.integers(len(cells), size=n)]
        return c, (c + rng.uniform(size=c.shape)).astype(np.float32)

    cell, states = cell_points(batch_size)
    use_col = rng.uniform(size=batch_size) < 0.5
    labels = np.where(use_col, h + cell[:, 1], cell[:, 0])

    neg_cell, neg_states = cell_points(batch_size * n_negatives * 2)
    neg_cell, neg_states = neg_cell.reshape(batch_size, -1, 2), neg_states.reshape(batch_size, -1, 2)
    neg_label_match = np.where(use_col[:, None], neg_cell[..., 1] + h == labels[:, None], neg_cell[..., 0] == labels[:, None])
    order = np.argsort(neg_label_match, axis=1, kind="stable")[:, :n_negatives]  # prefer non-matching cells
    neg_states = np.take_along_axis(neg_states, order[..., None], 1)

    n_labels = h + w
    neg_labels = rng.integers(n_labels - 2, size=(batch_size, n_negatives))
    own = np.sort(np.stack([cell[:, 0], h + cell[:, 1]], 1), 1)
    neg_labels += (neg_labels >= own[:, :1]).astype(int)  # skip both of the state's own labels
    neg_labels += (neg_labels >= own[:, 1:]).astype(int)
    return labels, states, neg_states, neg_labels
