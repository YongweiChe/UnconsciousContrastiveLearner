"""Continuous grid mazes with "street" labels (Sec. 6.3, App. D).

A maze is a 0/1 grid (1 = wall). The agent lives in continuous coordinates
(row, col) and moves 0.5 per step in any direction. Every row and every column
has a language label: rows are labels 0..H-1, columns are labels H..H+W-1, so
the label "row 3" is satisfied anywhere in grid row 3.
"""

from collections import deque

import numpy as np

STEP_SIZE = 0.5


def _fork():
    maze = np.zeros((15, 15))
    for i in range(1, 14, 2):
        maze[i, :10] = 1
    return maze


def _island():
    maze = np.zeros((11, 11))
    maze[3:8, 5:7] = 1
    return maze


def _slit():
    maze = np.zeros((11, 11))
    maze[:, 5] = 1
    maze[5, 5] = 0
    return maze


def _blocker():
    maze = np.zeros((11, 11))
    maze[3, :10] = 1
    return maze


MAZES = {"blank": lambda: np.zeros((10, 10)), "fork": _fork, "island": _island, "slit": _slit, "blocker": _blocker}


class ContinuousMaze:
    def __init__(self, grid, start):
        self.grid = grid
        self.position = np.asarray(start, dtype=float)

    def free(self, p):
        r, c = p
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1] and self.grid[int(r), int(c)] == 0

    def move(self, direction):
        """Moves STEP_SIZE along direction, stopping at the last free point before a wall."""
        target = self.position + STEP_SIZE * np.asarray(direction) / np.linalg.norm(direction)
        last = self.position
        for t in np.linspace(0, 1, 25)[1:]:
            p = self.position + t * (target - self.position)
            if not self.free(p):
                break
            last = p
        self.position = np.array(last)


def bfs_path(grid, start, goals):
    """Shortest 4-connected cell path from start to the nearest cell in goals ([] if unreachable)."""
    goals = set(goals)
    parent = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur in goals:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1] and grid[nxt] == 0 and nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    return []


def label_cells(grid, label):
    """Free cells satisfying a row/column label."""
    h = grid.shape[0]
    cells = np.argwhere(grid == 0)
    keep = cells[:, 0] == label if label < h else cells[:, 1] == label - h
    return [tuple(c) for c in cells[keep]]


def satisfies(grid, position, label):
    h = grid.shape[0]
    return int(position[0]) == label if label < h else int(position[1]) == label - h


def sample_free_points(grid, n, rng):
    """n points uniform over the free area of the maze."""
    cells = np.argwhere(grid == 0)
    return cells[rng.integers(len(cells), size=n)] + rng.uniform(0, 1, size=(n, 2))


def generate_trajectory(grid, start, goal, rng, noise_deg=10.0, max_iters=1000):
    """Noisy expert trajectory following the BFS cell path from start to goal.

    Returns (positions [T, 2], actions [T, 2]) with actions as unit (sin, cos) vectors.
    """
    path = bfs_path(grid, start, [goal])
    env = ContinuousMaze(grid, np.array(start) + 0.5)
    positions, actions = [], []
    i = 0
    for _ in range(max_iters):
        if i >= len(path) - 1:
            break
        d = np.array(path[i + 1]) + 0.5 - env.position
        angle = np.arctan2(d[0], d[1]) + np.deg2rad(noise_deg) * rng.standard_normal()
        action = np.array([np.sin(angle), np.cos(angle)])
        positions.append(env.position.copy())
        actions.append(action)
        prev_cell, before = tuple(env.position.astype(int)), env.position.copy()
        env.move(action)
        cell = tuple(env.position.astype(int))
        if cell == path[i + 1]:
            i += 1
        elif cell != prev_cell:  # drifted into a neighboring cell off the path: undo
            env.position = before
    positions.append(env.position.copy())
    actions.append(np.array([1.0, 0.0]))
    return np.array(positions), np.array(actions)
