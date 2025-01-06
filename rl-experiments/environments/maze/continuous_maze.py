import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque
import random

"""
Defines the continuous maze environment used for experiments.
"""

def get_dir(cur, next, eps=10.):
  dir = np.array(next) - np.array(cur)
  # return dir
  angle = np.arctan2(dir[0], dir[1]) + np.random.normal() * eps * (2 * np.pi / 360)
  return np.array([np.sin(angle), np.cos(angle)])

def bfs(grid, start, end):
    """
    returns shortest path on a grid. Discrete
    """
    rows, cols = grid.shape
    queue = deque([start])
    visited = set()
    visited.add(start)
    parent = {start: None}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while queue:
        current = queue.popleft()

        if current == end:
            break

        random.shuffle(directions)
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 0 and neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current

    path = []

    if current != end:
      return []

    step = end

    while step is not None:
        path.append(step)
        step = parent[step]

    path.reverse()

    if path[0] == start:
        return path
    else:
        return []


def gen_traj(maze, start, end, eps=10.):
  """
  generate continuous trajectories
  """
  # print(f'in gen_traj')
  path = np.array(bfs(maze, start, end))
  # print(f'goal path: {path}')

  env = ContinuousGridEnvironment(maze, np.array(start) + 0.5, {})

  i = 0
  iters = 0

  continuous_path = [env.agent_position]
  trajectory = []

  while i < len(path) - 1:
    goal = tuple(path[i + 1])
    cur_state = tuple(path[i])
    iters += 1
    if iters > 1000:
      print(f'got really lost')
      print(continuous_path)
      break
    
    prev_pos = env.agent_position
    opt_dir = get_dir(env.agent_position, path[i + 1] + (0.5, 0.5), eps=eps)
    trajectory.append((env.agent_position, opt_dir))

    prev_gridpos = (int(env.agent_position[0]), int(env.agent_position[1]))
    env.move_agent(opt_dir)
    cur_gridpos = (int(env.agent_position[0]), int(env.agent_position[1]))
    if cur_gridpos == goal:
        continuous_path.append(env.agent_position)
        i += 1
        if i == len(path) - 1:
            break
    elif cur_gridpos == prev_gridpos:
       continuous_path.append(env.agent_position)
    else:
        # print(f'outside, we are at {env.agent_position}, ({cur_gridpos}), resetting... to {prev_pos}..., the goal is {goal} and the cur_state is {cur_state}')
        # print(f'error, stepped into {env.agent_position} from {prev_pos} but am aiming for {goal}. Current gridpos: {cur_gridpos}. cur_state: {cur_state}')
        env.agent_position = prev_pos

  trajectory.append((env.agent_position, np.array([1., 0.])))
  return trajectory


def plot_traj(maze, path):
  # Plotting the grid with the path highlighted
  fig, ax = plt.subplots(figsize=(10, 10))
  ax.imshow(maze, cmap=plt.cm.binary)

  # Highlight the path
  if path:
      path_x, path_y = zip(*path)
      ax.plot(np.array(path_y) - 0.5, np.array(path_x) - 0.5, marker='o', color='r', linewidth=2)

  # Mark the start and end points
  ax.plot(path_y[0] - 0.5, path_x[0] - 0.5, marker='o', color='g', markersize=10)
  ax.plot(path_y[-1] - 0.5, path_x[-1] - 0.5, marker='o', color='b', markersize=10)

  ax.set_xticks(np.arange(-0.5, maze.shape[1], 1))
  ax.set_yticks(np.arange(-0.5, maze.shape[0], 1))
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.grid(which='both', color='black', linestyle='-', linewidth=1)
  ax.set_title('Maze with Shortest Path Highlighted')

  plt.show()

# def get_trajectories(maze, num_trajectories, plot=False):
#   valid_indices = np.argwhere(maze == 0)
#   np.random.shuffle(valid_indices)
#   traj_ds = []
#   for _ in range(num_trajectories):
#     start, end = np.random.randint(0, len(valid_indices), size=2)
#     traj = gen_traj(maze, tuple(valid_indices[start]), tuple(valid_indices[end]))
#     if len(traj) > 0:
#       if plot:
#         plot_traj(maze, [x[0] for x in traj])
#       traj_ds.append(traj)

#   return traj_ds

def get_trajectories(maze, num_trajectories, order_fn=None, plot=False):
    if order_fn is not None:
        print(f'using order fn')
    valid_indices = np.argwhere(maze == 0)
    np.random.shuffle(valid_indices)
    traj_ds = []
    
    for _ in range(num_trajectories):
        start_idx, end_idx = np.random.randint(0, len(valid_indices), size=2)
        start = tuple(valid_indices[start_idx])
        end = tuple(valid_indices[end_idx])
        if order_fn is not None and order_fn(start, end) > 0:
            # print(f'swapping')
            start, end = end, start
        # print(start, end)
        
        traj = gen_traj(maze, start, end)
        # print(traj)
        if len(traj) > 0:
            if plot:
                plot_traj(maze, [x[0] for x in traj])
            traj_ds.append(traj)
    
    return traj_ds


class ContinuousGridEnvironment:
    """
    maze environment for RL
    """
    def __init__(self, grid, start, language_map):
        self.agent_position = np.array(start)
        self.grid = grid
        self.step_size = 0.5  # Step size for each move
        self.language_map = language_map

    def move_agent(self, dir):
        unit_dir = np.array(dir) / np.linalg.norm(dir)
        move_vector = self.step_size * unit_dir
        # Calculate new position
        new_position = self.agent_position + move_vector

        is_valid, new_position = self.calculate_move(new_position)

        self.agent_position = new_position

    def calculate_move(self, position):
        # Interpolate positions for finer collision detection
        num_steps = 25
        r_steps = np.linspace(self.agent_position[0], position[0], num_steps)
        c_steps = np.linspace(self.agent_position[1], position[1], num_steps)

        last_valid = self.agent_position
        for r, c in zip(r_steps, c_steps):
            if not self.is_within_bounds([r, c]) or self.is_wall([r, c]):
                return False, last_valid
            else:
              last_valid = (r, c)

        return True, last_valid

    def is_within_bounds(self, position):
        r, c = position
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]

    def is_wall(self, position):
        r, c = int(position[0]), int(position[1])  # Convert to int for grid indexing
        return self.grid[r, c] == 1

    def display(self, ax, highlight_col=0):
        # Create an overlay for highlighting the specified column
        color_overlay = np.zeros(self.grid.shape)

        # Highlight the specified column
        color_overlay[:, highlight_col] = 1  # Assign a unique color index for the highlighted column

        # Create a custom color map with only white and a highlight color (e.g., yellow)
        cmap = mcolors.ListedColormap(['white', 'yellow'])
        bounds = [0, 1, 2]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        ax.imshow(self.grid, cmap='gray', extent=[0, self.grid.shape[1], 0, self.grid.shape[0]])
        ax.imshow(color_overlay, cmap=cmap, norm=norm, alpha=0.5, extent=[0, self.grid.shape[1], 0, self.grid.shape[0]])
        ax.scatter(self.agent_position[1], self.grid.shape[0] - self.agent_position[0], c='red', s=100)  # Make the agent more visible
        ax.set_xlim(0, self.grid.shape[1])
        ax.set_ylim(0, self.grid.shape[0])
        ax.grid(True)


# Example usage:

# Create a 9x9 numpy array
maze = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 1, 0]
])



# l_map = {
#     'the bottom left': np.array([[0, 0], [0, 1], [0, 2]]),
#     'the top left': np.array([[8, 0], [8, 1], [8, 2], [8, 3]]),
#     'the bottom right': np.array([[0, 6], [0, 7], [0, 8]]),
# }

# env = ContinuousGridEnvironment(maze, [0., 0.], l_map)
# env.move_agent([0., 1.])  # Move agent at 45 degrees
# env.display()
