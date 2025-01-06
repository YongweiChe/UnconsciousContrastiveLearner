import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from environments.maze.continuous_maze import get_trajectories

def set_collate_fn(batch):
    """
    Custom collate function to batch dynamic-sized inputs.
    """
    # Separate set1 and set2
    set1_list, set2_list = zip(*batch)

    # Convert to tensors
    set1_tensors = [torch.tensor(s, dtype=torch.float) for s in set1_list]
    set2_tensors = [torch.tensor(s, dtype=torch.float) for s in set2_list]

    # Pad sequences
    set1_padded = pad_sequence(set1_tensors, batch_first=True)
    set2_padded = pad_sequence(set2_tensors, batch_first=True)

    # Get lengths
    set1_lengths = torch.tensor([len(s) for s in set1_list])
    set2_lengths = torch.tensor([len(s) for s in set2_list])

    # Create masks
    batch_size, max_set_size = set1_padded.shape[:2]
    set1_mask = torch.arange(max_set_size).expand(batch_size, max_set_size) < set1_lengths.unsqueeze(1)
    
    batch_size, max_set_size = set2_padded.shape[:2]
    set2_mask = torch.arange(max_set_size).expand(batch_size, max_set_size) < set2_lengths.unsqueeze(1)

    return {
        'set1': set1_padded,
        'set2': set2_padded,
        'set1_mask': set1_mask.float(),
        'set2_mask': set2_mask.float(),
        'set1_lengths': set1_lengths,
        'set2_lengths': set2_lengths
    }

class TrajectoryDataset(Dataset):
    def __init__(self, maze, num_trajectories, num_negatives=10, gamma=0.1, order_fn=None):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.num_negatives = num_negatives
        self.maze = maze
        self.gamma = gamma
        print(f'gamma: {self.gamma}')

        self.trajectories = get_trajectories(maze, num_trajectories, order_fn=order_fn)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Anchor: the current data point
        
        traj = self.trajectories[idx]
        # print(traj)
        # start, end = np.sort(np.random.randint(0, len(traj), size=2))
        start = np.random.randint(0, len(traj))
        end = min(start + np.random.geometric(p=self.gamma), len(traj) - 1)

        anchor = self.trajectories[idx][start]
        anchor = np.array([np.array(anchor[0]), np.array(anchor[1])]).flatten()
        # Label of the anchor
        positive_example = self.trajectories[idx][end][0]

        negative_examples = []
        for i in range(self.num_negatives):
          idy = np.random.randint(0, len(self.trajectories))
          neg_state = self.trajectories[idy][np.random.randint(0, len(self.trajectories[idy]))][0]
          negative_examples.append(neg_state)

        return anchor, np.array(positive_example), np.array(negative_examples)

class LabelDataset(Dataset):
    def __init__(self, maze, size=100, num_negatives=10):
        super().__init__()
        self.num_categories = sum(maze.shape)
        self.num_examples = 2 * maze.shape[0] * maze.shape[1]
        self.num_negatives = num_negatives
        self.maze = maze
        pairs = []
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
              for _ in range(size):
                point1 = np.array((i + np.random.uniform(0, 1), j + np.random.uniform(0, 1)))
                point2 = np.array((i + np.random.uniform(0, 1), j + np.random.uniform(0, 1)))
                pairs.append((i, point1))
                pairs.append(((maze.shape[0] + j), point2))

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)
    
    def get_rowcol(self, point):
       point = (int(point[0]), int(point[1]))
       return point[0], point[1] + self.maze.shape[0]

    def __getitem__(self, idx):
        negatives = []
        while len(negatives) != self.num_negatives:
          i = np.random.randint(0, len(self.pairs))
          if self.pairs[i][0] != self.pairs[idx][0]:
            negatives.append(self.pairs[i][1])

        target_row, target_col = self.get_rowcol(self.pairs[idx][1])
            
        all_categories = list(range(self.num_categories))
        
        all_categories.remove(target_row)
        if target_col != target_row: 
            all_categories.remove(target_col)
        
        negative_categories = np.random.choice(all_categories, size=self.num_negatives, replace=True)

        return self.pairs[idx][0], self.pairs[idx][1], np.array(negatives), negative_categories

class SetDataset(Dataset):
    """
    Returns positive pairs of set-valued outputs from the same trajectory
    """
    def __init__(
        self,
        maze,
        num_trajectories,
        embedding_dim=2,
        num_negatives=10,
        gamma=0.1,
        order_fn=None,
        num_splits=4,
        padding_len=100,
    ):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.num_negatives = num_negatives
        self.num_splits = num_splits
        self.padding_len = padding_len
        self.maze = maze
        self.gamma = gamma
        print(f"gamma: {self.gamma}")

        self.trajectories = get_trajectories(maze, num_trajectories, order_fn=order_fn)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Anchor: the current data point

        traj = self.trajectories[idx]
        split_traj = np.split(
            traj, np.random.randint(0, len(traj), size=self.num_splits)
        )
        split_traj = list(filter(lambda x: x.shape[0] != 0, split_traj))
        # print(split_traj)

        i, j = np.random.randint(0, len(split_traj), 2)
        # print(i, j)

        set1 = np.stack([x[0] for x in split_traj[i]])
        set2 = np.stack([x[0] for x in split_traj[j]])

        return set1, set2