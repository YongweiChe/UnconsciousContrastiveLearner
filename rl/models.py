import torch
import torch.nn.functional as F
from torch import nn


def mlp(input_dim, embed_dim, width=64):
    return nn.Sequential(
        nn.Linear(input_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, embed_dim),
    )


class LabelEncoder(nn.Module):
    def __init__(self, n_labels, embed_dim, width=64):
        super().__init__()
        self.embedding = nn.Embedding(n_labels, embed_dim)
        self.net = nn.Sequential(nn.Linear(embed_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, embed_dim))

    def forward(self, labels):
        return self.net(self.embedding(labels))


class Agent(nn.Module):
    """phi_A(s, a), phi_B(s) and phi_C(label), with phi_B shared by both contrastive pairs.

    Both pairs use the critic f(x, y) = -|x - y|^2 / temperature, i.e. ucl's "neg_sq_l2"
    with scale 2 / temperature.
    """

    def __init__(self, n_labels, embed_dim=8, temperature=0.5):
        super().__init__()
        self.state_action = mlp(4, embed_dim)
        self.state = mlp(2, embed_dim)
        self.label = LabelEncoder(n_labels, embed_dim)
        self.scale = 2.0 / temperature


def info_nce(critic_scale, anchor, positive, negatives):
    """Cross entropy of the positive among [positive, negatives] under f = -scale/2 |x - y|^2."""
    pos = -0.5 * critic_scale * (anchor - positive).pow(2).sum(-1, keepdim=True)
    neg = -0.5 * critic_scale * (anchor[:, None] - negatives).pow(2).sum(-1)
    logits = torch.cat([pos, neg], 1)
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))
