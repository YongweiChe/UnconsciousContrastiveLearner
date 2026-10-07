import math

import torch
import torch.nn.functional as F
from torch import nn

from ucl import pairwise

# Which encoders a model has, which pairs it is trained on, and which encoder embeds
# the bridge B on the A-B side and on the B-C side (None: no Monte Carlo estimate).
KINDS = {
    # phi_A <-> phi_B and phi_B <-> phi_C with a shared phi_B (Sec. 6.1.1).
    "unconscious": dict(encoders=("A", "B", "C"), pairs=(("A", "B"), ("B", "C")), bridge=("B", "B")),
    # Two independent models phi_A <-> phi_B1 and phi_B2 <-> phi_C (Sec. 6.2).
    "disparate": dict(encoders=("A", "B1", "B2", "C"), pairs=(("A", "B1"), ("B2", "C")), bridge=("B1", "B2")),
    # Oracle trained on (A, C) pairs.
    "ground_truth": dict(encoders=("A", "C"), pairs=(("A", "C"),), bridge=None),
}


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveModel(nn.Module):
    """A set of encoders trained with symmetric InfoNCE on one or more modality pairs.

    Each pair has its own learnable logit scale, initialized to 1 / temperature, so the
    two halves of a disparate model are fully independent contrastive models.
    """

    def __init__(self, kind, dims, embed_dim, critic, temperature=1.0, hidden=128):
        super().__init__()
        spec = KINDS[kind]
        input_dims = {"A": dims[0], "B": dims[1], "B1": dims[1], "B2": dims[1], "C": dims[2]}
        self.kind = kind
        self.critic = critic
        self.pairs = spec["pairs"]
        self.bridge = spec["bridge"]
        self.encoders = nn.ModuleDict({name: MLP(input_dims[name], embed_dim, hidden) for name in spec["encoders"]})
        self.log_scales = nn.ParameterDict(
            {f"{l}_{r}": nn.Parameter(torch.tensor(math.log(1 / temperature))) for l, r in self.pairs}
        )

    def embed(self, name, x):
        return self.encoders[name](x)

    def scale(self, left, right):
        return self.log_scales[f"{left}_{right}"].exp()

    def pair_loss(self, left, right, x_left, x_right, norm_penalty=0.0):
        z_l, z_r = self.embed(left, x_left), self.embed(right, x_right)
        logits = pairwise(z_l, z_r, self.critic, self.scale(left, right))
        labels = torch.arange(len(logits), device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        if norm_penalty:
            loss = loss + norm_penalty * (z_l.norm(dim=-1).mean() + z_r.norm(dim=-1).mean())
        return loss
