"""Critic functions f(phi_1, phi_2) used throughout the paper.

Every critic is applied together with its (learned) logit scale. The scale is
irrelevant for direct retrieval (argmax is scale invariant) but it matters for
the Monte Carlo / LogSumExp estimator of Lemma 1, which needs the critic that
was actually trained.
"""

import torch
import torch.nn.functional as F

CRITICS = ("dot", "cosine", "neg_sq_l2")


def pairwise(x: torch.Tensor, y: torch.Tensor, critic: str, scale=1.0) -> torch.Tensor:
    """Returns the [N, M] matrix scale * f(x_i, y_j) for x: [N, d], y: [M, d].

    critic:
      "dot"       f = x^T y                   (Lemma 2 when x, y are unit norm)
      "cosine"    f = x^T y / (|x| |y|)
      "neg_sq_l2" f = -1/2 |x - y|^2          (Lemma 3)
    """
    if critic == "dot":
        s = x @ y.T
    elif critic == "cosine":
        s = F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T
    elif critic == "neg_sq_l2":
        sq = x.pow(2).sum(-1, keepdim=True) - 2 * x @ y.T + y.pow(2).sum(-1)
        s = -0.5 * sq.clamp_min(0)
    else:
        raise ValueError(f"unknown critic {critic!r}; expected one of {CRITICS}")
    return scale * s
