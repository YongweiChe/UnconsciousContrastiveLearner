"""Estimators of log p(c | a) / p(c) between unpaired modalities A and C.

direct_scores: compare phi_A and phi_C directly (the "Law" of Sec. 4.3/4.4).
mc_scores:     Monte Carlo estimate of Lemma 1 over samples of the bridge B (Sec. 5),
               log (1/M) sum_m exp(f_AB(a, b_m) + f_BC(b_m, c)).
"""

import math

import torch

from .critics import pairwise


def direct_scores(phi_a, phi_c, critic, scale=1.0):
    return pairwise(phi_a, phi_c, critic, scale)


def mc_scores(
    phi_a,
    phi_c,
    bridge_ab,
    bridge_bc,
    critic_ab,
    critic_bc,
    scale_ab=1.0,
    scale_bc=1.0,
    max_block=2**24,
):
    """Returns the [N, K] matrix log (1/M) sum_m exp(f_AB(a_n, b_m) + f_BC(b_m, c_k)).

    phi_a:     [N, d_ab]  A embeddings from the A-B model
    phi_c:     [K, d_bc]  C embeddings from the B-C model
    bridge_ab: [M, d_ab]  bridge samples b_m embedded by the A-B model
    bridge_bc: [M, d_bc]  the same b_m embedded by the B-C model (row m <-> row m)

    When both pairs share one B encoder, pass the same tensor twice. The sum over M
    is a streamed logsumexp, so memory stays bounded by max_block elements.
    """
    if bridge_ab.shape[0] != bridge_bc.shape[0]:
        raise ValueError("bridge_ab and bridge_bc must embed the same M bridge samples")
    left = pairwise(phi_a, bridge_ab, critic_ab, scale_ab)  # [N, M]
    right = pairwise(bridge_bc, phi_c, critic_bc, scale_bc)  # [M, K]
    n, m = left.shape
    k = right.shape[1]
    block = max(1, max_block // max(1, n * k))
    out = torch.full((n, k), -math.inf, dtype=left.dtype, device=left.device)
    for j in range(0, m, block):
        chunk = left[:, j : j + block, None] + right[None, j : j + block, :]
        out = torch.logaddexp(out, torch.logsumexp(chunk, dim=1))
    return out - math.log(m)
