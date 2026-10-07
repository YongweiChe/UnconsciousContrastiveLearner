"""Exact references from the true Gaussian densities of the synthetic data."""

import numpy as np
import torch

from ucl import recall_at_k


def gaussian_log_ratio(mx, sxx, my, syy, sxy):
    """f(x, y) = log p(y | x) - log p(y) for jointly Gaussian (x, y), as an [N, M] matrix."""
    gain = sxy.T @ np.linalg.inv(sxx)  # E[y | x] = my + gain (x - mx)
    s_cond = syy - gain @ sxy
    p_cond, p_marg = np.linalg.inv(s_cond), np.linalg.inv(syy)
    const = -0.5 * (np.linalg.slogdet(s_cond)[1] - np.linalg.slogdet(syy)[1])
    t = lambda z: torch.tensor(z, dtype=torch.float64)
    p_cond, p_marg, gain, mx, my = map(t, (p_cond, p_marg, gain, mx, my))

    def f(x, y):
        dc = y[None] - (my + (x - mx) @ gain.T)[:, None]
        dm = y - my
        return const - 0.5 * torch.einsum("nmi,ij,nmj->nm", dc, p_cond, dc) + 0.5 * torch.einsum("mi,ij,mj->m", dm, p_marg, dm)[None]

    return f


class ExactReferences:
    """Bayes-optimal and exact-Lemma-1 recall@1 on groups of n_candidates validation triplets."""

    def __init__(self, dist, a, c, n_candidates=32):
        g = dist.gaussian()
        self.f_ab = gaussian_log_ratio(g["mean_a"], g["cov_aa"], g["mean_b"], g["cov_bb"], g["cov_ab"])
        self.f_cb = gaussian_log_ratio(g["mean_c"], g["cov_cc"], g["mean_b"], g["cov_bb"], g["cov_cb"])  # = log p(c|b)/p(c)
        self.f_ac = gaussian_log_ratio(g["mean_a"], g["cov_aa"], g["mean_c"], g["cov_cc"], g["cov_ac"])
        self.a, self.c = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(c, dtype=torch.float64)
        self.groups = [slice(i, i + n_candidates) for i in range(0, len(a) - n_candidates + 1, n_candidates)]
        self.acc = torch.full((len(self.groups), n_candidates, n_candidates), -np.inf, dtype=torch.float64)
        self.seen = 0

    def bayes(self):
        return float(np.mean([recall_at_k(self.f_ac(self.a[g], self.c[g]), 1) for g in self.groups]))

    def add_bridge(self, b, chunk=50_000):
        """Streams bridge samples into the exact Lemma 1 sums; returns recall@1 so far."""
        b = torch.as_tensor(b, dtype=torch.float64)
        for start in range(0, len(b), chunk):
            bb = b[start : start + chunk]
            for i, g in enumerate(self.groups):
                terms = self.f_ab(self.a[g], bb)[:, None, :] + self.f_cb(self.c[g], bb)[None]
                self.acc[i] = torch.logaddexp(self.acc[i], torch.logsumexp(terms, -1))
        self.seen += len(b)
        return float(np.mean([recall_at_k(self.acc[i], 1) for i in range(len(self.groups))]))
