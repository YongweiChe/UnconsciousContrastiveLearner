"""Synthetic A <- B -> C data (Sec. 6.1) and two ways of violating Assumption 1.

    B ~ N(mu, Sigma),  U ~ N(mu_u, Sigma_u) independent of B,
    A = sqrt(1 - rho) M_A B + sqrt(rho) N_A U + eps_A + kappa 1
    C = sqrt(1 - rho) M_C B + sqrt(rho) N_C U + eps_C + kappa 1

with eps ~ N(0, noise^2 I). With rho = kappa = 0, A is independent of C given B (Assumption 1).

bypass (rho): the fraction of the A-C signal carried by a hidden U that bypasses the
    bridge B, at roughly constant total signal. rho = 1 leaves B independent of A and C.
corr_noise: std of a scalar kappa added to every coordinate of A and C.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BridgeDistribution:
    mu: np.ndarray
    sigma: np.ndarray
    m_a: np.ndarray
    m_c: np.ndarray
    noise_a: float
    noise_c: float
    mu_u: np.ndarray
    sigma_u: np.ndarray
    n_a: np.ndarray
    n_c: np.ndarray
    corr_noise: float = 0.0
    bypass: float = 0.0

    @classmethod
    def random(cls, rng, dims=(32, 16, 32), noise=(2.0, 2.0)):
        a_dim, b_dim, c_dim = dims
        root = rng.uniform(-1, 1, (b_dim, b_dim))
        mu = rng.uniform(-1, 1, b_dim)
        m_a = rng.uniform(-1, 1, (a_dim, b_dim))
        m_c = rng.uniform(-1, 1, (c_dim, b_dim))
        root_u = rng.uniform(-1, 1, (b_dim, b_dim))  # U's parameters come after B's, which they leave unchanged
        return cls(
            mu=mu, sigma=root.T @ root, m_a=m_a, m_c=m_c, noise_a=noise[0], noise_c=noise[1],
            mu_u=rng.uniform(-1, 1, b_dim), sigma_u=root_u.T @ root_u,
            n_a=rng.uniform(-1, 1, (a_dim, b_dim)), n_c=rng.uniform(-1, 1, (c_dim, b_dim)),
        )

    @property
    def dims(self):
        return self.m_a.shape[0], self.mu.shape[0], self.m_c.shape[0]

    def sample(self, n, rng):
        """Returns n (A, B, C) triplets as float32 arrays."""
        b = rng.multivariate_normal(self.mu, self.sigma, n)
        kappa = rng.normal(0.0, 1.0, (n, 1)) * self.corr_noise
        sb, su = np.sqrt(1 - self.bypass), np.sqrt(self.bypass)
        a = sb * b @ self.m_a.T + self.noise_a * rng.standard_normal((n, self.m_a.shape[0])) + kappa
        c = sb * b @ self.m_c.T + self.noise_c * rng.standard_normal((n, self.m_c.shape[0])) + kappa
        if self.bypass:
            u = rng.multivariate_normal(self.mu_u, self.sigma_u, n)
            a += su * u @ self.n_a.T
            c += su * u @ self.n_c.T
        return a.astype(np.float32), b.astype(np.float32), c.astype(np.float32)

    def gaussian(self):
        """Means and (cross-)covariances of the jointly Gaussian (A, B, C)."""
        sb, su = np.sqrt(1 - self.bypass), np.sqrt(self.bypass)
        s, su_cov = self.sigma, self.sigma_u
        ones = lambda r, c: np.ones((r, c)) * self.corr_noise**2
        da, dc = self.m_a.shape[0], self.m_c.shape[0]
        return dict(
            mean_a=sb * self.m_a @ self.mu + su * self.n_a @ self.mu_u,
            mean_b=self.mu,
            mean_c=sb * self.m_c @ self.mu + su * self.n_c @ self.mu_u,
            cov_aa=sb**2 * self.m_a @ s @ self.m_a.T + su**2 * self.n_a @ su_cov @ self.n_a.T + self.noise_a**2 * np.eye(da) + ones(da, da),
            cov_cc=sb**2 * self.m_c @ s @ self.m_c.T + su**2 * self.n_c @ su_cov @ self.n_c.T + self.noise_c**2 * np.eye(dc) + ones(dc, dc),
            cov_bb=s,
            cov_ab=sb * self.m_a @ s,
            cov_cb=sb * self.m_c @ s,
            cov_ac=sb**2 * self.m_a @ s @ self.m_c.T + su**2 * self.n_a @ su_cov @ self.n_c.T + ones(da, dc),
        )
