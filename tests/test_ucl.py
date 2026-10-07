import math

import numpy as np
import pytest
import torch
from scipy import special

from ucl import direct_scores, mc_scores, pairwise, recall_at_k


def brute_force_mc(a, c, b_ab, b_bc, critic, scale_ab, scale_bc):
    out = torch.empty(len(a), len(c), dtype=torch.float64)
    for i in range(len(a)):
        for k in range(len(c)):
            terms = [
                scale_ab * pairwise(a[i : i + 1], b_ab[m : m + 1], critic).item()
                + scale_bc * pairwise(b_bc[m : m + 1], c[k : k + 1], critic).item()
                for m in range(len(b_ab))
            ]
            out[i, k] = math.log(np.mean(np.exp(terms)))
    return out


@pytest.mark.parametrize("critic", ["dot", "cosine", "neg_sq_l2"])
def test_mc_matches_brute_force(critic):
    g = torch.Generator().manual_seed(0)
    a, c = torch.randn(4, 3, generator=g, dtype=torch.float64), torch.randn(5, 3, generator=g, dtype=torch.float64)
    b_ab, b_bc = torch.randn(7, 3, generator=g, dtype=torch.float64), torch.randn(7, 3, generator=g, dtype=torch.float64)
    expected = brute_force_mc(a, c, b_ab, b_bc, critic, 2.0, 0.5)
    got = mc_scores(a, c, b_ab, b_bc, critic, critic, 2.0, 0.5)
    torch.testing.assert_close(got, expected)
    # Streaming over tiny blocks gives the same answer.
    torch.testing.assert_close(mc_scores(a, c, b_ab, b_bc, critic, critic, 2.0, 0.5, max_block=1), expected)


def test_mc_is_overflow_safe():
    # Large logits (scale x unit vectors) must not overflow the streamed logsumexp.
    a = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    out = mc_scores(a, a, a, a, "dot", "dot", 500.0, 500.0)
    assert torch.isfinite(out).all()


def test_lemma2_uniform_hypersphere_closed_form():
    """With phi_B ~ Uniform(S^{d-1}) and the dot critic on unit vectors,
    E[exp(phi_B^T (a + c))] = Gamma(d/2) (2/k)^{d/2-1} I_{d/2-1}(k), with k = |a + c|."""
    d, m = 6, 400_000
    g = torch.Generator().manual_seed(1)
    unit = lambda n: torch.nn.functional.normalize(torch.randn(n, d, generator=g, dtype=torch.float64), dim=-1)
    a, c, bridge = unit(5), unit(5), unit(m)
    got = mc_scores(a, c, bridge, bridge, "dot", "dot")
    kappa = (a[:, None, :] + c[None, :, :]).norm(dim=-1).numpy()
    nu = d / 2 - 1
    expected = special.gammaln(d / 2) + nu * np.log(2 / kappa) + np.log(special.iv(nu, kappa))
    np.testing.assert_allclose(got.numpy(), expected, atol=0.02)
    # ...and is therefore a monotone function of a^T c, so it ranks like the direct comparison.
    direct = direct_scores(a, c, "dot")
    assert torch.equal(got.argmax(1), direct.argmax(1))


def test_lemma3_gaussian_closed_form():
    """With phi_B ~ N(0, cI) in k dims and f = -1/2 |x - y|^2,
    log E[exp(f(a, b) + f(b, c))] = -(k/2) log(2c + 1) - gamma (|a - c|^2 + delta a^T c),
    gamma = (c + 1) / (4c + 2), delta = 2 / (c + 1)."""
    k, var, m = 3, 1.5, 1_000_000
    g = torch.Generator().manual_seed(2)
    a = 0.7 * torch.randn(4, k, generator=g, dtype=torch.float64)
    c = 0.7 * torch.randn(4, k, generator=g, dtype=torch.float64)
    bridge = math.sqrt(var) * torch.randn(m, k, generator=g, dtype=torch.float64)
    got = mc_scores(a, c, bridge, bridge, "neg_sq_l2", "neg_sq_l2")
    gamma, delta = (var + 1) / (4 * var + 2), 2 / (var + 1)
    sq = torch.cdist(a, c).pow(2)
    expected = -(k / 2) * math.log(2 * var + 1) - gamma * (sq + delta * a @ c.T)
    torch.testing.assert_close(got, expected, atol=0.02, rtol=0)


def test_recall_at_k():
    s = torch.tensor([[3.0, 1.0, 2.0], [0.0, 1.0, 2.0], [5.0, 4.0, 3.0]])
    assert recall_at_k(s, 1) == pytest.approx(1 / 3)
    assert recall_at_k(s, 2) == pytest.approx(2 / 3)
    assert recall_at_k(s, 3) == 1.0
    # Ties are broken at random in expectation: a constant matrix scores chance.
    assert recall_at_k(torch.zeros(4, 4), 1) == pytest.approx(0.25)
    assert recall_at_k(torch.zeros(4, 4), 2) == pytest.approx(0.5)
    tied = torch.tensor([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 3.0]])  # row 0: tied with one other
    assert recall_at_k(tied, 1) == pytest.approx((0.5 + 1 + 1) / 3)
    with pytest.raises(ValueError):
        recall_at_k(torch.tensor([[math.inf, 0.0], [0.0, 1.0]]))

