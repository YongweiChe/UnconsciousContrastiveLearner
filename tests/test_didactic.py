import numpy as np
import pytest

from didactic.data import BridgeDistribution
from didactic.exact import ExactReferences


@pytest.mark.parametrize("bypass,corr_noise", [(0.0, 0.0), (0.5, 0.0), (0.0, 2.0)])
def test_gaussian_matches_samples(bypass, corr_noise):
    dist = BridgeDistribution.random(np.random.default_rng(0), dims=(6, 4, 5))
    dist.bypass, dist.corr_noise = bypass, corr_noise
    a, b, c = dist.sample(400_000, np.random.default_rng(1))
    g = dist.gaussian()
    emp = np.cov(np.concatenate([a, b, c], 1).T)
    da, db = 6, 4
    for block, (i, j) in {"cov_aa": ("a", "a"), "cov_ab": ("a", "b"), "cov_ac": ("a", "c"), "cov_cb": ("c", "b")}.items():
        sl = {"a": slice(0, da), "b": slice(da, da + db), "c": slice(da + db, None)}
        scale = np.abs(g[block]).max() + 1
        np.testing.assert_allclose(emp[sl[i], sl[j]], g[block], atol=0.03 * scale)
    np.testing.assert_allclose(a.mean(0), g["mean_a"], atol=0.05 * (np.abs(g["mean_a"]).max() + 1))


def test_full_bypass_disconnects_the_bridge():
    dist = BridgeDistribution.random(np.random.default_rng(0))
    dist.bypass = 1.0
    g = dist.gaussian()
    assert np.allclose(g["cov_ab"], 0) and np.allclose(g["cov_cb"], 0)
    assert not np.allclose(g["cov_ac"], 0)  # A and C remain dependent, through U


def test_exact_references():
    rng = np.random.default_rng(0)
    for bypass, lemma1_ok in [(0.0, lambda r: r > 0.5), (1.0, lambda r: abs(r - 1 / 32) < 1e-9)]:
        dist = BridgeDistribution.random(rng)
        dist.bypass = bypass
        a, b, c = dist.sample(320, rng)
        ref = ExactReferences(dist, a, c)
        assert ref.bayes() > 0.9  # the task stays solvable from (A, C)
        assert lemma1_ok(ref.add_bridge(dist.sample(2000, rng)[1]))
