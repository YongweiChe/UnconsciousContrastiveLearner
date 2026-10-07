"""End-to-end checks of the real-world pipeline on a synthetic embedding cache."""

import json
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from real_world import evaluate, prepare_data
from real_world.data import EmbeddingCache, read_manifest, save_scales, save_set, write_manifest


@pytest.fixture
def fake_world(tmp_path):
    """60 clips sharing a latent z; every (encoder, set) is a noisy random projection of z."""
    rng = np.random.default_rng(0)
    n, d_latent = 60, 6
    ids = [f"clip{i}" for i in range(n)]
    z = rng.standard_normal((n, d_latent))
    rows = [dict(id=i, audio="", image="", caption=f"cap {i}", labels="", split=s)
            for i, s in zip(ids, ["test"] * 30 + ["val"] * 5 + ["bridge"] * 25)]
    write_manifest(tmp_path / "manifest.csv", rows)
    cache = tmp_path / "cache"
    for encoder, dim in [("clip", 16), ("clap", 16), ("imagebind", 24), ("languagebind", 12)]:
        proj = rng.standard_normal((d_latent, dim))  # one shared space per encoder
        for set_name in ["image", "audio", "caption"]:
            emb = torch.tensor(z @ proj + 0.3 * rng.standard_normal((n, dim)), dtype=torch.float32)
            save_set(cache, encoder, set_name, ids, F.normalize(emb, dim=-1))
        names = [f"label{j}" for j in range(40)]
        save_set(cache, encoder, "ontology", names, F.normalize(torch.randn(40, dim), dim=-1))
        pool = torch.tensor(rng.standard_normal((50, d_latent)) @ proj, dtype=torch.float32)
        save_set(cache, encoder, "caption_pool", [f"pool caption {j}" for j in range(50)], F.normalize(pool, dim=-1))
        save_scales(cache, encoder, {"image|text": 4.0, "audio|image": 4.0, "audio|text": 4.0})
    return tmp_path


def run_cli(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.main()


@pytest.mark.parametrize("name", sorted(evaluate.EXPERIMENTS))
def test_every_experiment_runs(fake_world, name, monkeypatch):
    out = fake_world / "results"
    run_cli(evaluate, ["--experiment", name, "--manifest", str(fake_world / "manifest.csv"), "--cache", str(fake_world / "cache"),
                       "--out", str(out), "--n-trials", "5", "--n-candidates", "10", "--bridge-sizes", "4", "all"], monkeypatch)
    summary = json.loads((out / f"{name}.json").read_text())["summary"]
    assert any(k.startswith("Monte Carlo") for k in summary)
    for r in summary.values():
        assert 0.0 <= r["R@1"]["mean"] <= r["R@5"]["mean"] <= r["R@10"]["mean"] <= 1.0


def test_mc_bridges_disjoint_models(fake_world, monkeypatch):
    """ImageBind and CLIP live in unrelated spaces; Direct between them is meaningless,
    but the Monte Carlo sum over shared bridge images should still retrieve well."""
    spec = evaluate.EXPERIMENTS["imagebind_clip_via_image"]
    exp = evaluate.Experiment(spec, EmbeddingCache(fake_world / "cache"), read_manifest(fake_world / "manifest.csv"))
    res = evaluate.run_trials(exp, "test", 20, 10, ["all"], [1], seed=0)
    assert np.mean(res["Monte Carlo (M=25)"]["R@1"]) > 0.4  # chance is 0.1


def test_bridge_pool_excludes_test_clips(fake_world):
    exp = evaluate.Experiment(evaluate.EXPERIMENTS["imagebind_via_image"], EmbeddingCache(fake_world / "cache"),
                              read_manifest(fake_world / "manifest.csv"))
    assert not set(exp.pool) & set(exp.split["test"])
    assert not set(exp.pool) & set(exp.split["val"])


def test_manifest_splits_are_disjoint(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    for i in range(20):
        d = clips / f"vid{i}"
        d.mkdir(parents=True)
        (d / f"vid{i}.wav").write_bytes(b"0" * 2048)
        (d / f"vid{i}_frame.jpg").write_bytes(b"jpg")
        (d / f"vid{i}_description.txt").write_text("Speech" if i < 4 else "Dog, Bark")
    out = tmp_path / "manifest.csv"
    run_cli(prepare_data, ["manifest", "--clips", str(clips), "--out", str(out), "--exclude", "Speech"], monkeypatch)
    rows = read_manifest(out)
    assert len(rows) == 16
    splits = {s: {r["id"] for r in rows if r["split"] == s} for s in ("test", "val", "bridge")}
    assert sum(map(len, splits.values())) == 16 and len(splits["test"]) > 0 and len(splits["bridge"]) > 0
