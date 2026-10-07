"""Zero-shot retrieval between unpaired modalities with pretrained contrastive models.

    python -m real_world.evaluate --experiment clip_clap_via_text
    python -m real_world.evaluate --experiment imagebind_via_image --bridge-sizes 16 64 256 all

For each trial, n_candidates clips are drawn from the test split; each A is scored
against the C of every candidate, and recall@k counts how often the clip's own C
ranks in the top k. The Monte Carlo (LogSumExp) estimator of Lemma 1 sums over
bridge samples from the bridge split (or the AudioSet ontology), with each
pretrained pair's own logit scale as the critic temperature.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from ucl import direct_scores, mc_scores, recall_at_k

from .data import SET_MODALITY, EmbeddingCache, read_manifest
from .encoders import pair_key

# a, c: (encoder, embedding set) of the two unpaired modalities. The bridge set is
# embedded by encoder `ab` (the A-B model) and by encoder `bc` (the B-C model).
EXPERIMENTS = {
    # Sec. 6.2.1 / Fig. 4 (left): image <-> audio through text with CLIP and CLAP.
    "clip_clap_via_text": dict(
        a=("clip", "image"), c=("clap", "audio"), bridge="ontology", ab="clip", bc="clap",
        baselines={"CLIP image . CLAP audio": (("clip", "image"), ("clap", "audio"))},
    ),
    # Same, with a large pool of AudioCaps captions from clips outside the manifest.
    "clip_clap_via_caption_pool": dict(
        a=("clip", "image"), c=("clap", "audio"), bridge="caption_pool", ab="clip", bc="clap",
        baselines={"CLIP image . CLAP audio": (("clip", "image"), ("clap", "audio"))},
    ),
    # Sec. 6.2.1 / Fig. 4 (right): LanguageBind, image <-> audio through text.
    "languagebind_via_text": dict(
        a=("languagebind", "image"), c=("languagebind", "audio"), bridge="ontology", ab="languagebind", bc="languagebind",
        baselines={"LanguageBind direct": (("languagebind", "image"), ("languagebind", "audio"))},
    ),
    # Fig. 5 (right) at larger M: LanguageBind with the external caption pool.
    "languagebind_via_caption_pool": dict(
        a=("languagebind", "image"), c=("languagebind", "audio"), bridge="caption_pool", ab="languagebind", bc="languagebind",
        baselines={"LanguageBind direct": (("languagebind", "image"), ("languagebind", "audio"))},
    ),
    # Appendix C.1 / Fig. 5 (left): ImageBind, audio <-> text through images.
    "imagebind_via_image": dict(
        a=("imagebind", "audio"), c=("imagebind", "caption"), bridge="image", ab="imagebind", bc="imagebind",
        baselines={"ImageBind direct": (("imagebind", "audio"), ("imagebind", "caption"))},
    ),
    # Appendix C.2 (2): audio <-> text through images with ImageBind (audio-image) and CLIP (image-text).
    "imagebind_clip_via_image": dict(
        a=("imagebind", "audio"), c=("clip", "caption"), bridge="image", ab="imagebind", bc="clip",
        baselines={
            "ImageBind direct": (("imagebind", "audio"), ("imagebind", "caption")),
            "CLAP (paired)": (("clap", "audio"), ("clap", "caption")),
        },
    ),
    # Appendix C.2 (1): image <-> text through audio with ImageBind (image-audio) and CLAP (audio-text).
    # Image and text have no unpaired direct comparison here: ImageBind's image and text encoders
    # are OpenCLIP ViT-H, trained on image-text pairs, so both baselines are paired models.
    "imagebind_clap_via_audio": dict(
        a=("imagebind", "image"), c=("clap", "caption"), bridge="audio", ab="imagebind", bc="clap",
        baselines={
            "OpenCLIP ViT-H (paired, in ImageBind)": (("imagebind", "image"), ("imagebind", "caption")),
            "CLIP ViT-B-32 (paired)": (("clip", "image"), ("clip", "caption")),
        },
    ),
}


class Experiment:
    def __init__(self, spec, cache, manifest):
        self.spec, self.cache = spec, cache
        (enc_a, set_a), (enc_c, set_c) = spec["a"], spec["c"]
        assert enc_a == spec["ab"] and enc_c == spec["bc"], "A must come from the A-B model and C from the B-C model"
        bridge_mod = SET_MODALITY[spec["bridge"]]
        self.scale_ab = cache.scales(spec["ab"])[pair_key(SET_MODALITY[set_a], bridge_mod)]
        self.scale_bc = cache.scales(spec["bc"])[pair_key(bridge_mod, SET_MODALITY[set_c])]
        self.split = {s: [r["id"] for r in manifest if r["split"] == s] for s in ("test", "val", "bridge")}
        if spec["bridge"] in ("ontology", "caption_pool"):  # text-only pools, disjoint from the manifest
            self.pool = cache.get(spec["ab"], spec["bridge"])[1]
        else:
            self.pool = self.split["bridge"]

    def mc_scores(self, ids, bridge_ids):
        """[n, n] Monte Carlo scores for candidate clips ids, summing over bridge_ids."""
        (enc_a, set_a), (enc_c, set_c) = self.spec["a"], self.spec["c"]
        phi_a, phi_c = self.cache.get(enc_a, set_a, ids), self.cache.get(enc_c, set_c, ids)
        b_ab = self.cache.get(self.spec["ab"], self.spec["bridge"], bridge_ids)
        b_bc = self.cache.get(self.spec["bc"], self.spec["bridge"], bridge_ids)
        return mc_scores(phi_a, phi_c, b_ab, b_bc, "dot", "dot", self.scale_ab, self.scale_bc)

    def baseline_scores(self, ids):
        out = {}
        for name, ((e1, s1), (e2, s2)) in self.spec["baselines"].items():
            x, y = self.cache.get(e1, s1, ids), self.cache.get(e2, s2, ids)
            if x.shape[1] == y.shape[1]:
                out[name] = direct_scores(x, y, "dot")
        return out


def run_trials(exp, split, n_trials, n_candidates, bridge_sizes, ks, seed):
    rng = np.random.default_rng(seed)
    ids_pool = exp.split[split]
    if len(ids_pool) < n_candidates:
        raise SystemExit(f"{split} split has {len(ids_pool)} clips, fewer than n_candidates={n_candidates}")
    results = {}
    for _ in range(n_trials):
        ids = list(rng.choice(ids_pool, n_candidates, replace=False))
        scores = exp.baseline_scores(ids)
        for m in bridge_sizes:
            m = len(exp.pool) if m == "all" else min(int(m), len(exp.pool))
            bridge_ids = list(rng.choice(exp.pool, m, replace=False))
            scores[f"Monte Carlo (M={m})"] = exp.mc_scores(ids, bridge_ids)
        for name, s in scores.items():
            for k in ks:
                results.setdefault(name, {}).setdefault(f"R@{k}", []).append(recall_at_k(s, k))
    return results


def summarize(values):
    v = np.asarray(values)
    ci = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else float("nan")
    return {"mean": float(v.mean()), "ci95": float(ci), "n": len(v)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="results/real_world")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--n-candidates", type=int, default=25)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--bridge-sizes", nargs="+", default=["all"], help="numbers of bridge samples M, or 'all'")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    exp = Experiment(EXPERIMENTS[args.experiment], EmbeddingCache(args.cache), read_manifest(args.manifest))
    results = run_trials(exp, "test", args.n_trials, args.n_candidates, args.bridge_sizes, args.ks, args.seed)
    summary = {name: {k: summarize(v) for k, v in r.items()} for name, r in results.items()}

    print(f"\n{args.experiment}: {args.n_trials} trials x {args.n_candidates} candidates   "
          f"scales ab={exp.scale_ab:.2f} bc={exp.scale_bc:.2f}")
    for name, r in summary.items():
        cells = "  ".join(f"{k} {s['mean']:.3f}±{s['ci95']:.3f}" for k, s in r.items())
        print(f"  {name:32s} {cells}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = dict(vars(args), scale_ab=exp.scale_ab, scale_bc=exp.scale_bc, summary=summary, raw=results)
    (out / f"{args.experiment}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
