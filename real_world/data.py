"""Manifest of (audio, image, caption) clips and the embedding cache.

manifest.csv columns: id, audio, image, caption, labels, split
split is one of
    test    pairs we retrieve between
    val     held out (not used by the evaluation scripts)
    bridge  the pool of intermediate-modality samples phi_B used by the Monte Carlo estimator

Keeping bridge disjoint from test matters: if the test clip's own image (or audio)
is in the bridge pool, the Monte Carlo estimate gets a privileged match.

Embedding sets (cache/<encoder>/<set>.pt):
    image, audio  one per clip, ids = clip ids
    caption       one text per clip, ids = clip ids
    ontology      AudioSet label names (Sec. 6.2.1), ids = the names
    caption_pool  captions of clips *outside* the manifest (a large sample of p(text)), ids = the captions
"""

import csv
import json
from pathlib import Path

import torch

SET_MODALITY = {"image": "image", "audio": "audio", "caption": "text", "ontology": "text", "caption_pool": "text"}


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "audio", "image", "caption", "labels", "split"])
        w.writeheader()
        w.writerows(rows)


def ontology_names(path):
    """Label names from the AudioSet ontology.json (github.com/audioset/ontology)."""
    return sorted({item["name"] for item in json.loads(Path(path).read_text())})


def save_set(cache_dir, encoder, set_name, ids, emb):
    out = Path(cache_dir) / encoder
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": list(ids), "emb": emb}, out / f"{set_name}.pt")


def save_scales(cache_dir, encoder, scales):
    path = Path(cache_dir) / encoder / "scales.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scales, indent=1))


class EmbeddingCache:
    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self._sets = {}

    def get(self, encoder, set_name, ids=None):
        """Unit-norm embeddings [len(ids), d] in the order of ids (all rows if ids is None)."""
        key = (encoder, set_name)
        if key not in self._sets:
            path = self.dir / encoder / f"{set_name}.pt"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run: python -m real_world.embed --encoder {encoder} --sets {set_name}")
            blob = torch.load(path, weights_only=True)
            self._sets[key] = (blob["emb"], {i: n for n, i in enumerate(blob["ids"])}, blob["ids"])
        emb, index, all_ids = self._sets[key]
        if ids is None:
            return emb, all_ids
        return emb[[index[i] for i in ids]]

    def scales(self, encoder):
        return json.loads((self.dir / encoder / "scales.json").read_text())


def read_lines(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
