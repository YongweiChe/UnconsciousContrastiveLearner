"""Embeds manifest clips (and the AudioSet ontology) once and caches the results.

    python -m real_world.embed --manifest data/manifest.csv --ontology data/ontology.json \
        --encoder clip --sets image caption ontology
    python -m real_world.embed ... --encoder clap --sets audio caption ontology
    python -m real_world.embed ... --encoder imagebind --sets image audio caption
    python -m real_world.embed ... --encoder languagebind --sets image audio caption ontology

Every embedding is stored unit-norm (float32, CPU); the model's trained logit scales
go to cache/<encoder>/scales.json.
"""

import argparse
import os

# Some ops (e.g. CLAP's bicubic upsampling) are missing on Apple MPS; run those on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import SET_MODALITY, ontology_names, read_lines, read_manifest, save_scales, save_set
from .encoders import ENCODERS


@torch.no_grad()
def embed_all(encoder, modality, items, batch_size):
    chunks = []
    for start in tqdm(range(0, len(items), batch_size), desc=modality, leave=False):
        out = encoder.embed(modality, items[start : start + batch_size])
        chunks.append(F.normalize(out.float(), dim=-1).cpu())
    return torch.cat(chunks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    p.add_argument("--sets", nargs="+", required=True, choices=sorted(SET_MODALITY))
    p.add_argument("--ontology", help="AudioSet ontology.json, needed for --sets ontology")
    p.add_argument("--caption-pool", help="one caption per line, needed for --sets caption_pool")
    p.add_argument("--out", default="cache")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--clap-fusion", action="store_true")
    p.add_argument("--clap-ckpt")
    args = p.parse_args()

    opts = {
        "clip": dict(model=args.clip_model, pretrained=args.clip_pretrained),
        "clap": dict(fusion=args.clap_fusion, ckpt=args.clap_ckpt),
    }.get(args.encoder, {})
    encoder = ENCODERS[args.encoder](args.device, **opts)
    rows = read_manifest(args.manifest)

    for set_name in args.sets:
        modality = SET_MODALITY[set_name]
        if modality not in encoder.modalities:
            raise SystemExit(f"{args.encoder} has no {modality} encoder (needed for set {set_name!r})")
        if set_name == "ontology":
            if not args.ontology:
                raise SystemExit("--sets ontology needs --ontology path/to/ontology.json")
            ids = items = ontology_names(args.ontology)
        elif set_name == "caption_pool":
            if not args.caption_pool:
                raise SystemExit("--sets caption_pool needs --caption-pool path (see prepare_data caption-pool)")
            ids = items = read_lines(args.caption_pool)
        else:
            column = {"image": "image", "audio": "audio", "caption": "caption"}[set_name]
            ids, items = [r["id"] for r in rows], [r[column] for r in rows]
        emb = embed_all(encoder, modality, items, args.batch_size)
        save_set(args.out, args.encoder, set_name, ids, emb)
        print(f"{args.encoder}/{set_name}: {tuple(emb.shape)}")

    save_scales(args.out, args.encoder, encoder.scales())
    print(f"{args.encoder} logit scales: {encoder.scales()}")


if __name__ == "__main__":
    main()
