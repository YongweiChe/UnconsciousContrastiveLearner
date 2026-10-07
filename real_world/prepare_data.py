"""Builds data/manifest.csv from downloaded AudioSet / AudioCaps clips.

Step 0: download AudioCaps clips (needs yt-dlp and ffmpeg on PATH; some videos are no longer available).
    python -m real_world.prepare_data download --captions audiocaps_test.csv --out raw_mp4s/ [--limit 1000]
    (AudioCaps CSVs: https://github.com/cdjkim/audiocaps/tree/master/dataset)

Step 1: extract a wav and the middle frame of each mp4.
    python -m real_world.prepare_data extract --videos raw_mp4s/ --out data/clips

Step 2: build the manifest with disjoint test / val / bridge splits.
    # one directory per clip: <id>/<id>.wav, <id>/<id>_frame.jpg, optional <id>/<id>_description.txt
    python -m real_world.prepare_data manifest --clips data/clips --out data/manifest.csv \
        [--captions audiocaps.csv --audiocaps] [--exclude Speech Music]

Optional: a large text-only bridge pool of AudioCaps captions from clips outside the manifest.
    python -m real_world.prepare_data caption-pool --captions audiocaps.csv --manifest data/manifest.csv

Without --captions, the caption is the contents of <id>_description.txt (the AudioSet labels).
"""

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .data import read_manifest, write_manifest


def download(args):
    """Fetches each AudioCaps clip (youtube_id, start_time + 10 s) as <youtube_id>_<start>-<end>.mp4."""
    if not shutil.which("yt-dlp") or not shutil.which("ffmpeg"):
        raise SystemExit("download needs yt-dlp and ffmpeg on PATH (pip install yt-dlp; brew/apt install ffmpeg)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.captions, newline="") as f:
        clips = sorted({(r["youtube_id"], int(float(r["start_time"]))) for r in csv.DictReader(f)})
    ok = 0
    for yid, start in clips[: args.limit]:
        path = out / f"{yid}_{start}-{start + 10}.mp4"
        if path.exists():
            ok += 1
            continue
        cmd = ["yt-dlp", "-q", "--no-warnings", "-f", "bv*[height<=360]+ba/b[height<=360]/b", "--merge-output-format", "mp4", "--download-sections", f"*{start}-{start + 10}",
               "--force-keyframes-at-cuts", "-o", str(path), f"https://www.youtube.com/watch?v={yid}"]
        done = subprocess.run(cmd, capture_output=True).returncode == 0 and path.exists() and path.stat().st_size > 10_000
        if not done:  # yt-dlp can exit 0 on a partial download; drop stubs so reruns retry them
            path.unlink(missing_ok=True)
        ok += done
    print(f"{ok} / {min(len(clips), args.limit or len(clips))} clips in {out}")


def extract(videos, out):
    import cv2
    from moviepy.editor import VideoFileClip

    for path in sorted(Path(videos).rglob("*.mp4")):
        clip_dir = Path(out) / path.stem
        clip_dir.mkdir(parents=True, exist_ok=True)
        try:
            with VideoFileClip(str(path)) as video:
                video.audio.write_audiofile(str(clip_dir / f"{path.stem}.wav"), logger=None)
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
            ok, frame = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(clip_dir / f"{path.stem}_frame.jpg"), frame)
        except Exception as e:  # corrupt downloads are common; skip them
            print(f"skipping {path}: {e}")


def build_manifest(args):
    captions = {}
    if args.captions:
        with open(args.captions, newline="") as f:
            for r in csv.DictReader(f):
                # AudioCaps clips are named <youtube_id>_<start>-<end>; key captions the same way.
                key = f"{r['youtube_id']}_{r['start_time']}" if args.audiocaps else r[args.id_col]
                captions.setdefault(key, r[args.caption_col])  # first caption if a clip has several

    rows = []
    for clip_dir in sorted(p for p in Path(args.clips).iterdir() if p.is_dir()):
        cid = clip_dir.name
        audio, image = clip_dir / f"{cid}.wav", clip_dir / f"{cid}_frame.jpg"
        desc = clip_dir / f"{cid}_description.txt"
        if not (audio.exists() and image.exists()) or audio.stat().st_size < args.min_audio_bytes:
            continue
        labels = desc.read_text().strip() if desc.exists() else ""
        caption = captions.get(cid.rsplit("-", 1)[0] if args.audiocaps else cid, labels)
        if not caption or any(word in labels or word in caption for word in args.exclude):
            continue
        rows.append(dict(id=cid, audio=str(audio), image=str(image), caption=caption, labels=labels))

    order = np.random.default_rng(args.seed).permutation(len(rows))
    n_test, n_val = round(args.test_frac * len(rows)), round(args.val_frac * len(rows))
    for rank, i in enumerate(order):
        rows[i]["split"] = "test" if rank < n_test else "val" if rank < n_test + n_val else "bridge"
    write_manifest(args.out, rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("test", "val", "bridge")}
    print(f"wrote {args.out}: {len(rows)} clips {counts}")


def caption_pool(args):
    """AudioCaps captions of clips not in the manifest: a text-only bridge pool with no test leakage."""
    used = {r["id"].rsplit("_", 1)[0] for r in read_manifest(args.manifest)}  # youtube ids
    with open(args.captions, newline="") as f:
        pool = sorted({r[args.caption_col].strip() for r in csv.DictReader(f) if r["youtube_id"] not in used})
    Path(args.out).write_text("\n".join(pool) + "\n")
    print(f"wrote {args.out}: {len(pool)} captions (excluding {len(used)} manifest videos)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--videos", required=True)
    e.add_argument("--out", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--clips", required=True)
    m.add_argument("--out", default="data/manifest.csv")
    m.add_argument("--captions")
    m.add_argument("--id-col", default="youtube_id")
    m.add_argument("--caption-col", default="caption")
    m.add_argument("--audiocaps", action="store_true", help="captions CSV is AudioCaps (youtube_id, start_time, caption)")
    m.add_argument("--exclude", nargs="*", default=[], help="drop clips whose labels/caption contain any of these")
    m.add_argument("--min-audio-bytes", type=int, default=1024)
    m.add_argument("--test-frac", type=float, default=0.4)
    m.add_argument("--val-frac", type=float, default=0.1)
    m.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("download")
    d.add_argument("--captions", required=True, help="AudioCaps CSV (youtube_id, start_time, caption)")
    d.add_argument("--out", required=True)
    d.add_argument("--limit", type=int, help="download at most this many clips")
    c = sub.add_parser("caption-pool")
    c.add_argument("--captions", required=True, help="AudioCaps CSV (youtube_id, caption)")
    c.add_argument("--manifest", default="data/manifest.csv")
    c.add_argument("--caption-col", default="caption")
    c.add_argument("--out", default="data/caption_pool.txt")
    args = p.parse_args()
    {"download": lambda: download(args), "extract": lambda: extract(args.videos, args.out), "manifest": lambda: build_manifest(args),
     "caption-pool": lambda: caption_pool(args)}[args.cmd]()


if __name__ == "__main__":
    main()
