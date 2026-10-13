"""Turn a pile of clean (threat-free) X-ray images into
   (a) background images for training, and
   (b) a SEPARATE held-out set for measuring Pfa.

  python scripts/prepare_negatives.py --src D:/datasets/SIXray/negatives --out D:/datasets/negatives \
      --data D:/datasets/pidray/data_local.yaml --train 6000 --eval 5000

The two sets never share an image (exact-duplicate files are removed first), otherwise
Pfa would be measured on bags the model was taught to ignore.

Rule of thumb: backgrounds around 5-10% of the train set (61k -> 3k-6k).
Where to get clean bags: SIXray's negative images (~1M, subway scanners; image-level
labels, so make sure you take ONLY the negative list), or best of all, real clean bags
from the partner airport in Phase 1. Different scanner = different look, so Pfa on
SIXray is a proxy, but a far better one than synthetic shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from xray.data import list_images, load_data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", nargs="+", required=True, help="folders of clean images, or .txt files listing image paths"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True, help="data_local.yaml from check_dataset.py")
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--eval", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths: list[Path] = []
    for s in a.src:
        sp = Path(s)
        if sp.suffix.lower() == ".txt":
            paths += [Path(line.strip()) for line in sp.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            paths += list_images(sp)
    print(f"{len(paths)} candidate clean images")

    seen, uniq = set(), []
    for p in paths:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append(p)
    print(f"{len(uniq)} after removing exact duplicates")
    random.Random(a.seed).shuffle(uniq)
    need = a.train + a.eval
    if len(uniq) < need:
        r = len(uniq) / need
        a.train, a.eval = int(a.train * r), len(uniq) - int(a.train * r)
        print(f"not enough images; using train={a.train}, eval={a.eval}")

    out = Path(a.out)
    tr_dir, ev_dir = out / "train_bg" / "images", out / "clean_eval"
    for d in (tr_dir, ev_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for i, p in enumerate(uniq[: a.train]):
        shutil.copy(p, tr_dir / f"bg_{i:06d}{p.suffix.lower()}")
    for i, p in enumerate(uniq[a.train : a.train + a.eval]):
        shutil.copy(p, ev_dir / f"clean_{i:06d}{p.suffix.lower()}")
    # no labels folder for train_bg -> Ultralytics treats every image there as background

    cfg = load_data_yaml(a.data)
    train = cfg["train"] if isinstance(cfg["train"], list) else [cfg["train"]]
    new = {
        "train": train + [str(tr_dir.resolve())],
        "val": cfg["val"],
        "names": dict(enumerate(cfg["names"])),
        "nc": len(cfg["names"]),
    }
    if cfg.get("test"):
        new["test"] = cfg["test"]
    out_yaml = Path(a.data).with_name("data_with_bg.yaml")
    out_yaml.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"train backgrounds: {a.train} -> {tr_dir}\nheld-out clean bags: {a.eval} -> {ev_dir}\nwrote {out_yaml}")
    print("Put these two paths into configs/local.yaml (data_with_backgrounds, clean_dirs).")


if __name__ == "__main__":
    main()
