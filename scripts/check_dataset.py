"""Audit the dataset BEFORE training anything. Run this first on your local copy.

  python scripts/check_dataset.py --data D:/datasets/pidray/data.yaml

It checks:
  1. Where each split really is, and writes data_local.yaml with absolute paths
     (fixes the Roboflow '../valid/images' path problem for good).
  2. Image counts, labels, background (empty) images, box vs polygon labels.
  3. Class balance and object-size distribution per split.
  4. Native image resolution. If images are ~448 px, training at 832 does nothing
     useful (Phase 0 saw exactly this); the fix is the original full-res PIDray.
  5. LEAKAGE: the same source X-ray appearing in train AND val/test. Roboflow names
     files like xray_01580_png.rf.<hash>.jpg; augmented copies keep the
     'xray_01580' stem. If val stems also appear in train, the val scores are inflated.
     Optional --phash adds a perceptual-hash near-duplicate check on a sample.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from PIL import Image

from xray.data import SIZE_BINS, label_path_for, list_images, load_data_yaml, read_gt, resolve_split_dirs, size_bin

STEM_RE = re.compile(r"^(.*?)(?:_(?:png|jpg|jpeg|bmp))?\.rf\.[0-9a-f]+$", re.I)


def source_stem(p: Path) -> str:
    m = STEM_RE.match(p.stem)
    return m.group(1) if m else p.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--phash",
        type=int,
        default=0,
        help="sample size per split for perceptual-hash duplicate check (needs imagehash)",
    )
    ap.add_argument("--size-sample", type=int, default=400)
    a = ap.parse_args()

    cfg = load_data_yaml(a.data)
    names = cfg["names"]
    print(f"classes ({len(names)}): {names}\n")
    splits: dict[str, list[Path]] = {}
    for s in ("train", "val", "test"):
        ds = resolve_split_dirs(cfg, s)
        if ds:
            splits[s] = ds
    for s, ds in splits.items():
        for d in ds:
            print(f"{s:<5} -> {d}")
    if "val" not in splits:
        sys.exit("No validation split found.")

    # write a clean, absolute data yaml (a split with several folders stays a list)
    def as_yaml(ds):
        return str(ds[0]) if len(ds) == 1 else [str(d) for d in ds]

    fixed = {"train": as_yaml(splits["train"]), "val": as_yaml(splits["val"])}
    if "test" in splits:
        fixed["test"] = as_yaml(splits["test"])
    fixed.update({"names": dict(enumerate(names)), "nc": len(names)})
    out_yaml = Path(a.data).with_name("data_local.yaml")
    if Path(a.data).name != "data_local.yaml":
        out_yaml.write_text(yaml.safe_dump(fixed, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"\nWrote {out_yaml}  <- use this in every config from now on\n")

    stems = {}
    for s, ds in splits.items():
        imgs = [p for d in ds for p in list_images(d)]
        stems[s] = Counter(source_stem(p) for p in imgs)
        n_bg = n_poly = n_box = 0
        cls_c, bin_c = Counter(), Counter()
        for p in imgs:
            if not label_path_for(p).exists():
                n_bg += 1
                continue
            gts = read_gt(p, names)
            if not gts:
                n_bg += 1
            for g in gts:
                cls_c[g["cls"]] += 1
                bin_c[size_bin(g["area"])] += 1
                n_poly += g["fmt"] == "poly"
                n_box += g["fmt"] == "box"
        sizes = Counter()
        for p in random.Random(0).sample(imgs, min(a.size_sample, len(imgs))):
            with Image.open(p) as im:
                sizes[im.size] += 1
        tot = max(1, sum(bin_c.values()))
        print(f"=== {s}: {len(imgs)} images, {len(stems[s])} unique source stems, {n_bg} background/empty")
        print(
            f"    labels: {n_poly} polygons, {n_box} boxes  -> {'segmentation training possible' if n_poly else 'boxes only'}"
        )
        print(f"    image sizes (sample): {sizes.most_common(4)}")
        print("    objects by size: " + ", ".join(f"{b} {bin_c[b]} ({bin_c[b] / tot:.0%})" for b, *_ in SIZE_BINS))
        print("    instances: " + ", ".join(f"{c} {cls_c[c]}" for c in names))
        dup = sum(v - 1 for v in stems[s].values() if v > 1)
        if dup:
            print(f"    note: {dup} images are extra copies of the same source stem (Roboflow augmentation)")
        print()

    print("=== Leakage check (same source X-ray in two splits)")
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        if a_ in stems and b_ in stems:
            inter = set(stems[a_]) & set(stems[b_])
            frac = len(inter) / max(1, len(stems[b_]))
            flag = "  <-- possible LEAK (same file name); confirm with --phash" if frac > 0.01 else "  ok"
            print(f"  {a_} ∩ {b_}: {len(inter)} shared stems ({frac:.1%} of {b_}){flag}")

    if a.phash:
        try:
            import imagehash
        except ImportError:
            sys.exit("pip install imagehash for --phash")
        print(f"\n=== Perceptual-hash near-duplicates (sample {a.phash} val vs all train hashes, distance <= 4)")
        tr = [p for d in splits["train"] for p in list_images(d)]
        print(f"  hashing {len(tr)} train images (one-off, a few minutes)...")
        import numpy as np

        def h64(p):
            with Image.open(p) as im:
                return int(str(imagehash.phash(im)), 16)

        th = np.array([h64(p) for p in tr], dtype=np.uint64)
        hits = 0
        vals = [p for d in splits["val"] for p in list_images(d)]
        for p in random.Random(1).sample(vals, min(a.phash, len(vals))):
            x = np.bitwise_xor(th, np.uint64(h64(p)))
            dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
            if (dist <= 4).any():
                hits += 1
        print(f"  {hits}/{a.phash} sampled val images have a near-duplicate in train")


if __name__ == "__main__":
    main()
