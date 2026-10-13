"""Merge several YOLO-format datasets into one, on the unified class list in configs/taxonomy.yaml.

  python scripts/merge_datasets.py --out D:/datasets/merged `
      --source stcray D:/datasets/yolo/stcray/data.yaml `
      --source pidray D:/datasets/yolo/pidray/data.yaml

What it does:
  * remaps every label to the unified classes (unknown classes are dropped and reported)
  * prefixes file names with the source ("stcray__xxx.jpg") so nothing collides
  * hard-links images (no extra disk space on the same drive), copies otherwise
  * train = all sources' train splits; val = all sources' val splits
  * ALSO writes data_val_<source>.yaml per source, so you can see how the model does on
    each scanner separately. The gap between sources is your best early estimate of
    how much performance drops on an unseen scanner (the Phase 1 domain-gap problem).
  * --max-train-per-source N caps a large source so it doesn't drown a small one
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from xray.data import label_path_for, list_images, load_data_yaml, resolve_split_dir


def norm(s: str) -> str:
    return re.sub(r"[\s_\-]", "", s).lower()


def build_map(src_names: list[str], tax: dict) -> tuple[dict[int, int], list[str]]:
    classes = tax["classes"]
    lookup = {norm(c): i for i, c in enumerate(classes)}
    for alias, target in (tax.get("aliases") or {}).items():
        lookup[norm(str(alias))] = classes.index(target)
    mapping, dropped = {}, []
    for i, n in enumerate(src_names):
        j = lookup.get(norm(n))
        if j is None:
            dropped.append(n)
        else:
            mapping[i] = j
    return mapping, dropped


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", nargs=2, required=True, metavar=("NAME", "DATA_YAML"))
    ap.add_argument("--taxonomy", default=str(ROOT / "configs" / "taxonomy.yaml"))
    ap.add_argument("--max-train-per-source", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tax = yaml.safe_load(Path(a.taxonomy).read_text(encoding="utf-8"))
    classes = tax["classes"]
    out = Path(a.out)
    merged = {"train": [], "val": []}
    per_source_val = {}
    totals = Counter()

    for name, dy in a.source:
        cfg = load_data_yaml(dy)
        mapping, dropped = build_map(cfg["names"], tax)
        print(f"\n== {name}: {len(cfg['names'])} classes -> {len(set(mapping.values()))} unified")
        print("   mapped: " + ", ".join(f"{cfg['names'][i]}->{classes[j]}" for i, j in mapping.items()))
        if dropped:
            print(f"   DROPPED (not threats in taxonomy): {dropped}")
        used: set[Path] = set()
        for split in ("train", "val", "test"):
            sd = resolve_split_dir(cfg, split)
            if not sd or sd in used:  # e.g. val and test pointing at the same folder
                continue
            used.add(sd)
            imgs = list_images(sd)
            if split == "train" and a.max_train_per_source and len(imgs) > a.max_train_per_source:
                imgs = random.Random(a.seed).sample(imgs, a.max_train_per_source)
            od = out / split / name
            (od / "images").mkdir(parents=True, exist_ok=True)
            (od / "labels").mkdir(parents=True, exist_ok=True)
            st = Counter()
            for p in imgs:
                stem = f"{name}__{p.stem}"
                link_or_copy(p, od / "images" / f"{stem}{p.suffix.lower()}")
                lines = []
                lp = label_path_for(p)
                if lp.exists():
                    for line in lp.read_text(encoding="utf-8").splitlines():
                        parts = line.split()
                        if not parts:
                            continue
                        c = int(float(parts[0]))
                        if c in mapping:
                            lines.append(" ".join([str(mapping[c]), *parts[1:]]))
                            st["objects"] += 1
                        else:
                            st["objects_dropped"] += 1
                (od / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                st["images"] += 1
                st["background"] += not lines
            print(f"   {split:<5} {dict(st)}")
            totals.update({f"{split}_images": st["images"]})
            target = "train" if split == "train" else "val"
            if split == "val" or (split == "test" and name not in per_source_val):
                per_source_val[name] = str((od / "images").resolve())
            if target == "train" or split == "val":
                merged[target].append(str((od / "images").resolve()))

    base = {"names": dict(enumerate(classes)), "nc": len(classes)}
    (out / "data.yaml").write_text(
        yaml.safe_dump({"train": merged["train"], "val": merged["val"], **base}, sort_keys=False), encoding="utf-8"
    )
    for name, v in per_source_val.items():
        (out / f"data_val_{name}.yaml").write_text(
            yaml.safe_dump({"train": merged["train"], "val": v, **base}, sort_keys=False), encoding="utf-8"
        )
    print(f"\nmerged: {dict(totals)}\nwrote {out / 'data.yaml'} and data_val_<source>.yaml per source")
    print("Next: python scripts/check_dataset.py --data " + str(out / "data.yaml"))


if __name__ == "__main__":
    main()
