"""Convert a COCO-format dataset (JSON annotations + image folder) into YOLO format.

Original PIDray and CLCXray ship COCO JSON. Many others do too. One --split per
annotation file: --split <name> <annotations.json> <image folder>

  python scripts/convert_coco.py --out D:/datasets/yolo/pidray --mode seg `
      --split train D:/datasets/pidray/annotations/xray_train.json D:/datasets/pidray/train `
      --split val   D:/datasets/pidray/annotations/xray_test_hard.json D:/datasets/pidray/hard

  (PowerShell continues lines with a backtick `, Linux with \\.)

--mode seg  keeps polygon masks (needed for the -seg experiments). Instances with only a
            box get a 4-corner polygon, so one label file never mixes formats.
--mode box  writes plain boxes.

Images are hard-linked when possible (no extra disk space), otherwise copied.
Writes <out>/data.yaml, ready for check_dataset.py and merge_datasets.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def poly_area(xy: list[float]) -> float:
    xs, ys = xy[0::2], xy[1::2]
    return 0.5 * abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs))))


def convert_split(json_path: Path, img_dir: Path, out: Path, split: str, mode: str, cat_index: dict[int, int]):
    coco = json.loads(json_path.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in coco["images"]}
    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "labels").mkdir(parents=True, exist_ok=True)
    stats = Counter()
    for iid, im in imgs.items():
        src = img_dir / im["file_name"]
        if not src.exists():
            src = img_dir / Path(im["file_name"]).name
        if not src.exists():
            stats["missing_image"] += 1
            continue
        w, h = float(im["width"]), float(im["height"])
        dst = out / split / "images" / src.name
        link_or_copy(src, dst)
        lines = []
        for a in anns.get(iid, []):
            if a.get("iscrowd"):
                stats["crowd_skipped"] += 1
                continue
            c = cat_index[a["category_id"]]
            x, y, bw, bh = a["bbox"]
            seg = a.get("segmentation")
            if mode == "seg" and isinstance(seg, list) and seg and isinstance(seg[0], list):
                pts = max((s for s in seg if len(s) >= 6), key=poly_area, default=None)
            else:
                pts = None
            if mode == "seg":
                if pts is None:  # box only -> rectangle polygon
                    pts = [x, y, x + bw, y, x + bw, y + bh, x, y + bh]
                    stats["box_as_polygon"] += 1
                norm = [min(max(v / (w if i % 2 == 0 else h), 0.0), 1.0) for i, v in enumerate(pts)]
                lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in norm))
            else:
                lines.append(f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            stats["objects"] += 1
        (out / split / "labels" / f"{dst.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stats["images"] += 1
        stats["background_images"] += not lines
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", action="append", nargs=3, required=True, metavar=("NAME", "JSON", "IMAGES"))
    ap.add_argument("--mode", choices=["seg", "box"], default="seg")
    a = ap.parse_args()

    out = Path(a.out)
    specs = []
    for name, jp, idir in a.split:
        specs.append((name, Path(jp), Path(idir)))

    # one consistent class list across all splits (from the first file, sorted by category id)
    cats = sorted(json.loads(specs[0][1].read_text(encoding="utf-8"))["categories"], key=lambda c: c["id"])
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]
    print(f"classes: {names}")

    data = {"names": dict(enumerate(names)), "nc": len(names)}
    for name, jp, idir in specs:
        st = convert_split(jp, idir, out, name, a.mode, cat_index)
        print(f"{name}: {dict(st)}")
        data[name] = str((out / name / "images").resolve())
    if "val" not in data:
        first_non_train = next((n for n, *_ in specs if n != "train"), None)
        if first_non_train:
            data["val"] = data[first_non_train]
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
