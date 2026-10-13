"""Dataset helpers: resolve split folders, read YOLO labels (box OR polygon), list images.

PIDray-from-Roboflow labels are YOLO *segmentation* polygons (class + x1 y1 x2 y2 ...),
not 5-token boxes. Everything here accepts both formats.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_data_yaml(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names")
    if isinstance(names, dict):  # {0: 'Baton', 1: ...}
        names = [names[k] for k in sorted(names)]
    cfg["names"] = list(names)
    cfg["_yaml_path"] = str(Path(path).resolve())
    return cfg


def resolve_split_dir(data_cfg: dict, split: str, raw: str | None = None) -> Path | None:
    """Find a split's image folder. Roboflow exports often write '../valid/images'
    relative paths that don't match where data.yaml really sits, so we try a bounded
    set of candidates and accept the first one that exists and has images."""
    yaml_dir = Path(data_cfg["_yaml_path"]).parent
    root = Path(data_cfg["path"]) if data_cfg.get("path") else yaml_dir
    if not root.is_absolute():
        root = (yaml_dir / root).resolve()
    aliases = {"val": ["val", "valid"], "valid": ["valid", "val"], "test": ["test"], "train": ["train"]}
    cands: list[Path] = []
    if raw is None:
        raw = data_cfg.get(split) or data_cfg.get("val" if split == "valid" else split)
    if isinstance(raw, str):
        for base in (root, yaml_dir):
            cands.append((base / raw).resolve())
            cands.append((base / raw.replace("../", "").replace("./", "")).resolve())
    for name in aliases.get(split, [split]):
        for base in (root, yaml_dir):
            cands += [base / name / "images", base / "images" / name]
    for c in cands:
        if c.is_dir() and any(p.suffix.lower() in IMG_EXTS for p in c.iterdir()):
            return c
    return None


def resolve_split_dirs(data_cfg: dict, split: str) -> list[Path]:
    """Like resolve_split_dir, but a split may be a LIST of folders (merged datasets)."""
    raw = data_cfg.get(split) or data_cfg.get("val" if split == "valid" else split)
    if isinstance(raw, list):
        found = [resolve_split_dir(data_cfg, split, str(r)) for r in raw]
        missing = [r for r, f in zip(raw, found) if f is None]
        if missing:
            raise FileNotFoundError(f"split '{split}': folders not found or empty: {missing}")
        return found
    d = resolve_split_dir(data_cfg, split)
    return [d] if d else []


def split_images(data_cfg: dict, split: str) -> list[Path]:
    """All images of a split, across every folder it lists."""
    return [p for d in resolve_split_dirs(data_cfg, split) for p in list_images(d)]


def list_images(folder: str | os.PathLike) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path_for(img_path: Path) -> Path:
    """Ultralytics convention: .../images/xxx.jpg -> .../labels/xxx.txt"""
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_gt(img_path: Path, names: list[str]) -> list[dict]:
    """Return [{'cls': name, 'box': (x1,y1,x2,y2) normalized, 'area': frac, 'fmt': 'box'|'poly'}]."""
    lp = label_path_for(img_path)
    out: list[dict] = []
    if not lp.exists():
        return out
    with open(lp, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            cid, vals = int(float(p[0])), list(map(float, p[1:]))
            if len(vals) == 4:
                cx, cy, w, h = vals
                box, fmt = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), "box"
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                xs, ys = vals[0::2], vals[1::2]
                box, fmt = (min(xs), min(ys), max(xs), max(ys)), "poly"
            else:
                continue
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            out.append({"cls": names[cid], "box": box, "area": area, "fmt": fmt})
    return out


def iou(a, b) -> float:
    xa, ya, xb, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# Object-size bins as a fraction of image area. "small" (<1%) is the bin where Lighter
# and Bullet failed in Phase 0, so it gets reported on its own.
SIZE_BINS = [("tiny", 0.0, 0.0025), ("small", 0.0025, 0.01), ("medium", 0.01, 0.05), ("large", 0.05, 1.01)]


def size_bin(area: float) -> str:
    for name, lo, hi in SIZE_BINS:
        if lo <= area < hi:
            return name
    return "large"
