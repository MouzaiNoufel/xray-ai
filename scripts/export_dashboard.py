"""Export real bags (image + detections + policy action) as JSON for the officer dashboard.
Local replacement for the Colab 'Export for dashboard' cell.

  python scripts/export_dashboard.py --weights weights/e11_y26s_640.pt --out dashboard_bags.json --n 12
  python scripts/export_dashboard.py --weights ... --images D:/some/folder   # your own images
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from xray.config import load_local  # noqa: E402
from xray.data import list_images, load_data_yaml, split_images
from xray.policy import Policy, score_detections

TIER = {"Safe": "safe", "Manual Inspection": "manual", "High Risk": "high"}
INORGANIC = {"Gun", "Knife", "Scissors", "HandCuffs", "Baton", "Hammer", "Pliers", "Wrench", "Bullet"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", help="folder; default = validation split")
    ap.add_argument("--out", default="dashboard_bags.json")
    ap.add_argument("--n", type=int, default=9, help="total bags, balanced across tiers when possible")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    from ultralytics import YOLO

    policy = Policy.load(ROOT / "configs" / "policy.yaml")
    if a.images:
        pool = list_images(a.images)
    else:
        local = load_local()
        pool = split_images(load_data_yaml(local["data"]), "val")
        for cd in local.get("clean_dirs") or []:  # include real clean bags so "Safe" examples are genuine
            pool += random.Random(a.seed).sample(list_images(cd), min(30, len(list_images(cd))))
    cand = random.Random(a.seed).sample(pool, min(120, len(pool)))
    model = YOLO(a.weights)
    per_tier = max(1, a.n // 3)
    seen = {"safe": 0, "manual": 0, "high": 0}
    bags = []
    for p, r in zip(cand, model.predict([str(c) for c in cand], imgsz=a.imgsz, conf=0.1, verbose=False, stream=True)):
        dets = [
            (model.names[int(c)], float(s), b)
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())
        ]
        action, _ = score_detections([(d[0], d[1]) for d in dets], policy)
        tier = TIER[action]
        if seen[tier] >= per_tier:
            continue
        seen[tier] += 1
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((640, 640))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
        items = [
            {
                "cls": c,
                "conf": round(s, 3),
                "x": round(b[0] * 100, 1),
                "y": round(b[1] * 100, 1),
                "w": round((b[2] - b[0]) * 100, 1),
                "h": round((b[3] - b[1]) * 100, 1),
                "mat": "inorganic" if c in INORGANIC else "organic",
            }
            for c, s, b in dets
            if s >= policy.review_threshold(c)
        ]
        bags.append(
            {
                "id": p.stem[-5:],
                "tier": tier,
                "items": items,
                "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
            }
        )
        if sum(seen.values()) >= a.n:
            break
    Path(a.out).write_text(json.dumps(bags), encoding="utf-8")
    print(f"exported {len(bags)} bags {seen} -> {a.out}")


if __name__ == "__main__":
    main()
