"""Tiny fake dataset (Roboflow-style layout, polygon labels) to check your install runs
end to end in ~2 minutes before committing the GPU to a 100-epoch run.
It is NOT a model test; the numbers it produces mean nothing.

  python scripts/make_smoke_dataset.py --out smoke_data
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

NAMES = [
    "Baton",
    "Bullet",
    "Gun",
    "Hammer",
    "HandCuffs",
    "Knife",
    "Lighter",
    "Pliers",
    "Powerbank",
    "Scissors",
    "Sprayer",
    "Wrench",
]


def bag(rng, size=320, n_obj=2):
    img = Image.new("RGB", (size, size), (235, 225, 200))
    d = ImageDraw.Draw(img)
    lines = []
    for _ in range(rng.randint(2, 5)):  # benign clutter (not labelled)
        x, y = rng.randint(0, size - 60), rng.randint(0, size - 60)
        d.rectangle([x, y, x + rng.randint(20, 60), y + rng.randint(20, 60)], fill=(230, 160 + rng.randint(0, 40), 90))
    for _ in range(n_obj):
        c = rng.randrange(len(NAMES))
        cx, cy, r = rng.randint(50, size - 50), rng.randint(50, size - 50), rng.randint(10 + 2 * c, 20 + 2 * c)
        k = 3 + c % 6
        pts = [(cx + r * math.cos(2 * math.pi * i / k), cy + r * math.sin(2 * math.pi * i / k)) for i in range(k)]
        d.polygon(pts, fill=(40 + 15 * c, 90, 200 - 10 * c))
        lines.append(f"{c} " + " ".join(f"{x / size:.5f} {y / size:.5f}" for x, y in pts))
    return img, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smoke_data")
    a = ap.parse_args()
    rng = random.Random(0)
    root = Path(a.out)
    for split, n in (("train", 96), ("valid", 24), ("test", 12)):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, lines = bag(rng)
            # mimic Roboflow names; make 2 val images share a source stem with train (leak check demo)
            sid = i if not (split == "valid" and i < 2) else i
            stem = (
                f"xray_{split[:2]}{sid:05d}_png.rf.{rng.getrandbits(64):016x}"
                if not (split == "valid" and i < 2)
                else f"xray_tr{sid:05d}_png.rf.{rng.getrandbits(64):016x}"
            )
            img.save(root / split / "images" / f"{stem}.jpg")
            (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    (root / "clean").mkdir(exist_ok=True)
    for i in range(40):
        bag(rng, n_obj=0)[0].save(root / "clean" / f"c{i:03d}.jpg")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    (root / "data.yaml").write_text(
        f"train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: {len(NAMES)}\nnames:\n{names}\n"
    )
    print(f"smoke dataset -> {root.resolve()}")


if __name__ == "__main__":
    main()
