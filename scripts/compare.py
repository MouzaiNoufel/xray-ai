"""Leaderboard across every results/<name>/metrics.json -> printed + results/LEADERBOARD.md

python scripts/compare.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main():
    rows = []
    for mp in sorted((ROOT / "results").glob("*/metrics.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        s = m["sweep"]["0.25"]
        box = m["meta"].get("box", {})
        lat = m["meta"].get("latency", {})
        rows.append(
            [
                mp.parent.name,
                f(box.get("map50")),
                f(box.get("map50_95")),
                f(s["pd_any"]),
                f(s["pd_correct"]),
                f(s.get("pfa")),
                f(s["class_recall"].get("Lighter")),
                f(s["class_recall"].get("Bullet")),
                f(s["size_recall"].get("tiny")),
                f(s["size_recall"].get("small")),
                f(s["fp_per_class"].get("Powerbank", 0) / max(1, m["n_threat_bags"]), 3),
                f(lat.get("median_ms"), 1),
            ]
        )
    head = [
        "experiment",
        "mAP50",
        "mAP50-95",
        "Pd any",
        "Pd correct",
        "Pfa",
        "Lighter",
        "Bullet",
        "tiny rec",
        "small rec",
        "Powerbank FP/bag",
        "ms",
    ]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    txt = "All at threshold 0.25. Recall = IoU-matched, right class.\n\n" + "\n".join(md) + "\n"
    (ROOT / "results" / "LEADERBOARD.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
