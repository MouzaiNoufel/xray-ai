"""Evaluate any weights file with the standard protocol.

# your Phase 0 Colab weights, so every new model has the same baseline to beat
python scripts/evaluate.py --weights weights/phase0_best.pt --imgsz 448 --name e00_phase0_baseline

# with clean bags (Pfa)
python scripts/evaluate.py --weights weights/e11_y26s_640.pt --clean D:/datasets/negatives/clean_eval --name e11_y26s_640

# change policy.yaml thresholds and re-score instantly, without re-running the model
python scripts/evaluate.py --name e11_y26s_640 --from-cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xray.config import load_local  # noqa: E402
from xray.evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="results/<name>/")
    ap.add_argument("--weights")
    ap.add_argument("--data", default=None, help="defaults to configs/local.yaml 'data'")
    ap.add_argument("--split", default="val")
    ap.add_argument("--clean", nargs="*", default=None, help="folder(s) of clean-bag images for Pfa")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--policy", default=str(ROOT / "configs" / "policy.yaml"))
    ap.add_argument("--no-val", action="store_true", help="skip Ultralytics box metrics")
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--limit", type=int, help="only first N images (smoke tests)")
    a = ap.parse_args()

    out = ROOT / "results" / a.name
    data = a.data
    if data is None:
        local = load_local()
        data = local["data"]
        if a.clean is None:
            a.clean = local.get("clean_dirs")
    if a.from_cache:
        meta = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["meta"]
        a.weights, a.imgsz = meta["weights"], meta["imgsz"]
    evaluate(
        a.weights,
        data,
        str(out),
        a.policy,
        split=a.split,
        clean_dirs=a.clean,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        half=a.half,
        run_val=not a.no_val,
        from_cache=a.from_cache,
        limit=a.limit,
    )


if __name__ == "__main__":
    main()
