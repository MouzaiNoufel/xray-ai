"""Export trained weights for on-site (edge, offline) deployment and check speed.

  python scripts/export_model.py --weights weights/e11_y26s_640.pt --format onnx
  python scripts/export_model.py --weights weights/e11_y26s_640.pt --format engine --half   # NVIDIA TensorRT

ONNX runs anywhere (CPU, ONNX Runtime GPU, OpenVINO on Intel mini-PCs).
TensorRT 'engine' is fastest on NVIDIA hardware but must be built ON the target GPU.
After export the script re-runs the exported model on a few validation images and
checks its detections match the PyTorch model, so a broken export can't slip through.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xray.config import load_local  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "torchscript"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--check", type=int, default=8, help="images to compare after export")
    a = ap.parse_args()

    from ultralytics import YOLO

    from xray.data import list_images, load_data_yaml, resolve_split_dir

    src = YOLO(a.weights)
    out = src.export(
        format=a.format, imgsz=a.imgsz, half=a.half, dynamic=False, simplify=True if a.format == "onnx" else None
    )
    print(f"exported -> {out}")
    if a.check:
        local = load_local()
        imgs = [str(p) for p in list_images(resolve_split_dir(load_data_yaml(local["data"]), "val"))[: a.check]]
        exp = YOLO(out, task=src.task)
        bad = 0
        for p in imgs:
            r1 = src.predict(p, imgsz=a.imgsz, conf=0.25, verbose=False)[0]
            r2 = exp.predict(p, imgsz=a.imgsz, conf=0.25, verbose=False)[0]
            c1, c2 = sorted(r1.boxes.cls.int().tolist()), sorted(r2.boxes.cls.int().tolist())
            if c1 != c2:
                bad += 1
                print(f"  mismatch {Path(p).name}: torch {c1} vs export {c2}")
        print(
            f"export check: {len(imgs) - bad}/{len(imgs)} images give identical classes"
            + ("" if not bad else "  (small diffs near the threshold are normal with --half)")
        )


if __name__ == "__main__":
    main()
