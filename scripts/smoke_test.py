"""End-to-end smoke test: fake data -> dataset check -> negatives -> 2-epoch training -> evaluation.

  python scripts/smoke_test.py            (uses the GPU if there is one, about 1-3 min)
  python scripts/smoke_test.py --cpu

Everything goes to smoke_data/ and runs/smoke_*, with its own local config, so it never
touches configs/local.yaml or your real results. The numbers it prints mean nothing.
It only proves the whole pipeline runs on this machine.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args, env=None):
    print(f"\n$ python {' '.join(str(a) for a in args)}", flush=True)
    subprocess.run([PY, *map(str, args)], cwd=ROOT, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep smoke_data/ and runs/smoke_* afterwards")
    a = ap.parse_args()

    sd = ROOT / "smoke_data"
    for p in (sd, ROOT / "runs" / "smoke_det", ROOT / "results" / "smoke_det"):
        shutil.rmtree(p, ignore_errors=True)

    run("scripts/make_smoke_dataset.py", "--out", sd)
    run("scripts/check_dataset.py", "--data", sd / "data.yaml")
    run(
        "scripts/prepare_negatives.py",
        "--src",
        sd / "clean",
        "--out",
        sd / "neg",
        "--data",
        sd / "data_local.yaml",
        "--train",
        15,
        "--eval",
        25,
    )

    local = sd / "local.yaml"
    local.write_text(
        f"data: {(sd / 'data_local.yaml').as_posix()}\n"
        f"data_with_backgrounds: {(sd / 'data_with_bg.yaml').as_posix()}\n"
        f"clean_dirs:\n  - {(sd / 'neg' / 'clean_eval').as_posix()}\n",
        encoding="utf-8",
    )
    cfg = sd / "smoke_det.yaml"
    device = "cpu" if a.cpu else "0"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        sys.exit("PyTorch not installed")
    cfg.write_text(
        f"extends: {(ROOT / 'configs' / 'experiments' / '_base.yaml').as_posix()}\n"
        "name: smoke_det\nmodel: yolo26n.pt\nuse_backgrounds: true\n"
        f"train: {{imgsz: 320, epochs: 2, batch: 16, workers: 0, device: '{device}', plots: false}}\n"
        "eval: {batch: 16}\n",
        encoding="utf-8",
    )
    env = {**os.environ, "XRAY_LOCAL_CONFIG": str(local)}
    run("scripts/train.py", cfg, env=env)

    assert (ROOT / "results" / "smoke_det" / "metrics.json").exists(), "evaluation did not write metrics.json"
    print(f"\nSMOKE TEST PASSED on device={device}. The pipeline works on this machine.")
    if not a.keep:
        for p in (sd, ROOT / "runs" / "smoke_det", ROOT / "results" / "smoke_det", ROOT / "weights" / "smoke_det.pt"):
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
