"""Config-driven training with automatic resume, then the standard evaluation.

  python scripts/train.py configs/experiments/e11_y26s_640.yaml
  python scripts/train.py configs/experiments/e11_y26s_640.yaml --epochs 1 --fraction 0.02   # quick dry run

If the run was interrupted (power cut, Ctrl+C), just run the same command again:
it finds runs/<name>/weights/last.pt and resumes with optimizer state intact.
When training finishes, best.pt is copied to weights/<name>.pt and the evaluation in
xray/evaluate.py runs automatically (skip with --no-eval).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from xray.config import load_local  # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("extends", None)
    if base:
        parent = load_cfg(str((Path(path).parent / base).resolve()))
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(parent.get(k), dict):
                parent[k].update(v)
            else:
                parent[k] = v
        cfg = parent
    return cfg


def _resumable(ckpt_path: Path) -> bool:
    """True only for an interrupted run (checkpoint still holds optimizer + epoch)."""
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ck.get("optimizer") is not None and ck.get("epoch", -1) >= 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--fraction", type=float, help="train on a fraction of the train set (dry runs)")
    ap.add_argument("--batch", type=float)
    ap.add_argument("--device")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing last.pt and start over")
    a = ap.parse_args()

    from ultralytics import YOLO

    cfg = load_cfg(a.config)
    name = cfg["name"]
    tr = dict(cfg.get("train", {}))
    for k in ("epochs", "fraction", "device"):
        if getattr(a, k) is not None:
            tr[k] = getattr(a, k)
    if a.batch is not None:
        tr["batch"] = int(a.batch) if a.batch >= 1 else a.batch
    local = load_local()
    data = cfg.get("data") or local["data"]
    if cfg.get("use_backgrounds"):  # dataset variant with clean-bag backgrounds mixed into train
        data = local["data_with_backgrounds"]
    cfg.setdefault("eval", {}).setdefault("clean_dirs", local.get("clean_dirs"))
    project = str(ROOT / "runs")
    run_dir = Path(project) / name
    last = run_dir / "weights" / "last.pt"

    if last.exists() and not a.fresh:
        if _resumable(last):
            print(f"Resuming {name} from {last}")
            YOLO(str(last)).train(resume=True)
        else:  # Ultralytics strips optimizer state when a run completes
            print(f"{name} already finished training (use --fresh to retrain). Going to evaluation.")
    else:
        if a.fresh and run_dir.exists():
            shutil.rmtree(run_dir)
        model = YOLO(cfg["model"])
        if cfg.get("pretrained"):  # e.g. model: yolo26s-p2.yaml + pretrained: yolo26s.pt
            model.load(cfg["pretrained"])
        model.train(data=data, project=project, name=name, exist_ok=True, **tr)

    best = run_dir / "weights" / "best.pt"
    assert best.exists(), f"no best.pt in {run_dir}"
    (ROOT / "weights").mkdir(exist_ok=True)
    dst = ROOT / "weights" / f"{name}.pt"
    shutil.copy(best, dst)
    print(f"best weights -> {dst}")

    if not a.no_eval:
        from xray.evaluate import evaluate

        ev = cfg.get("eval", {})
        evaluate(
            weights=str(dst),
            data_yaml=data,
            out_dir=str(ROOT / "results" / name),
            policy_path=str(ROOT / cfg.get("policy", "configs/policy.yaml")),
            split=ev.get("split", "val"),
            clean_dirs=ev.get("clean_dirs"),
            imgsz=ev.get("imgsz", tr.get("imgsz", 640)),
            batch=ev.get("batch", 32),
            device=tr.get("device"),
        )


if __name__ == "__main__":  # required on Windows (dataloader workers spawn new processes)
    main()
