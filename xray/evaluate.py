"""One evaluation protocol for every experiment, so all comparisons are like for like.

Reports, from a SINGLE inference pass (cached to disk, so thresholds can be re-swept
without re-running the model):

  * box metrics (Ultralytics val: P, R, mAP50, mAP50-95)
  * bag-level Pd two ways:
      - "any alarm":     any prohibited detection >= t   (what Phase 0 reported)
      - "correct alarm": at least one real threat in the bag found with the right class
                         and location (IoU >= 0.3). This one is the honest number:
                         "any alarm" also counts a bag as caught when the model fires on
                         the wrong thing (e.g. a phantom Powerbank) and misses the knife.
  * per-class object recall and per-size-bin recall (tiny/small/medium/large)
  * false positives per class on threat images
  * Pfa on clean bags (if a clean-bag folder is given) with a 95% Wilson interval
  * the policy's Safe / Manual / High Risk distribution on threat bags and clean bags
  * latency at batch 1 (median and p95, end to end incl. pre/post-processing)
"""

from __future__ import annotations

import gzip
import json
import math
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

from .data import SIZE_BINS, iou, list_images, load_data_yaml, read_gt, resolve_split_dir, size_bin
from .policy import SAFE, Policy, score_detections

IOU_MATCH = 0.3
SWEEP = [round(0.05 * i, 2) for i in range(1, 19)]  # 0.05 .. 0.90
REPORT_T = [0.10, 0.25, 0.40, 0.60, 0.75]


# ----------------------------------------------------------------------------- inference
def _hk(half: bool) -> dict:
    """Only pass FP16 when asked (newer Ultralytics warns on half=False)."""
    return {"half": True} if half else {}


def _predict(model, paths: list[Path], imgsz: int, conf_floor: float, batch: int, device, half: bool):
    names = model.names
    out = []
    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i : i + batch]]
        for r in model.predict(
            chunk, imgsz=imgsz, conf=conf_floor, verbose=False, device=device, stream=True, **_hk(half)
        ):
            dets = []
            if r.boxes is not None and len(r.boxes):
                for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist()):
                    dets.append([names[int(c)], round(float(s), 4), [round(v, 5) for v in b]])
            out.append(dets)
        done = min(i + batch, len(paths))
        if (i // batch) % 50 == 0 or done == len(paths):
            print(f"    {done}/{len(paths)}", flush=True)
    return out


def _latency(model, paths: list[Path], imgsz: int, device, half: bool, n: int = 60) -> dict:
    import torch

    paths = paths[: n + 10]
    if len(paths) < 12:
        return {}
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for p in paths[:10]:
        model.predict(str(p), imgsz=imgsz, verbose=False, device=device, **_hk(half))
    times = []
    for p in paths[10:]:
        sync()
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=imgsz, verbose=False, device=device, **_hk(half))
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"median_ms": round(statistics.median(times), 2), "p95_ms": round(times[int(0.95 * (len(times) - 1))], 2)}


# ----------------------------------------------------------------------------- metrics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _match(gts: list[dict], preds: list, t: float):
    """Greedy class-aware matching. Returns (matched_gt_idx set, unmatched preds list)."""
    ps = sorted([p for p in preds if p[1] >= t], key=lambda p: -p[1])
    matched, fps = set(), []
    for cls, conf, box in ps:
        best, bi = 0.0, None
        for gi, g in enumerate(gts):
            if gi in matched or g["cls"] != cls:
                continue
            v = iou(g["box"], box)
            if v > best:
                best, bi = v, gi
        if bi is not None and best >= IOU_MATCH:
            matched.add(bi)
        else:
            fps.append((cls, conf))
    return matched, fps


def compute_metrics(threat: list[dict], clean: list[dict], policy: Policy) -> dict:
    prohibited = policy.prohibited
    m: dict = {"n_threat_images": len(threat), "n_clean_images": len(clean), "sweep": {}}
    threat_bags = [r for r in threat if any(g["cls"] in prohibited for g in r["gt"])]
    m["n_threat_bags"] = len(threat_bags)
    m["n_images_without_threat_labels"] = len(threat) - len(threat_bags)

    for t in SWEEP:
        any_hit = correct_hit = 0
        cls_tot, cls_hit = Counter(), Counter()
        bin_tot, bin_hit = Counter(), Counter()
        fp_cls = Counter()
        for r in threat_bags:
            gts = [g for g in r["gt"] if g["cls"] in prohibited]
            matched, fps = _match(gts, r["pred"], t)
            if any(p[0] in prohibited and p[1] >= t for p in r["pred"]):
                any_hit += 1
            if matched:
                correct_hit += 1
            for gi, g in enumerate(gts):
                cls_tot[g["cls"]] += 1
                bin_tot[size_bin(g["area"])] += 1
                if gi in matched:
                    cls_hit[g["cls"]] += 1
                    bin_hit[size_bin(g["area"])] += 1
            for c, _ in fps:
                fp_cls[c] += 1
        nb = max(1, len(threat_bags))
        row = {
            "pd_any": any_hit / nb,
            "pd_correct": correct_hit / nb,
            "object_recall": sum(cls_hit.values()) / max(1, sum(cls_tot.values())),
            "class_recall": {c: cls_hit[c] / cls_tot[c] for c in sorted(cls_tot)},
            "class_count": dict(cls_tot),
            "size_recall": {b: (bin_hit[b] / bin_tot[b] if bin_tot[b] else None) for b, *_ in SIZE_BINS},
            "size_count": {b: bin_tot[b] for b, *_ in SIZE_BINS},
            "fp_per_class": dict(fp_cls.most_common()),
            "fp_per_image": sum(fp_cls.values()) / nb,
        }
        if clean:
            alarms = [r for r in clean if any(p[0] in prohibited and p[1] >= t for p in r["pred"])]
            k, n = len(alarms), len(clean)
            fired = Counter(p[0] for r in alarms for p in r["pred"] if p[0] in prohibited and p[1] >= t)
            row.update({"pfa": k / n, "pfa_ci95": wilson(k, n), "pfa_by_class": dict(fired.most_common())})
        m["sweep"][f"{t:.2f}"] = row

    # policy outcome (uses policy.yaml thresholds, independent of the sweep)
    def dist(recs):
        c = Counter(score_detections([(p[0], p[1]) for p in r["pred"]], policy)[0] for r in recs)
        n = max(1, len(recs))
        return {k: c[k] / n for k in ("Safe", "Manual Inspection", "High Risk")}

    m["policy"] = {"threat_bags": dist(threat_bags)}
    m["policy"]["pd"] = 1 - m["policy"]["threat_bags"][SAFE]
    if clean:
        m["policy"]["clean_bags"] = dist(clean)
        m["policy"]["pfa"] = 1 - m["policy"]["clean_bags"][SAFE]
        k = round(m["policy"]["pfa"] * len(clean))
        m["policy"]["pfa_ci95"] = wilson(k, len(clean))
        # operating points: best Pd_correct subject to a Pfa budget
        ops = {}
        for budget in (0.02, 0.05, 0.10, 0.20):
            ok = [(t, r) for t, r in m["sweep"].items() if r["pfa"] <= budget]
            if ok:
                t, r = max(ok, key=lambda x: x[1]["pd_correct"])
                ops[f"pfa<={budget:.2f}"] = {
                    "threshold": float(t),
                    "pd_correct": r["pd_correct"],
                    "pd_any": r["pd_any"],
                    "pfa": r["pfa"],
                }
        m["operating_points"] = ops
    return m


# ----------------------------------------------------------------------------- driver
def evaluate(
    weights: str,
    data_yaml: str,
    out_dir: str,
    policy_path: str,
    split: str = "val",
    clean_dirs: list[str] | None = None,
    imgsz: int = 640,
    conf_floor: float = 0.05,
    batch: int = 32,
    device=None,
    half: bool = False,
    run_val: bool = True,
    from_cache: bool = False,
    limit: int | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "predictions.json.gz"
    policy = Policy.load(policy_path)
    data = load_data_yaml(data_yaml)
    names = data["names"]
    meta: dict = {
        "weights": str(weights),
        "data": data["_yaml_path"],
        "split": split,
        "imgsz": imgsz,
        "conf_floor": conf_floor,
        "clean_dirs": clean_dirs or [],
        "host": platform.node(),
    }

    if from_cache and cache.exists():
        print(f"Re-scoring from cache {cache} (no inference).")
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            blob = json.load(f)
        threat, clean, meta_old = blob["threat"], blob["clean"], blob["meta"]
        meta.update({k: v for k, v in meta_old.items() if k in ("box", "latency", "device")})
    else:
        import torch
        from ultralytics import YOLO

        model = YOLO(weights)
        split_dir = resolve_split_dir(data, split)
        assert split_dir, f"Could not find images for split '{split}' from {data_yaml}"
        imgs = list_images(split_dir)
        if limit:
            imgs = imgs[:limit]
        print(f"[1/4] threat set: {len(imgs)} images from {split_dir}")
        preds = _predict(model, imgs, imgsz, conf_floor, batch, device, half)
        threat = []
        for p, d in zip(imgs, preds):
            gts = read_gt(p, names)
            threat.append(
                {"img": p.name, "gt": [{"cls": g["cls"], "box": g["box"], "area": g["area"]} for g in gts], "pred": d}
            )

        clean = []
        for cd in clean_dirs or []:
            cimgs = list_images(cd)
            if limit:
                cimgs = cimgs[:limit]
            print(f"[2/4] clean set: {len(cimgs)} images from {cd}")
            for p, d in zip(cimgs, _predict(model, cimgs, imgsz, conf_floor, batch, device, half)):
                clean.append({"img": p.name, "pred": d})

        meta["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        print("[3/4] latency")
        meta["latency"] = _latency(model, imgs, imgsz, device, half)
        if run_val:
            print("[4/4] box metrics (ultralytics val)")
            v = model.val(
                data=data_yaml,
                split="val" if split in ("val", "valid") else split,
                imgsz=imgsz,
                batch=batch,
                device=device,
                verbose=False,
                plots=False,
                project=str(out),
                name="ultralytics_val",
                exist_ok=True,
                **_hk(half),
            )
            meta["box"] = {
                "precision": float(v.box.mp),
                "recall": float(v.box.mr),
                "map50": float(v.box.map50),
                "map50_95": float(v.box.map),
            }
        with gzip.open(cache, "wt", encoding="utf-8") as f:
            json.dump({"meta": meta, "threat": threat, "clean": clean}, f)

    metrics = compute_metrics(threat, clean, policy)
    metrics["meta"] = meta
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_report(metrics, policy), encoding="utf-8")
    _plot(metrics, out)
    print((out / "report.md").read_text(encoding="utf-8"))
    return metrics


# ----------------------------------------------------------------------------- report
def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render_report(m: dict, policy: Policy) -> str:
    meta = m["meta"]
    L = [
        f"# Evaluation: {Path(meta['weights']).name}",
        "",
        f"- data: `{meta['data']}` split `{meta['split']}`, imgsz {meta['imgsz']}, device {meta.get('device', '?')}",
        f"- threat bags: {m['n_threat_bags']} (images without threat labels: {m['n_images_without_threat_labels']}); clean bags: {m['n_clean_images']}",
        "",
    ]
    if "box" in meta:
        b = meta["box"]
        L += [
            "## Box metrics",
            "",
            "| P | R | mAP50 | mAP50-95 |",
            "|---|---|---|---|",
            f"| {_f(b['precision'])} | {_f(b['recall'])} | {_f(b['map50'])} | {_f(b['map50_95'])} |",
            "",
        ]
    if meta.get("latency"):
        L += [
            f"Latency (batch 1, end to end): median {meta['latency']['median_ms']} ms, p95 {meta['latency']['p95_ms']} ms",
            "",
        ]
    has_clean = m["n_clean_images"] > 0
    L += [
        "## Bag-level detection vs threshold",
        "",
        "| t | Pd any-alarm | Pd correct-alarm | object recall | FP / threat bag |"
        + (" Pfa (95% CI) |" if has_clean else ""),
        "|---|---|---|---|---|" + ("---|" if has_clean else ""),
    ]
    for t in REPORT_T:
        r = m["sweep"].get(f"{t:.2f}")
        if not r:
            continue
        row = f"| {t:.2f} | {_f(r['pd_any'])} | {_f(r['pd_correct'])} | {_f(r['object_recall'])} | {_f(r['fp_per_image'], 2)} |"
        if has_clean:
            lo, hi = r["pfa_ci95"]
            row += f" {_f(r['pfa'])} ({_f(lo)}-{_f(hi)}) |"
        L.append(row)
    r = m["sweep"]["0.25"]
    L += ["", "## Per-class recall at t=0.25 (IoU-matched, right class)", "", "| class | recall | n |", "|---|---|---|"]
    for c, v in sorted(r["class_recall"].items(), key=lambda x: x[1]):
        L.append(f"| {c} | {_f(v)} | {r['class_count'][c]} |")
    L += [
        "",
        "## Recall by object size at t=0.25 (fraction of image area)",
        "",
        "| bin | range | recall | n |",
        "|---|---|---|---|",
    ]
    for name, lo, hi in SIZE_BINS:
        L.append(
            f"| {name} | {lo * 100:.2g}%-{min(hi, 1) * 100:.3g}% | {_f(r['size_recall'][name])} | {r['size_count'][name]} |"
        )
    L += [
        "",
        "## False positives on threat bags at t=0.25",
        "",
        ", ".join(f"{k}: {v}" for k, v in r["fp_per_class"].items()) or "none",
        "",
    ]
    p = m["policy"]
    L += [
        "## Policy outcome (configs/policy.yaml)",
        "",
        f"review_conf={policy.review_conf}, high_risk_conf={policy.high_risk_conf}, overrides={policy.class_review_conf or 'none'}",
        "",
        "| set | Safe | Manual | High Risk |",
        "|---|---|---|---|",
        "| threat bags | "
        + " | ".join(_f(p["threat_bags"][k]) for k in ("Safe", "Manual Inspection", "High Risk"))
        + " |",
    ]
    if has_clean:
        L.append(
            "| clean bags | "
            + " | ".join(_f(p["clean_bags"][k]) for k in ("Safe", "Manual Inspection", "High Risk"))
            + " |"
        )
        lo, hi = p["pfa_ci95"]
        L += [
            "",
            f"Policy Pd = {_f(p['pd'])}, policy Pfa = {_f(p['pfa'])} (95% CI {_f(lo)}-{_f(hi)})",
            "",
            "## Operating points (best correct-alarm Pd within a Pfa budget)",
            "",
            "| Pfa budget | threshold | Pd correct | Pd any | Pfa |",
            "|---|---|---|---|---|",
        ]
        for k, v in m.get("operating_points", {}).items():
            L.append(f"| {k} | {v['threshold']:.2f} | {_f(v['pd_correct'])} | {_f(v['pd_any'])} | {_f(v['pfa'])} |")
        fired = m["sweep"]["0.25"].get("pfa_by_class", {})
        L += [
            "",
            "False alarms on clean bags at t=0.25 by class: "
            + (", ".join(f"{k}: {v}" for k, v in fired.items()) or "none"),
        ]
        L += [
            "",
            "> Pfa here is measured on a public clean-bag set from a different scanner than the training data. "
            "It is a proxy. The real Pfa comes from Phase 1 data on the airport's own machines.",
        ]
    else:
        L += ["", f"Policy Pd = {_f(p['pd'])}. **No clean bags given, so Pfa is unmeasured.** Pass `--clean <folder>`."]
    return "\n".join(L) + "\n"


def _plot(m: dict, out: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ts = [float(t) for t in m["sweep"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ts, [m["sweep"][f"{t:.2f}"]["pd_any"] for t in ts], label="Pd (any alarm)")
    ax.plot(ts, [m["sweep"][f"{t:.2f}"]["pd_correct"] for t in ts], label="Pd (correct alarm)")
    if m["n_clean_images"]:
        ax.plot(ts, [m["sweep"][f"{t:.2f}"]["pfa"] for t in ts], label="Pfa (clean bags)")
    ax.set_xlabel("confidence threshold")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(Path(m["meta"]["weights"]).name)
    fig.tight_layout()
    fig.savefig(out / "pd_pfa_vs_threshold.png", dpi=120)
    plt.close(fig)
