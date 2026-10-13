"""Risk policy: detections -> Safe / Manual Inspection / High Risk, plus multi-view late fusion.

The tiering lives in configs/policy.yaml, not in code, so the airport's security
authority can change it without touching the model. It is a DRAFT until validated
against real Algerian carry-on / hold-baggage rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SAFE, MANUAL, HIGH = "Safe", "Manual Inspection", "High Risk"
_RANK = {SAFE: 0, MANUAL: 1, HIGH: 2}


@dataclass
class Policy:
    high_risk_items: set[str]
    review_items: set[str]
    high_risk_conf: float = 0.6
    review_conf: float = 0.25
    # optional per-class overrides of the review threshold, e.g. {"Lighter": 0.15}
    class_review_conf: dict[str, float] = field(default_factory=dict)

    @property
    def prohibited(self) -> set[str]:
        return self.high_risk_items | self.review_items

    def review_threshold(self, cls: str) -> float:
        return self.class_review_conf.get(cls, self.review_conf)

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        with open(path, encoding="utf-8") as f:
            c = yaml.safe_load(f)
        return cls(
            high_risk_items=set(c["high_risk_items"]),
            review_items=set(c["review_items"]),
            high_risk_conf=float(c.get("high_risk_conf", 0.6)),
            review_conf=float(c.get("review_conf", 0.25)),
            class_review_conf={k: float(v) for k, v in (c.get("class_review_conf") or {}).items()},
        )


def score_detections(dets: list[tuple[str, float]], policy: Policy) -> tuple[str, list[str]]:
    """dets: [(class_name, confidence)]. Returns (action, reasons)."""
    action, reasons = SAFE, []
    for cls, conf in sorted(dets, key=lambda d: -d[1]):
        if cls in policy.high_risk_items and conf >= policy.high_risk_conf:
            a = HIGH
        elif cls in policy.prohibited and conf >= policy.review_threshold(cls):
            a = MANUAL
        else:
            continue
        reasons.append(f"{cls} ({conf:.2f})")
        if _RANK[a] > _RANK[action]:
            action = a
    return action, reasons


def fuse_views(dets_per_view: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    """Late fusion: per class keep the highest confidence seen in any view.
    Note: this raises Pd AND Pfa (any view can trigger). Validate on real dual-view data
    (e.g. DvXray, or Phase 1 scanner captures) before claiming a benefit."""
    best: dict[str, float] = {}
    for view in dets_per_view:
        for cls, conf in view:
            best[cls] = max(conf, best.get(cls, 0.0))
    return list(best.items())
