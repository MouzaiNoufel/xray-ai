from pathlib import Path

from xray.policy import HIGH, MANUAL, SAFE, Policy, fuse_views, score_detections

ROOT = Path(__file__).resolve().parents[1]


def pol(**kw):
    base = dict(
        high_risk_items={"Gun", "Knife"}, review_items={"Lighter", "Powerbank"}, high_risk_conf=0.6, review_conf=0.25
    )
    base.update(kw)
    return Policy(**base)


def test_empty_bag_is_safe():
    assert score_detections([], pol())[0] == SAFE


def test_high_risk_needs_high_confidence():
    assert score_detections([("Knife", 0.8)], pol())[0] == HIGH
    assert score_detections([("Knife", 0.4)], pol())[0] == MANUAL
    assert score_detections([("Knife", 0.1)], pol())[0] == SAFE


def test_highest_tier_wins_regardless_of_order():
    dets = [("Powerbank", 0.9), ("Gun", 0.7), ("Lighter", 0.3)]
    action, reasons = score_detections(dets, pol())
    assert action == HIGH
    assert len(reasons) == 3


def test_unknown_class_is_ignored():
    assert score_detections([("Umbrella", 0.99)], pol())[0] == SAFE


def test_per_class_review_threshold():
    p = pol(class_review_conf={"Lighter": 0.1})
    assert score_detections([("Lighter", 0.15)], p)[0] == MANUAL
    assert score_detections([("Powerbank", 0.15)], p)[0] == SAFE


def test_fusion_keeps_best_view():
    fused = dict(fuse_views([[("Knife", 0.2)], [("Knife", 0.78), ("Lighter", 0.3)]]))
    assert fused == {"Knife": 0.78, "Lighter": 0.3}
    assert score_detections(list(fused.items()), pol())[0] == HIGH


def test_project_policy_file_loads():
    p = Policy.load(ROOT / "configs" / "policy.yaml")
    assert "Gun" in p.high_risk_items
    assert not (p.high_risk_items & p.review_items), "a class cannot be in both tiers"
