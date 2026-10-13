"""compute_metrics on hand-made predictions, so the Pd / Pfa arithmetic is pinned down."""

import pytest

from xray.evaluate import compute_metrics, wilson
from xray.policy import Policy

POL = Policy(high_risk_items={"Knife"}, review_items={"Powerbank"}, high_risk_conf=0.6, review_conf=0.25)
KNIFE = {"cls": "Knife", "box": (0.1, 0.1, 0.3, 0.3), "area": 0.04}


def bag(gt, pred):
    return {"img": "x", "gt": gt, "pred": pred}


THREAT = [
    bag([KNIFE], [["Knife", 0.9, [0.1, 0.1, 0.3, 0.3]]]),  # caught correctly
    bag([KNIFE], [["Powerbank", 0.8, [0.6, 0.6, 0.8, 0.8]]]),  # alarm, but on the wrong thing
    bag([KNIFE], []),  # missed
    bag([KNIFE], [["Knife", 0.3, [0.1, 0.1, 0.3, 0.3]]]),  # caught only at low threshold
]
CLEAN = [{"img": f"c{i}", "pred": []} for i in range(8)] + [
    {"img": "c8", "pred": [["Powerbank", 0.5, [0, 0, 0.1, 0.1]]]},
    {"img": "c9", "pred": [["Knife", 0.2, [0, 0, 0.1, 0.1]]]},
]


def test_any_alarm_vs_correct_alarm():
    m = compute_metrics(THREAT, CLEAN, POL)
    s = m["sweep"]["0.25"]
    assert s["pd_any"] == pytest.approx(3 / 4)  # wrong-class alarm counts here...
    assert s["pd_correct"] == pytest.approx(2 / 4)  # ...but not here
    assert s["fp_per_class"] == {"Powerbank": 1}


def test_pfa_on_clean_bags():
    m = compute_metrics(THREAT, CLEAN, POL)
    assert m["sweep"]["0.25"]["pfa"] == pytest.approx(1 / 10)
    assert m["sweep"]["0.15"]["pfa"] == pytest.approx(2 / 10)
    assert m["policy"]["pfa"] == pytest.approx(1 / 10)


def test_threshold_lowers_pd():
    m = compute_metrics(THREAT, CLEAN, POL)
    assert m["sweep"]["0.50"]["pd_correct"] == pytest.approx(1 / 4)


def test_wilson_interval_brackets_estimate():
    lo, hi = wilson(5, 100)
    assert lo < 0.05 < hi
    assert wilson(0, 50)[0] == 0.0
