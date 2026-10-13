import importlib.util
from pathlib import Path

import yaml

from xray.policy import Policy

ROOT = Path(__file__).resolve().parents[1]
TAX = yaml.safe_load((ROOT / "configs" / "taxonomy.yaml").read_text(encoding="utf-8"))

spec = importlib.util.spec_from_file_location("merge_datasets", ROOT / "scripts" / "merge_datasets.py")
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


def test_alias_targets_exist():
    assert set(TAX["aliases"].values()) <= set(TAX["classes"])


def test_known_dataset_names_map():
    stcray = ["Explosive", "Gun", "3D Printed Gun", "Shaving Razor", "Other Sharp Item", "Nail Cutter", "Powerbank"]
    pidray = [
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
    hixray = ["Portable_Charger_1", "Portable_Charger_2", "Water", "Laptop", "Nonmetallic_Lighter"]
    for names in (stcray, pidray):
        _, dropped = merge.build_map(names, TAX)
        assert dropped == [], dropped
    m, dropped = merge.build_map(hixray, TAX)
    assert sorted(dropped) == ["Laptop", "Water"]
    assert TAX["classes"][m[0]] == "Powerbank" and TAX["classes"][m[4]] == "Lighter"


def test_unified_policy_covers_every_class_once():
    p = Policy.load(ROOT / "configs" / "policy_unified.yaml")
    assert not (p.high_risk_items & p.review_items)
    assert p.prohibited == set(TAX["classes"])
