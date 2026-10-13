import pytest

from xray.data import iou, load_data_yaml, read_gt, resolve_split_dir, size_bin

NAMES = ["Gun", "Knife"]


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_reads_box_and_polygon_labels(tmp_path):
    img = tmp_path / "valid" / "images" / "a.jpg"
    _write(img, "x")
    _write(tmp_path / "valid" / "labels" / "a.txt", "0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.3 0.1 0.3 0.2 0.1 0.2\n")
    g = read_gt(img, NAMES)
    assert [x["cls"] for x in g] == ["Gun", "Knife"]
    assert g[0]["fmt"] == "box" and g[0]["box"] == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert g[1]["fmt"] == "poly" and g[1]["box"] == pytest.approx((0.1, 0.1, 0.3, 0.2))
    assert g[1]["area"] == pytest.approx(0.02)


def test_missing_label_file_is_background(tmp_path):
    img = tmp_path / "images" / "bg.jpg"
    _write(img, "x")
    assert read_gt(img, NAMES) == []


def test_resolves_broken_roboflow_relative_paths(tmp_path):
    # Roboflow writes '../valid/images' although valid/ sits next to data.yaml
    _write(tmp_path / "valid" / "images" / "a.jpg", "x")
    _write(tmp_path / "data.yaml", "train: ../train/images\nval: ../valid/images\nnames: [Gun, Knife]\n")
    cfg = load_data_yaml(tmp_path / "data.yaml")
    assert resolve_split_dir(cfg, "val") == (tmp_path / "valid" / "images").resolve()


def test_iou_and_size_bins():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    assert iou((0, 0, 2, 1), (1, 0, 3, 1)) == pytest.approx(1 / 3)
    assert size_bin(0.001) == "tiny" and size_bin(0.005) == "small" and size_bin(0.2) == "large"
