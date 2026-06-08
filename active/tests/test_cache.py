import json
from pathlib import Path
from palmdef_risk.cache import CacheManager


def test_mill_miss_when_files_absent(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    assert not cm.mill_valid(2020, 2023)


def test_mill_hit_when_files_present(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    d = cm.mill_dir(2020, 2023)
    d.mkdir(parents=True)
    (d / "mill_t2.gpkg").write_text("")
    (d / "mill_t3.gpkg").write_text("")
    assert cm.mill_valid(2020, 2023)


def test_forest_miss_when_no_metadata(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    key = cm.forest_key((108.0, -2.0, 114.0, 2.0), 5000, "tmf", [2015, 2020, 2024], 75)
    assert not cm.forest_valid(key, [108.5, -1.5, 113.5, 1.5])


def test_forest_hit_when_extent_covers(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    bbox = (108.5, -1.5, 113.5, 1.5)
    key = cm.forest_key(bbox, 5000, "tmf", [2015, 2020, 2024], 75)
    d = cm.forest_dir(key)
    d.mkdir(parents=True)
    meta = {"downloaded_extent": [107.0, -3.0, 115.0, 3.0]}
    (d / "metadata.json").write_text(json.dumps(meta))
    assert cm.forest_valid(key, [108.5, -1.5, 113.5, 1.5])


def test_forest_miss_when_extent_too_small(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    bbox = (108.5, -1.5, 113.5, 1.5)
    key = cm.forest_key(bbox, 5000, "tmf", [2015, 2020, 2024], 75)
    d = cm.forest_dir(key)
    d.mkdir(parents=True)
    meta = {"downloaded_extent": [109.0, -1.0, 113.0, 1.0]}
    (d / "metadata.json").write_text(json.dumps(meta))
    assert not cm.forest_valid(key, [108.5, -1.5, 113.5, 1.5])


def test_status_report_keys(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    k = cm.forest_key((0, 0, 1, 1), 0, "tmf", [], 75)
    kv = cm.variables_key((0, 0, 1, 1), 0, False, None, 180)
    report = cm.status_report(2020, 2023, [0, 0, 1, 1], k, kv)
    assert set(report.keys()) == {"mill", "forest", "variables"}
    assert report["mill"] == "miss"
