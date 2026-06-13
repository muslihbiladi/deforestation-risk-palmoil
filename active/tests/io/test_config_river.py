import yaml
import pytest
from palmdef_risk.io.config import RunConfig


def _base_cfg_dict(tmp_path):
    """Minimal valid config dict; river block omitted on purpose."""
    aoi = tmp_path / "aoi.gpkg"
    aoi.write_text("")  # presence only; from_yaml does not open it
    return {
        "run": {"project": "p", "area": "a", "task": "t"},
        "aoi": {"source": str(aoi), "buffer": 0.0},
        "crs": "EPSG:32750",
        "forest": {"source": "tmf", "years": [2015, 2020, 2024], "perc": 75},
        "variables": {"use_ghsl_towns": False, "ghsl_years": None, "osm_timeout": 180},
        "user_inputs": {
            "peatland": {"path": "x", "type": "binary"},
            "hgu": {"path": "x"},
            "plantation": {"t2": None, "t3": None},
        },
        "mill": {"source": "trase", "path": None},
        "process": {"gravity": {"sigma_km": 25.0, "radius_km": 80.0}},
        "model": {"variants": ["A"]},
        "output": {"project_future": False, "projection_year": 2035},
    }


def _write(tmp_path, d):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(d))
    return p


def test_river_source_defaults_to_big(tmp_path):
    cfg = RunConfig.from_yaml(_write(tmp_path, _base_cfg_dict(tmp_path)))
    assert cfg.river_source == "big"


def test_river_source_parsed_from_yaml(tmp_path):
    d = _base_cfg_dict(tmp_path)
    d["user_inputs"]["river"] = {"source": "osm", "path": None}
    cfg = RunConfig.from_yaml(_write(tmp_path, d))
    assert cfg.river_source == "osm"
    assert cfg.river_path is None


def test_invalid_river_source_rejected(tmp_path):
    d = _base_cfg_dict(tmp_path)
    d["user_inputs"]["river"] = {"source": "nope", "path": None}
    cfg = RunConfig.from_yaml(_write(tmp_path, d))
    errs = cfg.validate()
    assert any("river.source" in e for e in errs)


def test_user_source_requires_path(tmp_path):
    d = _base_cfg_dict(tmp_path)
    d["user_inputs"]["river"] = {"source": "user", "path": None}
    cfg = RunConfig.from_yaml(_write(tmp_path, d))
    errs = cfg.validate()
    assert any("river.path required" in e for e in errs)


def test_user_source_with_path_ok(tmp_path):
    rv = tmp_path / "myriver.gpkg"
    rv.write_text("")
    d = _base_cfg_dict(tmp_path)
    d["user_inputs"]["river"] = {"source": "user", "path": str(rv)}
    cfg = RunConfig.from_yaml(_write(tmp_path, d))
    errs = cfg.validate()
    assert not any("river" in e for e in errs)
