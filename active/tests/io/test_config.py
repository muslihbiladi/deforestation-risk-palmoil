import pytest
import yaml
from pathlib import Path
from palmdef_risk.io.config import RunConfig


def test_new_fields_parse(minimal_config_yaml):
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert cfg.sigma_km == 25.0
    assert cfg.radius_km == 80.0
    assert cfg.sensitivity_sigmas == [15.0, 25.0, 40.0]
    assert cfg.Vbeta == 1000
    assert cfg.nsamp == 10000
    assert cfg.mill_source == "trase"
    assert cfg.cache_dir == "cache/"


def test_crs_may_be_null(tmp_path, user_input_files):
    cfg_dict = _base_cfg(user_input_files)
    cfg_dict["crs"] = None
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg_dict))
    cfg = RunConfig.from_yaml(p)
    assert cfg.crs is None


def test_old_lq_fields_absent(minimal_config_yaml):
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert not hasattr(cfg, "kde_bandwidth_km")
    assert not hasattr(cfg, "lq_direction")
    assert not hasattr(cfg, "run_gwr")


def test_validation_passes_on_valid_config(minimal_config_yaml):
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    errors = cfg.validate()
    assert errors == []


def test_validation_unknown_variant(minimal_config_yaml, tmp_path):
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["model"]["variants"] = ["A", "Z"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(d))
    errors = RunConfig.from_yaml(p).validate()
    assert any("variant" in e for e in errors)


def test_vbeta_warning_logged(minimal_config_yaml, tmp_path, caplog):
    import logging
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["model"]["Vbeta"] = 500
    p = tmp_path / "vbeta.yaml"
    p.write_text(yaml.dump(d))
    with caplog.at_level(logging.WARNING, logger="palmdef_risk"):
        RunConfig.from_yaml(p).validate()
    assert any("Vbeta" in r.message for r in caplog.records)


def test_sigma_gt_zero_validation(minimal_config_yaml, tmp_path):
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["process"]["gravity"]["sigma_km"] = 0
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(d))
    errors = RunConfig.from_yaml(p).validate()
    assert any("sigma" in e for e in errors)


def test_validation_fails_bad_projection_year(minimal_config_yaml, tmp_path):
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["output"]["project_future"] = True
    d["output"]["projection_year"] = 2020
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(d))
    errors = RunConfig.from_yaml(p).validate()
    assert any("projection_year" in e for e in errors)


def test_run_folder_name_format(minimal_config_yaml):
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    name = cfg.run_folder_name("20260511_143022")
    assert name == "test_proj_test_area_test_20260511_143022"


def test_default_variants_are_a_to_e(minimal_config_yaml):
    import yaml
    from palmdef_risk.io.config import RunConfig
    raw = yaml.safe_load(minimal_config_yaml.read_text())
    del raw["model"]["variants"]  # force the default
    minimal_config_yaml.write_text(yaml.dump(raw))
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert cfg.model_variants == ["A", "B", "C", "D", "E"]


def test_validate_accepts_d_and_e(minimal_config_yaml):
    import yaml
    from palmdef_risk.io.config import RunConfig
    raw = yaml.safe_load(minimal_config_yaml.read_text())
    raw["model"]["variants"] = ["A", "D", "E"]
    minimal_config_yaml.write_text(yaml.dump(raw))
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert "model.variants" not in " ".join(cfg.validate())


def test_validate_rejects_unknown_variant(minimal_config_yaml):
    import yaml
    from palmdef_risk.io.config import RunConfig
    raw = yaml.safe_load(minimal_config_yaml.read_text())
    raw["model"]["variants"] = ["A", "Z"]
    minimal_config_yaml.write_text(yaml.dump(raw))
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert any("unknown variant" in e.lower() for e in cfg.validate())


def test_plantation_source_defaults_to_user(minimal_config_yaml):
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert cfg.plantation_source == "user"


def test_plantation_source_download_parses(minimal_config_yaml, tmp_path):
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["user_inputs"]["plantation"]["source"] = "download"
    p = tmp_path / "dl.yaml"
    p.write_text(yaml.dump(d))
    cfg = RunConfig.from_yaml(p)
    assert cfg.plantation_source == "download"
    assert cfg.validate() == []  # forest.years has t1,t2,t3


def test_plantation_source_download_requires_three_forest_years(minimal_config_yaml, tmp_path):
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["user_inputs"]["plantation"]["source"] = "download"
    d["forest"]["years"] = [2015, 2020]  # only t1, t2
    p = tmp_path / "dl2.yaml"
    p.write_text(yaml.dump(d))
    errors = RunConfig.from_yaml(p).validate()
    assert any("plantation.source=download" in e for e in errors)


def test_plantation_source_invalid_value_rejected(minimal_config_yaml, tmp_path):
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["user_inputs"]["plantation"]["source"] = "satellite"
    p = tmp_path / "bad_src.yaml"
    p.write_text(yaml.dump(d))
    errors = RunConfig.from_yaml(p).validate()
    assert any("plantation.source" in e for e in errors)


def _base_cfg(uif):
    return {
        "run": {"project": "test_proj", "area": "test_area", "task": "test"},
        "aoi": {"source": str(uif["hgu"]), "buffer": 0.0},
        "crs": "EPSG:32750",
        "cache_dir": "cache/",
        "forest": {"source": "tmf", "years": [2015, 2020, 2024], "perc": 75},
        "variables": {"use_ghsl_towns": False, "ghsl_years": None, "osm_timeout": 180},
        "user_inputs": {
            "peatland": {"path": str(uif["peatland"]), "type": "binary"},
            "hgu": {"path": str(uif["hgu"])},
            "plantation": {"t2": str(uif["plantation_t2"]), "t3": None,
                           "industrial_value": 1, "smallholder_value": 2},
        },
        "mill": {"source": "trase", "path": None},
        "process": {
            "gravity": {"sigma_km": 25.0, "radius_km": 80.0},
            "sensitivity": {"sigmas_km": [15.0, 25.0, 40.0]},
        },
        "model": {
            "variants": ["A", "B"], "nsamp": 10000, "csize": 10,
            "Vbeta": 1000, "burnin": 100, "mcmc": 100, "thin": 1, "seed": 42,
        },
        "parallel": {"max_workers": None, "cpu_fraction": 0.9,
                     "ram_per_dist_gb": 0.5, "ram_per_icar_gb": 1.0,
                     "ram_per_predict_gb": 0.75},
        "output": {"project_future": False, "projection_year": 2035},
    }
