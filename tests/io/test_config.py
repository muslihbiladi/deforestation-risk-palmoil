import pytest
import yaml
from pathlib import Path
from palmdef_risk.io.config import RunConfig


def test_load_minimal_config(minimal_config_yaml):
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert cfg.project == "test_proj"
    assert cfg.forest_years == [2015, 2020, 2024]
    assert cfg.peatland_type == "binary"
    assert cfg.plantation_t3 is None
    assert cfg.lq_direction == "mp"


def test_validation_passes_on_valid_config(minimal_config_yaml):
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    errors = cfg.validate()
    assert errors == []


def test_validation_fails_bad_projection_year(minimal_config_yaml, tmp_path):
    import yaml
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["output"]["project_future"] = True
    d["output"]["projection_year"] = 2020  # <= years[-1]=2024
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump(d))
    cfg = RunConfig.from_yaml(bad)
    errors = cfg.validate()
    assert any("projection_year" in e for e in errors)


def test_validation_fails_unknown_model_variant(minimal_config_yaml, tmp_path):
    import yaml
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["model"]["variants"] = ["A", "Z"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump(d))
    cfg = RunConfig.from_yaml(bad)
    errors = cfg.validate()
    assert any("variant" in e for e in errors)


def test_validation_fails_ghsl_without_years(minimal_config_yaml, tmp_path):
    import yaml
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["variables"]["use_ghsl_towns"] = True
    d["variables"]["ghsl_years"] = None
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump(d))
    cfg = RunConfig.from_yaml(bad)
    errors = cfg.validate()
    assert any("ghsl_years" in e for e in errors)


def test_run_folder_name_format(minimal_config_yaml):
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    name = cfg.run_folder_name("20260511_143022")
    assert name == "test_proj_test_area_test_20260511_143022"
