import pytest
import yaml
from pathlib import Path
from palmdef_risk.io.run import create_run, load_run, RunContext


def test_create_run_builds_folder_tree(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    assert ctx.data_dir.exists()
    assert (ctx.data_dir / "raw" / "forest").exists()
    assert (ctx.data_dir / "raw" / "variables").exists()
    assert (ctx.data_dir / "raw" / "mill").exists()
    assert (ctx.data_dir / "raw" / "user_inputs").exists()
    assert (ctx.run_dir / "output" / "models").exists()
    assert (ctx.run_dir / "output" / "diagnostics").exists()
    assert (ctx.run_dir / "logs").exists()


def test_create_run_freezes_config(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    assert (ctx.run_dir / "config.yaml").exists()


def test_create_run_folder_name_contains_project(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    assert "test_proj" in ctx.run_dir.name
    assert "test_area" in ctx.run_dir.name
    assert "test" in ctx.run_dir.name


def test_load_run_reconstructs_context(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    ctx2 = load_run(ctx.run_dir)
    assert ctx2.config.project == ctx.config.project
    assert ctx2.run_dir == ctx.run_dir


def test_create_run_dry_run_does_not_create_folder(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs", dry_run=True)
    assert not ctx.run_dir.exists()


def test_run_context_paths(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    assert ctx.raw_dir == ctx.data_dir / "raw"
    assert ctx.output_dir == ctx.run_dir / "output"
    assert ctx.log_dir == ctx.run_dir / "logs"


def test_crs_autodetected_when_null(tmp_path, user_input_files):
    """When config.crs is null, create_run fills it in via utm.py."""
    cfg = {
        "run": {"project": "p", "area": "a", "task": "t"},
        "aoi": {"source": str(user_input_files["hgu"]), "buffer": 0.0},
        "crs": None,
        "cache_dir": "cache/",
        "forest": {"source": "tmf", "years": [2015, 2020, 2024], "perc": 75},
        "variables": {"use_ghsl_towns": False, "ghsl_years": None, "osm_timeout": 180},
        "user_inputs": {
            "peatland": {"path": str(user_input_files["peatland"]), "type": "binary"},
            "hgu": {"path": str(user_input_files["hgu"])},
            "plantation": {"t2": str(user_input_files["plantation_t2"]), "t3": None,
                           "industrial_value": 1, "smallholder_value": 2},
        },
        "mill": {"source": "trase", "path": None},
        "process": {"gravity": {"sigma_km": 25.0, "radius_km": 80.0},
                    "sensitivity": {"sigmas_km": [15.0, 25.0, 40.0]}},
        "model": {"variants": ["A"], "nsamp": 100, "csize": 10, "Vbeta": 1000,
                  "burnin": 10, "mcmc": 10, "thin": 1, "seed": 42},
        "parallel": {"max_workers": None, "cpu_fraction": 0.9, "ram_per_dist_gb": 0.5,
                     "ram_per_icar_gb": 1.0, "ram_per_predict_gb": 0.75},
        "output": {"project_future": False, "projection_year": 2035},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    ctx = create_run(p, runs_root=tmp_path / "runs")
    assert ctx.config.crs is not None
    assert ctx.config.crs.startswith("EPSG:")


def test_run_subdirs_no_kde_or_correlation(tmp_path, minimal_config_yaml):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    dirs = [str(p.relative_to(ctx.run_dir)) for p in ctx.run_dir.rglob("*") if p.is_dir()]
    assert not any("kde" in d for d in dirs)
    assert not any("correlation" in d for d in dirs)
    assert any("diagnostics" in d for d in dirs)
