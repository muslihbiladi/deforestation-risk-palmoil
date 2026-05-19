import pytest
from pathlib import Path
from palmdef_risk.io.run import create_run, load_run, RunContext


def test_create_run_builds_folder_tree(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    assert ctx.data_dir.exists()
    assert (ctx.data_dir / "raw" / "forest").exists()
    assert (ctx.data_dir / "raw" / "variables").exists()
    assert (ctx.data_dir / "raw" / "mill").exists()
    assert (ctx.data_dir / "raw" / "user_inputs").exists()
    assert (ctx.data_dir / "intermediate" / "kde").exists()
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
