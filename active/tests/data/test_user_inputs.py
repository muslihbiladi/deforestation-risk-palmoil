import pytest
from pathlib import Path
from palmdef_risk.io.run import create_run
from palmdef_risk.data.user_inputs import ingest_user_inputs


def test_ingest_copies_all_files(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    result = ingest_user_inputs(ctx)
    assert result["peatland"].exists()
    assert result["hgu"].exists()
    assert result["plantation_t2"].exists()
    assert result["plantation_t3"] is None


def test_ingest_copies_to_raw_user_inputs(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    result = ingest_user_inputs(ctx)
    assert ctx.raw_dir / "user_inputs" in result["peatland"].parents


def test_ingest_fails_missing_peatland(minimal_config_yaml, tmp_path):
    import yaml
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["user_inputs"]["peatland"]["path"] = "/nonexistent/peatland.gpkg"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump(d))
    ctx = create_run(bad, runs_root=tmp_path / "runs")
    with pytest.raises(FileNotFoundError, match="peatland"):
        ingest_user_inputs(ctx)


def test_ingest_fails_undefined_crs(tmp_path, user_input_files, minimal_config_yaml):
    from osgeo import ogr, osr
    # Create a vector without CRS
    no_crs_path = tmp_path / "no_crs.gpkg"
    driver = ogr.GetDriverByName("GPKG")
    ds = driver.CreateDataSource(str(no_crs_path))
    ds.CreateLayer("layer", None, ogr.wkbPolygon)
    ds.FlushCache()
    ds = None

    import yaml
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["user_inputs"]["peatland"]["path"] = str(no_crs_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump(d))
    ctx = create_run(bad, runs_root=tmp_path / "runs")
    with pytest.raises(ValueError, match="CRS undefined"):
        ingest_user_inputs(ctx)


def test_river_source_user_copies_file(minimal_config_yaml, tmp_path, tiny_vector):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    ctx.config.river_source = "user"
    ctx.config.river_path = str(tiny_vector)
    result = ingest_user_inputs(ctx)
    assert result["river"] is not None
    assert result["river"].exists()


def test_river_source_big_skips_copy(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    ctx.config.river_source = "big"
    result = ingest_user_inputs(ctx)
    assert result["river"] is None


def test_river_source_user_without_path_raises(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    ctx.config.river_source = "user"
    ctx.config.river_path = None
    with pytest.raises(ValueError, match="river.path"):
        ingest_user_inputs(ctx)
