# tests/process/test_distances.py
import numpy as np
import pytest
from pathlib import Path
from osgeo import gdal


def _make_ctx(tmp_path, write_raster, write_vector, minimal_config_yaml):
    from palmdef_risk.io.run import create_run
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    arr = np.ones((10, 10), dtype=np.uint8)
    arr[5, 5] = 0
    gt = [500000, 30, 0, 9000300, 0, -30]
    d = ctx.data_dir
    d.mkdir(parents=True, exist_ok=True)
    write_raster(d / "forest_t2.tif", arr, gt, 32750, nodata=255)
    write_raster(d / "fcc12.tif", arr, gt, 32750, nodata=255)
    write_raster(d / "forest_t3.tif", arr, gt, 32750, nodata=255)
    write_raster(d / "fcc23.tif", arr, gt, 32750, nodata=255)
    write_raster(d / "plantation.tif", arr, gt, 32750, nodata=255)
    # Vector inputs are read from raw_dir/variables (where Stage 1 downloads them)
    vec_dir = ctx.raw_dir / "variables"
    vec_dir.mkdir(parents=True, exist_ok=True)
    write_vector(vec_dir / "road.gpkg", epsg=32750)
    write_vector(vec_dir / "river.gpkg", epsg=32750)
    write_vector(vec_dir / "town.gpkg", epsg=32750)
    return ctx


def test_compute_all_distances_creates_expected_files(
    tmp_path, write_raster, write_vector, minimal_config_yaml
):
    from palmdef_risk.process.distances import compute_all_distances
    ctx = _make_ctx(tmp_path, write_raster, write_vector, minimal_config_yaml)
    compute_all_distances(ctx)
    d = ctx.data_dir
    expected = [
        "dist_edge.tif", "dist_defor.tif",
        "dist_road.tif", "dist_river.tif", "dist_town.tif",
        "dist_plantation_edge.tif",
        "dist_edge_forecast.tif", "dist_defor_forecast.tif",
    ]
    for name in expected:
        assert (d / name).exists(), f"Missing: {name}"


def test_dist_mill_not_created(
    tmp_path, write_raster, write_vector, minimal_config_yaml
):
    from palmdef_risk.process.distances import compute_all_distances
    ctx = _make_ctx(tmp_path, write_raster, write_vector, minimal_config_yaml)
    compute_all_distances(ctx)
    assert not (ctx.data_dir / "dist_mill.tif").exists()
