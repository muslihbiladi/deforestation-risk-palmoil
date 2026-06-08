# tests/process/test_gravity.py
import numpy as np
import pytest
from osgeo import gdal


def test_gaussian_filter_higher_near_source(tmp_path, write_raster):
    from palmdef_risk.process.gravity import _apply_gaussian_filter
    arr = np.zeros((50, 50), dtype=np.uint8)
    arr[25, 25] = 1
    ref = write_raster(tmp_path / "mills.tif", arr,
                       gt=[500000, 100, 0, 9005000, 0, -100], epsg=32750)
    out = tmp_path / "gravity_raw.tif"
    _apply_gaussian_filter(ref, out, sigma_km=0.5, radius_km=2.0)
    ds = gdal.Open(str(out))
    result = ds.GetRasterBand(1).ReadAsArray().astype(float)
    ds = None
    assert result[25, 25] == result.max()
    assert result[0, 0] < result[25, 25] * 0.01


def test_orthogonalize_produces_residual_raster(tmp_path, write_raster):
    """orthogonalize_gravity must write gravity_resid.tif."""
    from palmdef_risk.process.gravity import orthogonalize_gravity
    rng = np.random.default_rng(42)
    gt = [500000, 100, 0, 9005000, 0, -100]
    gravity = rng.uniform(0, 1, (20, 20)).astype(np.float32)
    road = rng.uniform(0, 5000, (20, 20)).astype(np.float32)
    town = rng.uniform(0, 20000, (20, 20)).astype(np.float32)
    g_path = write_raster(tmp_path / "gravity_raw.tif", gravity, gt, 32750,
                          dtype=gdal.GDT_Float32, nodata=-9999.0)
    r_path = write_raster(tmp_path / "dist_road.tif", road, gt, 32750,
                          dtype=gdal.GDT_Float32, nodata=-9999.0)
    t_path = write_raster(tmp_path / "dist_town.tif", town, gt, 32750,
                          dtype=gdal.GDT_Float32, nodata=-9999.0)
    out = tmp_path / "gravity_resid.tif"
    orthogonalize_gravity(g_path, r_path, t_path, out)
    assert out.exists()
    ds = gdal.Open(str(out))
    resid = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    valid = resid[resid != -9999.0]
    assert abs(valid.mean()) < 0.1
