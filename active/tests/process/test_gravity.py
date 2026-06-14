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


def test_gaussian_filter_truncates_beyond_radius(tmp_path, write_raster):
    """Mills beyond radius_km must contribute exactly 0 (hard circular catchment).

    Regression for the previously dead radius_km param: the old full-support FFT
    kernel left a nonzero tail past radius_km (~14% of mass beyond 2*sigma at the
    sensitivity-sweep bandwidth). WORKFLOW §3.3 specifies accessibility summed
    only over mills "within 80 km".
    """
    from palmdef_risk.process.gravity import _apply_gaussian_filter
    arr = np.zeros((101, 101), dtype=np.uint8)
    arr[50, 50] = 1  # single mill at centre
    ref = write_raster(tmp_path / "mills.tif", arr,
                       gt=[500000, 100, 0, 9005000, 0, -100], epsg=32750)
    out = tmp_path / "gravity_raw.tif"
    # 100 m pixels: sigma = 1 km (10 px), radius = 2 km (20 px).
    _apply_gaussian_filter(ref, out, sigma_km=1.0, radius_km=2.0)
    ds = gdal.Open(str(out))
    result = ds.GetRasterBand(1).ReadAsArray().astype(float)
    ds = None

    assert result[50, 50] == result.max()      # mill pixel is the maximum
    assert result[50, 60] > 0.0                 # 10 px = 1 km < 2 km: inside catchment
    assert result[50, 80] < 1e-9                # 30 px = 3 km > 2 km: truncated along axis
    # 18 px per axis (Euclid 25.5 px = 2.5 km > 2 km): must be 0 -> CIRCULAR catchment,
    # not a separable square one.
    assert result[68, 68] < 1e-9


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
