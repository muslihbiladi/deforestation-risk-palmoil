import pytest
import numpy as np
from pathlib import Path
from osgeo import gdal


def _write_float32(path, arr, gt, proj):
    driver = gdal.GetDriverByName("GTiff")
    ny, nx = arr.shape
    ds = driver.Create(str(path), nx, ny, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
    ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    ds.FlushCache()
    ds = None


def test_compute_lq_mp_pm_reciprocal(tmp_path):
    from palmdef_risk.process.lq import compute_lq
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32750)
    proj = srs.ExportToWkt()
    gt = [500000, 30, 0, 9000000, 0, -30]

    M = np.full((10, 10), 2.0, dtype=np.float32)
    P = np.full((10, 10), 4.0, dtype=np.float32)
    m_path = tmp_path / "M.tif"
    p_path = tmp_path / "P.tif"
    _write_float32(m_path, M, gt, proj)
    _write_float32(p_path, P, gt, proj)

    mp_out = tmp_path / "lq_mp.tif"
    pm_out = tmp_path / "lq_pm.tif"
    compute_lq(m_path, p_path, mp_out, pm_out, epsilon=0.001)

    ds_mp = gdal.Open(str(mp_out))
    lq_mp = ds_mp.GetRasterBand(1).ReadAsArray()
    ds_mp = None

    valid = lq_mp[lq_mp != -9999.0]
    assert np.allclose(valid, 1.0, atol=0.01)


def test_compute_lq_respects_nodata(tmp_path):
    from palmdef_risk.process.lq import compute_lq
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32750)
    proj = srs.ExportToWkt()
    gt = [500000, 30, 0, 9000000, 0, -30]

    M = np.ones((10, 10), dtype=np.float32)
    M[0, 0] = -9999.0
    P = np.ones((10, 10), dtype=np.float32)
    m_path = tmp_path / "M.tif"
    p_path = tmp_path / "P.tif"
    _write_float32(m_path, M, gt, proj)
    _write_float32(p_path, P, gt, proj)

    mp_out = tmp_path / "lq_mp.tif"
    pm_out = tmp_path / "lq_pm.tif"
    compute_lq(m_path, p_path, mp_out, pm_out)

    ds = gdal.Open(str(mp_out))
    result = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    assert result[0, 0] == -9999.0


def test_classify_lq_zones(tmp_path):
    from palmdef_risk.process.lq import classify_lq_zones
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32750)
    proj = srs.ExportToWkt()
    gt = [500000, 30, 0, 9000000, 0, -30]

    lq = np.array([[0.3, 0.6, 1.0, 1.3, 1.7]], dtype=np.float32)
    lq_path = tmp_path / "lq.tif"

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(lq_path), 5, 1, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(lq.astype(np.float32))
    ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    ds.FlushCache()
    ds = None

    zones_path = tmp_path / "lq_zones.tif"
    classify_lq_zones(lq_path, zones_path)

    ds = gdal.Open(str(zones_path))
    zones = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    assert list(zones[0]) == [1, 2, 3, 4, 5]
