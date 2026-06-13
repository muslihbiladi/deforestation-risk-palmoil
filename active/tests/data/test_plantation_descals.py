"""Tests for the Descals plantation download accumulation core.

The Zenodo network path (ensure_descals_cache) is NOT exercised here — only the
pure accumulation logic and the GDAL clip/accumulate pipeline against synthetic
local rasters (no network).
"""
import numpy as np
from osgeo import gdal, osr

from palmdef_risk.data.plantation import accumulate_classes, _build_year_raster


def _write(path, arr, gt, epsg=4326, dtype=gdal.GDT_Byte, nodata=None):
    drv = gdal.GetDriverByName("GTiff")
    ny, nx = arr.shape
    ds = drv.Create(str(path), nx, ny, 1, dtype)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(arr)
    if nodata is not None:
        ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds.FlushCache()
    ds = None
    return path


# ── accumulate_classes (pure) ────────────────────────────────────────────────

def test_accumulate_thresholds_by_year():
    extent = np.array([[1, 2, 0], [1, 2, 1]], dtype=np.uint8)
    yop = np.array([[1995, 2010, 0], [2018, 2005, 2020]], dtype=np.int16)
    out, eff, clamped = accumulate_classes(extent, yop, 2010)
    assert eff == 2010 and not clamped
    # planted <= 2010 keeps its class; later plantings and non-palm -> 0
    assert out.tolist() == [[1, 2, 0], [0, 2, 0]]


def test_accumulate_clamps_beyond_dataset_max():
    extent = np.array([[1, 2]], dtype=np.uint8)
    yop = np.array([[2021, 1990]], dtype=np.int16)
    out, eff, clamped = accumulate_classes(extent, yop, 2035)
    assert eff == 2021 and clamped is True
    assert out.tolist() == [[1, 2]]


def test_accumulate_excludes_below_min_year():
    extent = np.array([[1, 1]], dtype=np.uint8)
    yop = np.array([[1989, 1990]], dtype=np.int16)  # 1989 predates dataset
    out, _, _ = accumulate_classes(extent, yop, 2000)
    assert out.tolist() == [[0, 1]]


def test_accumulate_remaps_class_values():
    extent = np.array([[1, 2]], dtype=np.uint8)
    yop = np.array([[2000, 2000]], dtype=np.int16)
    out, _, _ = accumulate_classes(extent, yop, 2005,
                                   industrial_value=10, smallholder_value=20)
    assert out.tolist() == [[10, 20]]


# ── _build_year_raster (GDAL clip + accumulate) ──────────────────────────────

def test_build_year_raster_matches_accumulation(tmp_path):
    # 3x3 grid in EPSG:4326 at 0.001 deg; extent and yop share the same grid.
    res = 0.001
    xmin, ymax = 100.0, 2.0
    nx = ny = 3
    gt = [xmin, res, 0, ymax, 0, -res]
    extent = np.array([[1, 2, 0],
                       [1, 0, 2],
                       [2, 1, 0]], dtype=np.uint8)
    yop = np.array([[1995, 2010, 0],
                    [2019, 0, 2005],
                    [2000, 2018, 0]], dtype=np.int16)
    ext_p = _write(tmp_path / "extent.tif", extent, gt)
    yop_p = _write(tmp_path / "yop.tif", yop, gt, dtype=gdal.GDT_Int16)

    aoi = (xmin, ymax - ny * res, xmin + nx * res, ymax)  # full bbox
    out = tmp_path / "out" / "plantation_t2.tif"
    _build_year_raster(ext_p, yop_p, aoi, 2010, out, output_crs=None,
                       verbose=False)

    ds = gdal.Open(str(out))
    arr = ds.GetRasterBand(1).ReadAsArray()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None

    expected, _, _ = accumulate_classes(extent, yop, 2010)
    assert arr.tolist() == expected.tolist()
    assert nd == 255
