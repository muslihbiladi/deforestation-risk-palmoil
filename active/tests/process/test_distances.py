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
    write_raster(d / "plantation_t3.tif", arr, gt, 32750, nodata=255)
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
    ]
    for name in expected:
        assert (d / name).exists(), f"Missing: {name}"
    # Forecast distances now live under data/forecast/ with model names
    for name in ["dist_edge.tif", "dist_defor.tif", "dist_plantation_edge.tif"]:
        assert (d / "forecast" / name).exists(), f"Missing forecast: {name}"


def test_dist_mill_not_created(
    tmp_path, write_raster, write_vector, minimal_config_yaml
):
    from palmdef_risk.process.distances import compute_all_distances
    ctx = _make_ctx(tmp_path, write_raster, write_vector, minimal_config_yaml)
    compute_all_distances(ctx)
    assert not (ctx.data_dir / "dist_mill.tif").exists()


def _write_tiled(path, arr, dtype, nodata=None, block=16):
    from osgeo import osr
    drv = gdal.GetDriverByName("GTiff")
    ny, nx = arr.shape
    ds = drv.Create(
        str(path), nx, ny, 1, dtype,
        options=["TILED=YES", f"BLOCKXSIZE={block}", f"BLOCKYSIZE={block}"],
    )
    ds.SetGeoTransform([500000, 30, 0, 9000000, 0, -30])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32750)
    ds.SetProjection(srs.ExportToWkt())
    b = ds.GetRasterBand(1)
    b.WriteArray(arr)
    if nodata is not None:
        b.SetNoDataValue(nodata)
    ds.FlushCache()
    ds = None
    return path


def _proximity_baseline(src_path, out_path, target_value):
    """Pre-refactor full-array mask → MEM → ComputeProximity (the baseline)."""
    ds = gdal.Open(str(src_path))
    arr = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ny, nx = arr.shape
    ds = None
    mask = (arr == target_value).astype(np.uint8)
    mem = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(gt)
    mem.SetProjection(proj)
    mem.GetRasterBand(1).WriteArray(mask)
    out_ds = gdal.GetDriverByName("GTiff").Create(
        str(out_path), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    gdal.ComputeProximity(mem.GetRasterBand(1), out_ds.GetRasterBand(1),
                          options=["DISTUNITS=GEO"])
    out_ds.FlushCache()
    out_ds = None
    mem = None


@pytest.mark.parametrize("target_value", [0, 1])
def test_proximity_from_raster_windowed_matches_full(tmp_path, target_value):
    """Windowed-mask proximity is identical to the full-array-mask baseline."""
    from palmdef_risk.process.distances import _proximity_from_raster

    arr = np.ones((70, 70), dtype=np.uint8)
    arr[5, 5] = 0
    arr[40:45, 20:30] = 0      # block spanning tile boundaries
    arr[0, :] = 0
    arr[:, -1] = 0
    src = _write_tiled(tmp_path / "src.tif", arr, gdal.GDT_Byte, nodata=255)

    out_new = tmp_path / "new.tif"
    out_base = tmp_path / "base.tif"
    _proximity_from_raster(Path(src), out_new, target_value=target_value)
    _proximity_baseline(src, out_base, target_value=target_value)

    def _read(p):
        ds = gdal.Open(str(p))
        a = ds.GetRasterBand(1).ReadAsArray()
        nd = ds.GetRasterBand(1).GetNoDataValue()
        ds = None
        return a, nd

    got, nd = _read(out_new)
    expected, _ = _read(out_base)
    assert np.array_equal(got, expected)
    assert nd == -9999.0
