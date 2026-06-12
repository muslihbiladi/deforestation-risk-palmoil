# tests/process/test_plantation.py
import numpy as np
from osgeo import gdal


def _f32(write_raster, tmp_path, name, arr, gt):
    return write_raster(tmp_path / name, arr.astype(np.float32), gt, 32750,
                        dtype=gdal.GDT_Float32, nodata=-9999.0)


def test_orthogonalize_plantation_writes_residual(tmp_path, write_raster):
    from palmdef_risk.process.plantation import orthogonalize_plantation
    rng = np.random.default_rng(0)
    gt = [500000, 100, 0, 9005000, 0, -100]
    plant = rng.uniform(1, 5000, (20, 20))
    edge = rng.uniform(1, 5000, (20, 20))
    defor = rng.uniform(1, 5000, (20, 20))
    road = rng.uniform(1, 5000, (20, 20))
    out = tmp_path / "plantation_resid.tif"
    r2 = orthogonalize_plantation(
        _f32(write_raster, tmp_path, "dist_plantation_edge.tif", plant, gt),
        _f32(write_raster, tmp_path, "dist_edge.tif", edge, gt),
        _f32(write_raster, tmp_path, "dist_defor.tif", defor, gt),
        _f32(write_raster, tmp_path, "dist_road.tif", road, gt),
        out,
    )
    assert out.exists()
    assert 0.0 <= r2 <= 1.0
    ds = gdal.Open(str(out))
    resid = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    valid = resid[resid != -9999.0]
    # OLS residual is mean-zero by construction
    assert abs(valid.mean()) < 1e-3


def test_orthogonalize_plantation_respects_nodata(tmp_path, write_raster):
    from palmdef_risk.process.plantation import orthogonalize_plantation
    gt = [500000, 100, 0, 9001000, 0, -100]
    plant = np.full((10, 10), 100.0)
    plant[0, 0] = -9999.0  # nodata pixel must stay nodata in output
    edge = np.full((10, 10), 50.0)
    defor = np.full((10, 10), 75.0)
    road = np.full((10, 10), 25.0)
    out = tmp_path / "plantation_resid.tif"
    orthogonalize_plantation(
        _f32(write_raster, tmp_path, "dist_plantation_edge.tif", plant, gt),
        _f32(write_raster, tmp_path, "dist_edge.tif", edge, gt),
        _f32(write_raster, tmp_path, "dist_defor.tif", defor, gt),
        _f32(write_raster, tmp_path, "dist_road.tif", road, gt),
        out,
    )
    ds = gdal.Open(str(out))
    resid = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    assert resid[0, 0] == -9999.0


def test_compute_plantation_resid_forecast(tmp_path, write_raster, write_vector,
                                           minimal_config_yaml):
    import numpy as np
    from osgeo import gdal
    from palmdef_risk.io.run import create_run
    from palmdef_risk.process.plantation import compute_plantation_resid_forecast

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    d = ctx.data_dir
    fcast = d / "forecast"
    fcast.mkdir(parents=True, exist_ok=True)
    gt = [500000, 100, 0, 9001000, 0, -100]
    rng = np.random.default_rng(1)
    for path in [fcast / "dist_plantation_edge.tif", fcast / "dist_edge.tif",
                 fcast / "dist_defor.tif", d / "dist_road.tif"]:
        write_raster(path, rng.uniform(1, 5000, (10, 10)).astype(np.float32),
                     gt, 32750, dtype=gdal.GDT_Float32, nodata=-9999.0)

    r2 = compute_plantation_resid_forecast(ctx)
    assert (fcast / "plantation_resid.tif").exists()
    assert 0.0 <= r2 <= 1.0


def test_compute_plantation_resid_skips_when_absent(tmp_path, write_vector,
                                                    minimal_config_yaml):
    from palmdef_risk.io.run import create_run
    from palmdef_risk.process.plantation import compute_plantation_resid
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    # No dist_plantation_edge.tif in data_dir → must skip, return 0.0
    assert compute_plantation_resid(ctx) == 0.0
    assert not (ctx.data_dir / "plantation_resid.tif").exists()
