import numpy as np
from pathlib import Path
from osgeo import gdal, osr


def _write_multiband_tiled(path, band_arrs, nodata=255, block=16):
    """Multi-band Byte GeoTIFF, small tiles → exercises windowed export."""
    ny, nx = band_arrs[0].shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(path), nx, ny, len(band_arrs), gdal.GDT_Byte,
        options=["TILED=YES", f"BLOCKXSIZE={block}", f"BLOCKYSIZE={block}"],
    )
    ds.SetGeoTransform([500000, 30, 0, 9000000, 0, -30])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32750)
    ds.SetProjection(srs.ExportToWkt())
    for i, a in enumerate(band_arrs, start=1):
        b = ds.GetRasterBand(i)
        b.WriteArray(a)
        if nodata is not None:
            b.SetNoDataValue(nodata)
    ds.FlushCache()
    ds = None
    return path


def _read(path):
    ds = gdal.Open(str(path))
    arr = ds.GetRasterBand(1).ReadAsArray()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    return arr, nd


def test_export_bands_streaming_matches_source(tmp_path):
    """gdal.Translate per-band export copies each band's values verbatim."""
    from palmdef_risk.data.forest import export_bands
    rng = np.random.default_rng(1)
    bands = [rng.integers(0, 2, (70, 70)).astype(np.uint8) for _ in range(3)]
    bands[0][0, :] = 255   # nodata-valued pixels must survive the copy
    src = _write_multiband_tiled(tmp_path / "forest.tif", bands, nodata=255)

    out = export_bands(str(src), output_dir=str(tmp_path), prefix="forest_t",
                       verbose=False)
    assert len(out) == 3
    for b in range(1, 4):
        got, nd = _read(tmp_path / f"forest_t{b}.tif")
        assert np.array_equal(got, bands[b - 1]), f"band {b} mismatch"
        assert nd == 255


def test_export_period_fcc_windowed_matches_full(tmp_path):
    """Windowed period-FCC is bit-identical to the all-bands-in-RAM baseline."""
    from palmdef_risk.data.forest import export_period_fcc
    rng = np.random.default_rng(2)
    bands = [rng.integers(0, 2, (70, 70)).astype(np.uint8) for _ in range(3)]
    bands[0][3:6, 3:6] = 255   # nodata footprints differ per band
    bands[1][40, 40] = 255
    bands[2][0, :] = 255
    src = _write_multiband_tiled(tmp_path / "forest.tif", bands, nodata=255)

    out = export_period_fcc(str(src), output_dir=str(tmp_path), verbose=False)
    assert [Path(p).name for p in out] == ["fcc12.tif", "fcc23.tif"]

    # Baseline: the pre-refactor whole-array computation.
    nodata_mask = np.zeros((70, 70), dtype=bool)
    for a in bands:
        nodata_mask |= (a == 255)
    for i, fname in enumerate(["fcc12.tif", "fcc23.tif"]):
        fcc = (bands[i].astype(np.uint8) & bands[i + 1].astype(np.uint8))
        fcc[bands[i] == 0] = 255
        fcc[nodata_mask] = 255
        got, nd = _read(tmp_path / fname)
        assert np.array_equal(got, fcc), f"{fname} mismatch"
        assert nd == 255


def test_download_forest_passes_output_crs(minimal_config_yaml, tmp_path):
    """download_forest must pass output_crs=ctx.config.crs to get_fcc."""
    from unittest.mock import patch
    from palmdef_risk.io.run import create_run
    from palmdef_risk.data.forest import download_forest
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    captured = {}
    original_get_fcc = None
    try:
        import palmdef_risk.data.forest as fm
        original_get_fcc = fm.get_fcc
        def fake_get_fcc(*args, **kwargs):
            captured["output_crs"] = kwargs.get("output_crs")
            return {}
        fm.get_fcc = fake_get_fcc
        try:
            download_forest(ctx, use_cache=False)
        except Exception:
            pass
    finally:
        if original_get_fcc:
            fm.get_fcc = original_get_fcc
    if captured:
        assert captured.get("output_crs") == ctx.config.crs, (
            f"output_crs={captured.get('output_crs')} but expected {ctx.config.crs}"
        )


def test_forest_outputs_three_years():
    from palmdef_risk.data.forest import _forest_outputs
    assert _forest_outputs([2015, 2020, 2024]) == [
        "forest_cover.tif",
        "forest_t1.tif", "forest_t2.tif", "forest_t3.tif",
        "fcc123.tif", "fcc12.tif", "fcc23.tif",
    ]


def test_forest_complete_requires_full_set(tmp_path):
    """A partial download (only fcc23.tif) is NOT 'done'; the full set is."""
    from palmdef_risk.data.forest import _forest_complete, _forest_outputs
    years = [2015, 2020, 2024]
    out = tmp_path / "forest"
    out.mkdir()

    # Only the former guard file present → must report 'not done' so it re-runs.
    (out / "fcc23.tif").write_bytes(b"x")
    assert _forest_complete(out, years) is False

    # Full set, all non-empty → done.
    for name in _forest_outputs(years):
        (out / name).write_bytes(b"x")
    assert _forest_complete(out, years) is True


def test_forest_complete_rejects_empty_file(tmp_path):
    """A zero-byte output counts as incomplete (interrupted write)."""
    from palmdef_risk.data.forest import _forest_complete, _forest_outputs
    years = [2015, 2020, 2024]
    out = tmp_path / "forest"
    out.mkdir()
    for name in _forest_outputs(years):
        (out / name).write_bytes(b"x")
    (out / "forest_t2.tif").write_bytes(b"")   # truncated → incomplete
    assert _forest_complete(out, years) is False
