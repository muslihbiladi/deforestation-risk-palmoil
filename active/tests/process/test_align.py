import pytest
import numpy as np
from pathlib import Path
from osgeo import gdal, osr
from palmdef_risk.io.helpers import verify_alignment


def _write_tiled(path, arr, dtype, nodata, block=16):
    """Write a TILED GeoTIFF with a small block size so the streaming mask
    application is exercised across many partial 2D blocks."""
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
    band = ds.GetRasterBand(1)
    band.WriteArray(arr)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    ds.FlushCache()
    ds = None
    return path


def _ref_with_footprint(path):
    """70x70 Byte reference (NoData=255): edges + interior block + scattered."""
    ref = np.ones((70, 70), dtype=np.uint8)
    ref[0, :] = 255
    ref[-1, :] = 255
    ref[:, 0] = 255
    ref[20:30, 40:55] = 255           # interior rectangle (spans tile boundaries)
    ref[5::13, 7::11] = 255           # scattered single pixels
    _write_tiled(path, ref, gdal.GDT_Byte, 255)
    return ref


def _read(path):
    ds = gdal.Open(str(path))
    arr = ds.GetRasterBand(1).ReadAsArray()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    return arr, nd


def test_get_mask_properties_no_full_array_load(tmp_path):
    """get_mask_properties exposes ref_path and no longer materialises the
    full-reference invalid_mask boolean (the align-stage OOM source)."""
    from palmdef_risk.io.helpers import get_mask_properties
    ref = tmp_path / "ref.tif"
    _ref_with_footprint(ref)
    props = get_mask_properties(str(ref))
    assert props["ref_path"] == str(ref)
    assert "invalid_mask" not in props


def test_apply_mask_streaming_matches_full_array_byte(tmp_path):
    """Windowed apply_mask is byte-identical to arr[ref==nodata]=nodata."""
    from palmdef_risk.io.helpers import apply_mask
    ref_arr = _ref_with_footprint(tmp_path / "ref.tif")

    content = (np.arange(70 * 70).reshape(70, 70) % 200).astype(np.uint8)
    tgt = _write_tiled(tmp_path / "tgt.tif", content, gdal.GDT_Byte, 255)

    apply_mask(str(tgt), str(tmp_path / "ref.tif"))

    expected = content.copy()
    expected[ref_arr == 255] = 255
    got, nd = _read(tgt)
    assert np.array_equal(got, expected)
    assert nd == 255


def test_apply_mask_float_streaming_matches_full_array(tmp_path):
    """Windowed apply_mask_float is bit-identical to the full-array float path."""
    from palmdef_risk.io.helpers import apply_mask_float
    ref_arr = _ref_with_footprint(tmp_path / "ref.tif")

    content = (np.arange(70 * 70).reshape(70, 70) * 0.5 - 100.0).astype(np.float32)
    tgt = _write_tiled(tmp_path / "tgt.tif", content, gdal.GDT_Float32, -9999.0)

    apply_mask_float(str(tgt), str(tmp_path / "ref.tif"))

    expected = content.astype(np.float32).copy()
    expected[ref_arr == 255] = -9999.0
    got, nd = _read(tgt)
    assert np.array_equal(got, expected)
    assert nd == -9999.0


def test_rasterize_vector_produces_aligned_raster(tiny_raster, tiny_vector, tmp_path):
    from palmdef_risk.io.helpers import get_mask_properties, rasterize_vector
    mask_props = get_mask_properties(str(tiny_raster))
    out = tmp_path / "burned.tif"
    ok = rasterize_vector(str(tiny_vector), str(out), burn_value=1, mask_props=mask_props)
    assert ok
    assert out.exists()
    assert verify_alignment(str(out), str(tiny_raster))


def test_merge_plantation_raster(tmp_path):
    from palmdef_risk.process.align import merge_plantation
    import numpy as np
    from osgeo import gdal
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:4, 2:4] = 1   # industrial
    arr[6:8, 6:8] = 2   # smallholder
    from tests.conftest import _write_raster
    src = tmp_path / "plantation_raw.tif"
    _write_raster(src, arr, gt=[500000, 30, 0, 9000000, 0, -30],
                  epsg=32750, dtype=gdal.GDT_Byte, nodata=255)
    out = tmp_path / "plantation.tif"
    merge_plantation(str(src), str(out), industrial_value=1, smallholder_value=2)
    ds = gdal.Open(str(out))
    result = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    assert result[3, 3] == 1
    assert result[7, 7] == 1
    assert result[0, 9] == 0


def _write_hgu_gpkg(path, epsg=32750):
    """Single polygon covering pixels 10–20 in a 30×30 raster at 100m pixel size."""
    from osgeo import ogr, osr
    driver = ogr.GetDriverByName("GPKG")
    if path.exists():
        driver.DeleteDataSource(str(path))
    ds = driver.CreateDataSource(str(path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    layer = ds.CreateLayer("hgu", srs, ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for pt in [(501000, 9001000), (502000, 9001000), (502000, 9002000),
               (501000, 9002000), (501000, 9001000)]:
        ring.AddPoint(*pt)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(poly)
    layer.CreateFeature(feat)
    ds = None
    return path


def test_discover_inputs_finds_plantation_in_variables(tmp_path):
    """Downloaded plantation lands in variables/; user-supplied in user_inputs/.

    Covers the t3 alignment path: _discover_inputs must surface plantation_t3
    from variables/ (download source) so align_all can produce plantation_t3.tif.
    """
    from types import SimpleNamespace
    from palmdef_risk.process.align import _discover_inputs
    raw = tmp_path / "raw"
    (raw / "user_inputs").mkdir(parents=True)
    (raw / "variables").mkdir(parents=True)
    (raw / "user_inputs" / "plantation_t2.tif").write_bytes(b"x")
    (raw / "variables" / "plantation_t3.tif").write_bytes(b"x")
    d = _discover_inputs(SimpleNamespace(raw_dir=raw))
    assert d["plantation_t2"] == raw / "user_inputs" / "plantation_t2.tif"
    assert d["plantation_t3"] == raw / "variables" / "plantation_t3.tif"


def test_protected_filename_constant():
    from palmdef_risk.process.align import _PROTECTED_FILENAME
    assert _PROTECTED_FILENAME == "protected"


def test_hgu_signed_distance_negative_inside(tmp_path, write_raster):
    from palmdef_risk.process.align import compute_hgu_signed_distance
    import numpy as np
    from osgeo import gdal
    from pathlib import Path
    ref_arr = np.ones((30, 30), dtype=np.uint8)
    ref = write_raster(tmp_path / "ref.tif", ref_arr,
                       gt=[500000, 100, 0, 9003000, 0, -100], epsg=32750)
    hgu = _write_hgu_gpkg(tmp_path / "hgu.gpkg")
    out = tmp_path / "hgu_signed_dist.tif"
    compute_hgu_signed_distance(str(hgu), str(ref), str(out))
    ds = gdal.Open(str(out))
    arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
    ds = None
    # Pixel (15,15): raster origin=9003000, pixel size=-100
    # Row 15 → y = 9003000 - 15*100 = 9001500; Col 15 → x = 500000 + 15*100 = 501500
    # (501500, 9001500) is inside polygon (501000-502000, 9001000-9002000) → negative
    assert arr[15, 15] < 0, f"Expected negative inside polygon, got {arr[15,15]}"
    # Pixel (0, 0): y=9003000, x=500000 → outside → positive
    assert arr[0, 0] > 0, f"Expected positive outside polygon, got {arr[0,0]}"
