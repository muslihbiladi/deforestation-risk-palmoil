"""Focused tests for the shared GEE helpers in data/_ee_utils.py.

These pin the behaviour that was previously duplicated (and diverged) between
forest.py and variables.py — especially the unified, CRS-aware _clip_to_vector,
whose cutlineSRS handling came from variables.py and is load-bearing when the
cutline vector and the raster are in different CRS.
"""
import numpy as np
from osgeo import gdal, ogr, osr

from palmdef_risk.data._ee_utils import (
    _parse_aoi, _snap_extent, _make_grid, _clip_to_vector,
)


def test_parse_aoi_tuple_with_buffer():
    assert _parse_aoi((100.0, -5.0, 105.0, 0.0), buff=0.5) == (99.5, -5.5, 105.5, 0.5)


def test_snap_extent_expands_to_grid():
    scale = 0.1
    xmin, ymin, xmax, ymax = _snap_extent((100.04, -4.96, 104.93, -0.02), scale)
    # floor on mins, ceil on maxs
    assert xmin <= 100.04 < xmin + scale
    assert ymin <= -4.96 < ymin + scale
    assert xmax - scale < 104.93 <= xmax
    assert ymax - scale < -0.02 <= ymax


def test_make_grid_covers_extent():
    scale = 0.01
    extent = _snap_extent((100.0, -1.0, 100.5, -0.5), scale)
    tiles = _make_grid(extent, tile_size=0.2, scale=scale)
    assert tiles
    # union of tiles spans the full extent
    assert min(t[0] for t in tiles) == extent[0]
    assert min(t[1] for t in tiles) == extent[1]
    assert max(t[2] for t in tiles) == extent[2]
    assert max(t[3] for t in tiles) == extent[3]


def _write_utm_raster(path, fill=1):
    """30x30 Byte raster, UTM 50S (EPSG:32750), 30 m pixels, all == fill."""
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), 30, 30, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([500000, 30, 0, 9000900, 0, -30])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32750)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.full((30, 30), fill, dtype=np.uint8))
    ds.FlushCache()
    ds = None
    return path


def _write_cutline_4326(path):
    """Interior block (UTM cols/rows 10..20) written as an EPSG:4326 polygon.

    The polygon is built in UTM then transformed to 4326 and stored with a 4326
    SRS, so clipping a UTM raster with it only works if _clip_to_vector sets
    cutlineSRS (i.e. reprojects the cutline back to the raster CRS).
    """
    utm = osr.SpatialReference(); utm.ImportFromEPSG(32750)
    utm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    wgs = osr.SpatialReference(); wgs.ImportFromEPSG(4326)
    wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(utm, wgs)

    # cols 10..20 -> x 500300..500600 ; rows 10..20 -> y 9000300..9000600
    corners_utm = [(500300, 9000300), (500600, 9000300),
                   (500600, 9000600), (500300, 9000600), (500300, 9000300)]
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in corners_utm:
        lon, lat, _ = ct.TransformPoint(x, y)
        ring.AddPoint(lon, lat)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    drv = ogr.GetDriverByName("GPKG")
    if path.exists():
        drv.DeleteDataSource(str(path))
    ds = drv.CreateDataSource(str(path))
    layer = ds.CreateLayer("cut", wgs, ogr.wkbPolygon)
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(poly)
    layer.CreateFeature(feat)
    ds = None
    return path


def test_clip_to_vector_reprojects_cutline_crs(tmp_path):
    """Clipping a UTM raster with a 4326 cutline must succeed via cutlineSRS."""
    ras = _write_utm_raster(tmp_path / "ras.tif")
    cut = _write_cutline_4326(tmp_path / "cut.gpkg")
    out = tmp_path / "clipped.tif"

    _clip_to_vector(str(ras), str(out), str(cut), nodata=255)

    assert out.exists()
    ds = gdal.Open(str(out))
    arr = ds.GetRasterBand(1).ReadAsArray()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    assert nd == 255
    # The cutline reprojected correctly: interior keeps data (1), and the output
    # is cropped to roughly the interior block (not the full 30x30 raster).
    assert (arr == 1).any(), "interior pixels should survive the clip"
    assert arr.shape[0] < 30 and arr.shape[1] < 30, "output should be cropped to cutline"
