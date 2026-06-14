"""Shared Google Earth Engine download helpers.

AOI parsing, pixel-grid snapping/tiling, the computePixels tile-download kernel,
mosaicking, and vector clipping were copy-pasted between ``forest.py`` and
``variables.py``. They are unified here and imported by both (and by
``plantation.py``).

The two former copies of ``_clip_to_vector`` had diverged: ``forest.py`` carried
an optional ``buff`` (only ever called with ``buff=0.0``) and hard-coded
``dstNodata=255`` but had **no** CRS handling, while ``variables.py`` was
CRS-aware (sets ``cutlineSRS`` so GDAL reprojects the cutline on the fly) and
took a configurable ``nodata`` but had no buffer. The unified version below is
the **superset**: ``buff`` + configurable ``nodata`` + CRS-aware cutline. For
the forest clip path (raster + AOI both EPSG:4326) ``cutlineSRS`` is a no-op, so
the change is behaviour-preserving there while fixing the latent CRS bug if a
non-4326 AOI is ever passed.

The tile-download kernel is unified to the parameterized ``scale`` form (the
fixed-``SCALE`` forest copy was a special case); callers pass their own scale.
"""

import os
import math
import time
import logging
from pathlib import Path

import ee
from osgeo import gdal, ogr, osr

from palmdef_risk.constants import NODATA_BYTE, GTIFF_OPTS

logger = logging.getLogger(__name__)

# Suppress GDAL warnings (callers also call this, harmless to repeat).
gdal.UseExceptions()


# ============================================================
# AOI parsing
# ============================================================

def _get_extent_from_vector(vector_path):
    """Get bounding box from a vector file in EPSG:4326.

    :param vector_path: Path to GPKG, SHP, or GeoJSON file.
    :return: Tuple (xmin, ymin, xmax, ymax) in EPSG:4326.
    """
    ds = ogr.Open(str(vector_path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open vector file: {vector_path}")
    layer = ds.GetLayer()
    srs_src = layer.GetSpatialRef()

    # Get extent in source CRS
    xmin, xmax, ymin, ymax = layer.GetExtent()

    # Transform to EPSG:4326 if needed
    srs_4326 = osr.SpatialReference()
    srs_4326.ImportFromEPSG(4326)
    srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    if srs_src is not None:
        srs_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not srs_src.IsSame(srs_4326):
            transform = osr.CoordinateTransformation(srs_src, srs_4326)
            # Transform all four corners and take envelope
            corners = [
                (xmin, ymin), (xmin, ymax),
                (xmax, ymin), (xmax, ymax),
            ]
            xs, ys = [], []
            for cx, cy in corners:
                tx, ty, _ = transform.TransformPoint(cx, cy)
                xs.append(tx)
                ys.append(ty)
            xmin, ymin = min(xs), min(ys)
            xmax, ymax = max(xs), max(ys)

    ds = None
    return (xmin, ymin, xmax, ymax)


def _parse_aoi(aoi, buff=0.0):
    """Parse AOI into (xmin, ymin, xmax, ymax) with optional buffer.

    :param aoi: Tuple (xmin, ymin, xmax, ymax) or path to vector file.
    :param buff: Buffer in degrees.
    :return: Tuple (xmin, ymin, xmax, ymax).
    """
    if isinstance(aoi, (tuple, list)) and len(aoi) == 4:
        xmin, ymin, xmax, ymax = aoi
    elif isinstance(aoi, (str, Path)):
        xmin, ymin, xmax, ymax = _get_extent_from_vector(aoi)
    else:
        raise ValueError(
            "aoi must be a (xmin, ymin, xmax, ymax) tuple "
            "or a path to a vector file."
        )
    return (xmin - buff, ymin - buff, xmax + buff, ymax + buff)


# ============================================================
# Grid snapping and tiling
# ============================================================

def _snap_extent(extent, scale):
    """Snap extent to the global pixel grid.

    Ensures all tiles share the same pixel origin, preventing misalignment
    during mosaicking.

    :param extent: Tuple (xmin, ymin, xmax, ymax).
    :param scale: Pixel size in degrees.
    :return: Snapped (xmin, ymin, xmax, ymax).
    """
    xmin, ymin, xmax, ymax = extent
    return (
        math.floor(xmin / scale) * scale,
        math.floor(ymin / scale) * scale,
        math.ceil(xmax / scale) * scale,
        math.ceil(ymax / scale) * scale,
    )


def _make_grid(extent, tile_size, scale):
    """Create a grid of tile extents, all snapped to the pixel grid.

    :param extent: Tuple (xmin, ymin, xmax, ymax) - already snapped.
    :param tile_size: Approximate tile size in degrees.
    :param scale: Pixel size in degrees.
    :return: List of (xmin, ymin, xmax, ymax) tuples.
    """
    xmin, ymin, xmax, ymax = extent

    # Snap tile_size to a whole number of pixels
    ts = round(tile_size / scale) * scale

    tiles = []
    y = ymin
    while y < ymax:
        x = xmin
        y_top = min(y + ts, ymax)
        while x < xmax:
            x_right = min(x + ts, xmax)
            tiles.append((x, y, x_right, y_top))
            x = x_right
        y = y_top
    return tiles


# ============================================================
# Tile downloading with computePixels
# ============================================================

def _download_tile(args):
    """Download one tile using ee.data.computePixels.

    Requests an exact pixel grid from EE so adjacent tiles align perfectly.

    :param args: Tuple of (tile_extent, ee_image, tile_index, output_dir,
        scale, n_bands, max_retries, verbose).
    :return: Path to the output tile file, or None on failure.
    """
    (tile_extent, ee_image, tile_index, output_dir,
     scale, n_bands, max_retries, verbose) = args
    xmin, ymin, xmax, ymax = tile_extent
    tile_file = os.path.join(output_dir, f"tile_{tile_index:04d}.tif")

    n_cols = round((xmax - xmin) / scale)
    n_rows = round((ymax - ymin) / scale)

    if n_cols == 0 or n_rows == 0:
        if verbose:
            logger.info(f"  Tile {tile_index} SKIPPED: zero dimension")
        return None

    request = {
        "expression": ee_image,
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {"width": n_cols, "height": n_rows},
            "affineTransform": {
                "scaleX": scale, "shearX": 0, "translateX": xmin,
                "shearY": 0, "scaleY": -scale, "translateY": ymax,
            },
            "crsCode": "EPSG:4326",
        },
    }

    for attempt in range(1, max_retries + 1):
        try:
            result = ee.data.computePixels(request)

            with open(tile_file, "wb") as f:
                f.write(result)

            # Verify the file is valid
            ds = gdal.Open(tile_file)
            if ds is None:
                raise RuntimeError("GDAL cannot open downloaded tile")
            if ds.RasterCount != n_bands:
                raise RuntimeError(
                    f"Expected {n_bands} bands, got {ds.RasterCount}"
                )
            ds = None

            if verbose:
                logger.info(f"  Tile {tile_index} OK "
                      f"({n_cols}x{n_rows} px, {n_bands} bands)")
            return tile_file

        except Exception as e:
            if verbose:
                logger.warning(f"  Tile {tile_index} attempt {attempt}/{max_retries} "
                      f"FAILED: {e}")
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30))

    if verbose:
        logger.warning(f"  Tile {tile_index} FAILED after {max_retries} attempts")
    return None


# ============================================================
# Mosaicking and clipping
# ============================================================

def _mosaic_tiles(tile_files, output_file, crop_extent=None):
    """Mosaic tiles into a single GeoTIFF using GDAL VRT.

    :param tile_files: List of tile file paths (may contain None).
    :param output_file: Path for the output GeoTIFF.
    :param crop_extent: Optional (xmin, ymin, xmax, ymax) to crop.
    """
    valid_files = [f for f in tile_files if f is not None and os.path.exists(f)]
    if not valid_files:
        raise RuntimeError("No valid tiles to mosaic.")

    # Remove any NoData from tiles (EE may set NoData=0 which conflicts with
    # valid 0 = non-forest / not-built etc.)
    for f in valid_files:
        ds = gdal.Open(f, gdal.GA_Update)
        if ds is not None:
            for b in range(1, ds.RasterCount + 1):
                ds.GetRasterBand(b).DeleteNoDataValue()
            ds.FlushCache()
            ds = None

    # Build VRT
    vrt_file = output_file.replace(".tif", "_mosaic.vrt")
    vrt_options = gdal.BuildVRTOptions(resolution="highest")
    vrt_ds = gdal.BuildVRT(vrt_file, valid_files, options=vrt_options)
    vrt_ds.FlushCache()
    vrt_ds = None

    # Translate VRT to GeoTIFF (with optional crop)
    translate_options = {
        "format": "GTiff",
        "creationOptions": GTIFF_OPTS,
    }
    if crop_extent is not None:
        xmin, ymin, xmax, ymax = crop_extent
        translate_options["projWin"] = [xmin, ymax, xmax, ymin]

    gdal.Translate(output_file, vrt_file, **translate_options)

    # Clean up VRT
    if os.path.exists(vrt_file):
        os.remove(vrt_file)


def _clip_to_vector(input_file, output_file, vector_path, buff=0.0, nodata=NODATA_BYTE):
    """Clip a raster to a vector polygon boundary.

    Unified superset of the two former copies:
      - ``buff > 0`` builds a buffered cutline (from forest.py's copy).
      - ``nodata`` sets the fill value for outside-cutline pixels
        (configurable, from variables.py's copy).
      - ``cutlineSRS`` is set from the vector's own CRS so GDAL reprojects the
        cutline on the fly when the vector and raster differ (from variables.py;
        a no-op when both are EPSG:4326, as in the forest clip path).

    :param input_file: Path to input GeoTIFF.
    :param output_file: Path for clipped output GeoTIFF.
    :param vector_path: Path to vector file (GPKG, SHP, GeoJSON).
    :param buff: Buffer in degrees applied to the vector geometry.
    :param nodata: NoData fill value for pixels outside the cutline.
    """
    ds = ogr.Open(str(vector_path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open vector file: {vector_path}")
    layer = ds.GetLayer()
    layer_name = layer.GetName()
    # CRS of the cutline vector (so GDAL can reproject it if needed)
    vec_srs = layer.GetSpatialRef()
    cutline_srs = vec_srs.ExportToWkt() if vec_srs is not None else None
    ds = None

    # If a buffer is requested, build a buffered cutline vector.
    if buff > 0.0:
        ds_in = ogr.Open(str(vector_path))
        layer_in = ds_in.GetLayer()
        srs = layer_in.GetSpatialRef()

        buff_path = output_file.replace(".tif", "_buff.gpkg")
        drv = ogr.GetDriverByName("GPKG")
        if os.path.exists(buff_path):
            drv.DeleteDataSource(buff_path)
        ds_buff = drv.CreateDataSource(buff_path)
        layer_buff = ds_buff.CreateLayer("buffered", srs, ogr.wkbPolygon)

        for feat in layer_in:
            geom = feat.GetGeometryRef().Buffer(buff)
            feat_buff = ogr.Feature(layer_buff.GetLayerDefn())
            feat_buff.SetGeometry(geom)
            layer_buff.CreateFeature(feat_buff)
            feat_buff = None

        ds_buff = None
        ds_in = None
        cutline_path = buff_path
        cutline_layer = "buffered"
    else:
        cutline_path = str(vector_path)
        cutline_layer = layer_name
        buff_path = None

    # CRITICAL: Remove any NoData from source raster before warping.
    # EE's computePixels may set NoData=0, which causes GDAL Warp to treat
    # valid 0 pixels as NoData and discard them.
    src_ds = gdal.Open(input_file, gdal.GA_Update)
    if src_ds is not None:
        for b in range(1, src_ds.RasterCount + 1):
            src_ds.GetRasterBand(b).DeleteNoDataValue()
        src_ds.FlushCache()
        src_ds = None

    warp_kwargs = dict(
        format="GTiff",
        cutlineDSName=cutline_path,
        cutlineLayer=cutline_layer,
        cropToCutline=True,
        creationOptions=GTIFF_OPTS,
        dstNodata=nodata,
    )
    # Tell GDAL the CRS of the cutline so it can reproject it if needed.
    if cutline_srs is not None:
        warp_kwargs["cutlineSRS"] = cutline_srs

    gdal.Warp(output_file, input_file, options=gdal.WarpOptions(**warp_kwargs))

    # Cleanup buffered cutline
    if buff_path and os.path.exists(buff_path):
        ogr.GetDriverByName("GPKG").DeleteDataSource(buff_path)
