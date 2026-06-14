"""forest_downloader: Retrieve forest cover change data from Google Earth Engine.

This module provides functions to download forest cover change (FCC) maps
from Google Earth Engine using two global products:
- Tropical Moist Forests (TMF) by JRC
- Global Forest Change (GFC) by Hansen/UMD

Based on the geefcc Python package by Ghislain Vieilledent (Cirad).
Simplified into a single file with user-defined AOI support.

Usage:
    import ee
    import forest_downloader as fd

    ee.Initialize(project="your-project",
                  opt_url="https://earthengine-highvolume.googleapis.com")

    # Using extent (xmin, ymin, xmax, ymax)
    fd.get_fcc(
        aoi=(100.0, -5.0, 105.0, 0.0),
        buff=0.1,
        years=[2001, 2010, 2020],
        source="tmf",
        output_file="fcc_tmf.tif",
    )

    # Using a vector file
    fd.get_fcc(
        aoi="path/to/aoi.gpkg",
        buff=0.1,
        years=[2001, 2010, 2020],
        source="gfc",
        perc=75,
        output_file="fcc_gfc.tif",
    )
"""

import os
import io
import math
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
from osgeo import gdal, ogr, osr

# Suppress GDAL warnings
gdal.UseExceptions()

# ============================================================
# Constants
# ============================================================

# ~30m resolution in degrees (1/3600 degree)
SCALE = 0.000277777777778

# Maximum pixels per request for computePixels (~48MB limit)
# For 3 bands of uint8: 48MB / 3 ≈ 16M pixels → ~4000x4000
MAX_TILE_PIXELS = 12_000_000  # conservative: ~3500x3500


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
        raise FileNotFoundError(
            f"Cannot open vector file: {vector_path}"
        )
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

    # Apply buffer
    xmin -= buff
    ymin -= buff
    xmax += buff
    ymax += buff

    return (xmin, ymin, xmax, ymax)


# ============================================================
# Grid snapping and tiling
# ============================================================

def _snap_extent(extent, scale=SCALE):
    """Snap extent to the global pixel grid.

    This ensures all tiles share the same pixel origin, preventing
    misalignment during mosaicking.

    :param extent: Tuple (xmin, ymin, xmax, ymax).
    :param scale: Pixel size in degrees.
    :return: Snapped (xmin, ymin, xmax, ymax).
    """
    xmin, ymin, xmax, ymax = extent
    xmin_s = math.floor(xmin / scale) * scale
    ymin_s = math.floor(ymin / scale) * scale
    xmax_s = math.ceil(xmax / scale) * scale
    ymax_s = math.ceil(ymax / scale) * scale
    return (xmin_s, ymin_s, xmax_s, ymax_s)


def _make_grid(extent, tile_size=0.5, scale=SCALE):
    """Create a grid of tile extents, all snapped to the pixel grid.

    :param extent: Tuple (xmin, ymin, xmax, ymax) - already snapped.
    :param tile_size: Approximate tile size in degrees.
    :param scale: Pixel size in degrees.
    :return: List of (xmin, ymin, xmax, ymax) tuples.
    """
    xmin, ymin, xmax, ymax = extent

    # Snap tile_size to whole number of pixels
    tile_pixels = round(tile_size / scale)
    tile_size_snapped = tile_pixels * scale

    tiles = []
    y = ymin
    while y < ymax:
        x = xmin
        y_top = min(y + tile_size_snapped, ymax)
        while x < xmax:
            x_right = min(x + tile_size_snapped, xmax)
            tiles.append((x, y, x_right, y_top))
            x = x_right
        y = y_top

    return tiles


def _save_grid_to_gpkg(tiles, output_file):
    """Save tile grid as GeoPackage for reference/debugging.

    :param tiles: List of (xmin, ymin, xmax, ymax) tuples.
    :param output_file: Path for the output GPKG.
    """
    driver = ogr.GetDriverByName("GPKG")
    if os.path.exists(output_file):
        driver.DeleteDataSource(output_file)
    ds = driver.CreateDataSource(output_file)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer("grid", srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("tile_id", ogr.OFTInteger))

    for idx, (xmin, ymin, xmax, ymax) in enumerate(tiles):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xmin, ymin)
        ring.AddPoint(xmax, ymin)
        ring.AddPoint(xmax, ymax)
        ring.AddPoint(xmin, ymax)
        ring.AddPoint(xmin, ymin)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("tile_id", idx)
        feat.SetGeometry(poly)
        layer.CreateFeature(feat)
        feat = None

    ds = None


# ============================================================
# EE image builders
# ============================================================

def ee_tmf(years):
    """Build a multi-band ee.Image of binary forest cover from TMF.

    Uses JRC Tropical Moist Forests v1_2025 AnnualChanges dataset.
    Classes 1 (undisturbed) and 2 (degraded) are treated as forest.

    :param years: List of years (2000-2025).
    :return: ee.Image with one band per year (uint8: 1=forest, 0=non-forest).
    """
    tmf = ee.ImageCollection("projects/JRC/TMF/v1_2025/AnnualChanges")

    bands = []
    for i, year in enumerate(years):
        band_name = f"Dec{year - 1}"
        ap = tmf.select(band_name).mosaic()
        # Reclassify: 1 (undisturbed) and 2 (degraded) → forest (1)
        ap_forest = ap.where(ap.eq(2), 1)
        # Everything else → non-forest (0)
        forest = ap_forest.where(ap_forest.neq(1), 0)
        bands.append(forest.rename(f"forest_t{i + 1}"))

    return ee.Image.cat(bands).toUint8()


def ee_gfc(years, perc=75):
    """Build a multi-band ee.Image of binary forest cover from GFC.

    Uses Hansen/UMD Global Forest Change dataset.
    Forest is defined as tree cover >= perc% in 2000, minus loss years.

    :param years: List of years (2001-2025).
    :param perc: Tree cover percentage threshold (default: 75).
    :return: ee.Image with one band per year (uint8: 1=forest, 0=non-forest).
    """
    gfc = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    treecover = gfc.select(["treecover2000"])
    lossyear = gfc.select(["lossyear"])

    # Forest in year 2000 based on threshold
    forest2000 = treecover.gte(perc)

    bands = []
    for i, year in enumerate(years):
        if year == 2001:
            # 2001 = baseline (forest2000, no loss subtracted)
            forest_yr = forest2000
        elif year == 2002:
            # 2002 = subtract loss that happened in 2001
            loss = lossyear.eq(1)
            forest_yr = forest2000.where(loss.eq(1), 0)
        else:
            # Other years = subtract cumulative loss up to year-1
            index = year - 2001
            loss = lossyear.gte(1).And(lossyear.lte(index))
            forest_yr = forest2000.where(loss.eq(1), 0)

        bands.append(forest_yr.rename(f"forest_t{i + 1}"))

    return ee.Image.cat(bands).toUint8()


# ============================================================
# Tile downloading with computePixels
# ============================================================

def _download_tile(args):
    """Download one tile using ee.data.computePixels.

    This method requests an exact pixel grid from EE, ensuring
    perfect alignment between adjacent tiles.

    :param args: Tuple of (tile_extent, forest_img, tile_index,
        output_dir, verbose, n_bands, max_retries).
    :return: Path to the output tile file, or None on failure.
    """
    (tile_extent, forest_img, tile_index, output_dir,
     verbose, n_bands, max_retries) = args
    xmin, ymin, xmax, ymax = tile_extent
    tile_file = os.path.join(output_dir, f"tile_{tile_index:04d}.tif")

    # Calculate exact pixel dimensions
    n_cols = round((xmax - xmin) / SCALE)
    n_rows = round((ymax - ymin) / SCALE)

    if n_cols == 0 or n_rows == 0:
        if verbose:
            print(f"  Tile {tile_index} SKIPPED: zero dimension")
        return None

    # Build the computePixels request
    request = {
        "expression": forest_img,
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {
                "width": n_cols,
                "height": n_rows,
            },
            "affineTransform": {
                "scaleX": SCALE,
                "shearX": 0,
                "translateX": xmin,
                "shearY": 0,
                "scaleY": -SCALE,
                "translateY": ymax,
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
                print(f"  Tile {tile_index} OK "
                      f"({n_cols}x{n_rows} px, {n_bands} bands)")
            return tile_file

        except Exception as e:
            if verbose:
                print(f"  Tile {tile_index} attempt {attempt}/{max_retries} "
                      f"FAILED: {e}")
            if attempt < max_retries:
                wait = min(2 ** attempt, 30)
                time.sleep(wait)

    if verbose:
        print(f"  Tile {tile_index} FAILED after {max_retries} attempts")
    return None


# ============================================================
# Mosaicking
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

    # Remove any NoData from tiles (EE may set NoData=0 which
    # conflicts with valid 0 = non-forest)
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
        "creationOptions": ["COMPRESS=DEFLATE", "TILED=YES"],
    }

    if crop_extent is not None:
        xmin, ymin, xmax, ymax = crop_extent
        translate_options["projWin"] = [xmin, ymax, xmax, ymin]

    gdal.Translate(output_file, vrt_file, **translate_options)

    # Clean up VRT
    if os.path.exists(vrt_file):
        os.remove(vrt_file)


def _clip_to_vector(input_file, output_file, vector_path, buff=0.0):
    """Clip a raster to a vector polygon boundary.

    :param input_file: Path to input GeoTIFF.
    :param output_file: Path for clipped output GeoTIFF.
    :param vector_path: Path to vector file (GPKG, SHP, GeoJSON).
    :param buff: Buffer in degrees applied to the vector geometry.
    """
    ds = ogr.Open(str(vector_path))
    if ds is None:
        raise FileNotFoundError(
            f"Cannot open vector file: {vector_path}"
        )
    layer = ds.GetLayer()
    layer_name = layer.GetName()
    ds = None

    # If buffer is needed, create a buffered vector
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
    # EE's computePixels may set NoData=0, which causes GDAL Warp
    # to treat valid 0 (non-forest) pixels as NoData and discard them.
    src_ds = gdal.Open(input_file, gdal.GA_Update)
    if src_ds is not None:
        for b in range(1, src_ds.RasterCount + 1):
            src_ds.GetRasterBand(b).DeleteNoDataValue()
        src_ds.FlushCache()
        src_ds = None

    # Warp with cutline
    warp_options = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=cutline_path,
        cutlineLayer=cutline_layer,
        cropToCutline=True,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
        dstNodata=255,
    )
    gdal.Warp(output_file, input_file, options=warp_options)

    # Cleanup buffered vector
    if buff_path and os.path.exists(buff_path):
        ogr.GetDriverByName("GPKG").DeleteDataSource(buff_path)


# ============================================================
# Post-processing: export bands and sum
# ============================================================

def export_bands(input_file, output_dir=None, prefix="forest_t",
                 verbose=True):
    """Export each band of a multi-band raster as a separate file.

    :param input_file: Path to input multi-band GeoTIFF.
    :param output_dir: Directory for output files. If None, uses the
        same directory as input_file.
    :param prefix: Prefix for output filenames (default: "forest_t").
    :param verbose: Print progress messages.
    :return: List of output file paths.
    """
    ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {input_file}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_file))
    os.makedirs(output_dir, exist_ok=True)

    n_bands = ds.RasterCount

    # Stream each band out with gdal.Translate(bandList=[b]) — GDAL copies via
    # internal block I/O, so no full band is ever materialised as a numpy array
    # (the prior ReadAsArray per band held a full ~1.8 GB Byte band for an
    # Indonesia-scale raster). Pixel values and the nodata value are preserved,
    # so the exported band data is identical to the prior path.
    output_files = []
    for b in range(1, n_bands + 1):
        nd = ds.GetRasterBand(b).GetNoDataValue()
        nodata_val = int(nd) if nd is not None else 255
        out_path = os.path.join(output_dir, f"{prefix}{b}.tif")

        gdal.Translate(
            out_path, ds,
            options=gdal.TranslateOptions(
                format="GTiff",
                bandList=[b],
                outputType=gdal.GDT_Byte,
                noData=nodata_val,
                creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
            ),
        )

        output_files.append(out_path)
        if verbose:
            print(f"  Band {b} exported: {out_path}")

    ds = None

    if verbose:
        print(f"Exported {n_bands} bands from {input_file}")

    return output_files


def export_period_fcc(input_file, output_dir=None, verbose=True):
    """Export period-specific deforestation rasters from a multi-band forest file.

    For a 3-band raster with years=[t1, t2, t3], this creates:
        fcc12.tif  →  deforestation between t1 and t2
        fcc23.tif  →  deforestation between t2 and t3

    Pixel values:
        1   = remained forest (forest at both ti and tj)
        0   = deforested during this period (forest at ti, non-forest at tj)
        255 = NoData: outside AOI, or not forest at ti (not in analysis domain)

    Logic: fcc_ij = band_i AND band_j (bitwise). A pixel is 1 only
    if it was forest at both the start and end of the period. Pixels that
    were not forest at ti are masked as NoData — they are not "deforested",
    they were simply never in the analysis domain for this period.

    :param input_file: Path to multi-band forest GeoTIFF (from get_fcc).
    :param output_dir: Directory for output files. If None, uses the
        same directory as input_file.
    :param verbose: Print progress messages.
    :return: List of output file paths.
    """
    ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {input_file}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_file))
    os.makedirs(output_dir, exist_ok=True)

    n_bands = ds.RasterCount
    n_cols = ds.RasterXSize
    n_rows = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    if n_bands < 2:
        ds = None
        raise ValueError("Need at least 2 bands to compute period FCC.")

    # Stream over GDAL block windows: the nodata mask ORs every band, so all
    # bands are needed per pixel — but only one *block* of each band is held at
    # a time, not the full bands list + full nodata mask (the prior path kept
    # all n_bands full arrays resident, ~1.8 GB for Indonesia). Every operation
    # is elementwise, so per-window output is identical to the whole-array path.
    src_bands = [ds.GetRasterBand(b) for b in range(1, n_bands + 1)]
    nds = [b.GetNoDataValue() for b in src_bands]
    bx, by = src_bands[0].GetBlockSize()

    driver = gdal.GetDriverByName("GTiff")
    out_dss = []
    output_files = []
    for i in range(n_bands - 1):
        out_path = os.path.join(output_dir, f"fcc{i + 1}{i + 2}.tif")
        out_ds = driver.Create(
            out_path, n_cols, n_rows, 1, gdal.GDT_Byte,
            options=["COMPRESS=DEFLATE", "TILED=YES"],
        )
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        out_ds.GetRasterBand(1).SetNoDataValue(255)
        out_dss.append(out_ds)
        output_files.append(out_path)

    for yoff in range(0, n_rows, by):
        ywin = min(by, n_rows - yoff)
        for xoff in range(0, n_cols, bx):
            xwin = min(bx, n_cols - xoff)
            blocks = [b.ReadAsArray(xoff, yoff, xwin, ywin) for b in src_bands]
            nodata_mask = np.zeros((ywin, xwin), dtype=bool)
            for data, nd in zip(blocks, nds):
                if nd is not None:
                    nodata_mask |= (data == int(nd))
            for i in range(n_bands - 1):
                # fcc_ij: 1 if forest at both ti and tj, 0 if deforested.
                # Pixels not forest at ti are outside the analysis domain → NoData.
                fcc = (blocks[i].astype(np.uint8) & blocks[i + 1].astype(np.uint8))
                fcc[blocks[i] == 0] = 255
                fcc[nodata_mask] = 255
                out_dss[i].GetRasterBand(1).WriteArray(fcc, xoff, yoff)

    for i, out_ds in enumerate(out_dss):
        out_ds.FlushCache()
        out_dss[i] = None
    ds = None

    if verbose:
        for i, out_path in enumerate(output_files):
            print(f"  Period {i+1}→{i+2} exported: {out_path}")
        print(f"Exported {len(output_files)} period FCC rasters")

    return output_files


def sum_raster_bands(input_file, output_file, verbose=True):
    """Sum all bands of a raster into a single band.

    For a forest cover raster with n binary bands (one per year),
    the sum encodes the deforestation trajectory:
        0 = non-forest at all dates
        1..n-1 = deforested during a specific period
        n = remaining forest at the last date

    NoData pixels (255) are preserved in the output.

    :param input_file: Path to input multi-band GeoTIFF.
    :param output_file: Path for output single-band GeoTIFF.
    :param verbose: Print progress messages.
    """
    ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {input_file}")

    n_bands = ds.RasterCount
    n_cols = ds.RasterXSize
    n_rows = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    if verbose:
        print(f"Summing {n_bands} bands from {input_file}")

    # Read all bands and build a NoData mask
    nodata_mask = np.zeros((n_rows, n_cols), dtype=bool)
    band_sum = np.zeros((n_rows, n_cols), dtype=np.uint8)

    for b in range(1, n_bands + 1):
        band_data = ds.GetRasterBand(b).ReadAsArray()
        nd = ds.GetRasterBand(b).GetNoDataValue()
        if nd is not None:
            nodata_mask |= (band_data == int(nd))
        # Only sum valid pixels (treat nodata as 0 in sum)
        valid = band_data.copy()
        if nd is not None:
            valid[band_data == int(nd)] = 0
        band_sum += valid.astype(np.uint8)

    ds = None

    # Set NoData pixels to 255 in the output
    band_sum[nodata_mask] = 255

    # Write output
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_file, n_cols, n_rows, 1, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(band_sum)
    out_ds.GetRasterBand(1).SetNoDataValue(255)
    out_ds.FlushCache()
    out_ds = None

    if verbose:
        print(f"Output written to {output_file}")


# ============================================================
# Reprojection
# ============================================================

def reproject_raster(input_file, output_file, dst_crs="EPSG:32750",
                     resampling="near", verbose=True):
    """Reproject a raster to a different coordinate reference system.

    :param input_file: Path to input GeoTIFF.
    :param output_file: Path for output reprojected GeoTIFF.
    :param dst_crs: Target CRS as EPSG string (e.g. "EPSG:32750")
        or WKT. Default: EPSG:32750 (UTM 50S).
    :param resampling: Resampling method. Use "near" (default) for
        categorical data (forest/non-forest, FCC), "bilinear" for
        continuous data. Options: "near", "bilinear", "cubic",
        "average", "mode".
    :param verbose: Print progress messages.
    :return: Path to the output file.
    """
    resample_map = {
        "near": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "average": gdal.GRA_Average,
        "mode": gdal.GRA_Mode,
    }

    if resampling not in resample_map:
        raise ValueError(
            f"Unknown resampling: {resampling}. "
            f"Options: {list(resample_map.keys())}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Read source nodata
    src_ds = gdal.Open(input_file)
    if src_ds is None:
        raise FileNotFoundError(f"Cannot open raster: {input_file}")
    src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()
    src_ds = None

    # Forest cover data uses 255 as NoData; fall back to it if source has none
    if src_nodata is None:
        src_nodata = 255

    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=dst_crs,
        resampleAlg=resample_map[resampling],
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
        srcNodata=src_nodata,
        dstNodata=src_nodata,
    )

    gdal.Warp(output_file, input_file, options=warp_options)

    if verbose:
        print(f"Reprojected: {input_file} → {output_file} [{dst_crs}]")

    return output_file


def reproject_all(file_list, output_dir=None, dst_crs="EPSG:32750",
                  suffix="_proj", resampling="near", verbose=True):
    """Reproject multiple rasters to a different CRS.

    :param file_list: List of input GeoTIFF paths.
    :param output_dir: Output directory. If None, saves alongside
        originals with suffix appended.
    :param dst_crs: Target CRS (default: "EPSG:32750").
    :param suffix: Suffix added to filenames (default: "_proj").
    :param resampling: Resampling method (default: "near").
    :param verbose: Print progress messages.
    :return: List of output file paths.
    """
    output_files = []
    for f in file_list:
        if f is None or not os.path.exists(f):
            continue
        base, ext = os.path.splitext(os.path.basename(f))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{base}{suffix}{ext}")
        else:
            out_path = os.path.join(
                os.path.dirname(f), f"{base}{suffix}{ext}"
            )
        reproject_raster(f, out_path, dst_crs=dst_crs,
                         resampling=resampling, verbose=verbose)
        output_files.append(out_path)

    return output_files


def _set_nodata(file_path, nodata=255):
    """Set NoData value on all bands of a raster in-place."""
    ds = gdal.Open(file_path, gdal.GA_Update)
    if ds is not None:
        for b in range(1, ds.RasterCount + 1):
            ds.GetRasterBand(b).SetNoDataValue(nodata)
        ds.FlushCache()
        ds = None


# ============================================================
# Main function: get_fcc
# ============================================================

def get_fcc(
    aoi,
    years,
    source="tmf",
    buff=0.0,
    perc=75,
    tile_size=None,
    crop_to_aoi=True,
    output_file="forest_cover.tif",
    parallel=True,
    max_retries=3,
    export_individual=True,
    export_fcc=True,
    export_period=True,
    output_crs=None,
    verbose=True,
):
    """Download forest cover change data from Google Earth Engine.

    :param aoi: Area of interest. Either:
        - A tuple (xmin, ymin, xmax, ymax) in EPSG:4326.
        - A path to a vector file (GPKG, SHP, GeoJSON).
    :param years: List of years defining the time periods.
        E.g. [2015, 2020, 2025] produces a 3-band raster.
    :param source: Data source, either "tmf" or "gfc".
    :param buff: Buffer in degrees around the AOI (default: 0.0).
    :param perc: Tree cover threshold for GFC (default: 75).
        Only used when source="gfc".
    :param tile_size: Tile size in degrees for downloads. If None
        (default), auto-calculated based on number of bands to stay
        within the computePixels 48MB limit.
    :param crop_to_aoi: Whether to crop the output to the buffered AOI
        (default: True).
    :param output_file: Path for the multi-band output GeoTIFF
        (default: "forest_cover.tif").
    :param parallel: Use multithreading for tile downloads
        (default: True).
    :param max_retries: Max retry attempts per tile (default: 3).
    :param export_individual: Export each band as a separate
        single-band file (forest_t1.tif, forest_t2.tif, etc.)
        (default: True).
    :param export_fcc: Export the FCC trajectory raster from
        sum_raster_bands (fcc123.tif) (default: True).
    :param export_period: Export period-specific deforestation
        rasters (fcc12.tif, fcc23.tif, etc.) (default: True).
    :param output_crs: Reproject all outputs to this CRS. If None
        (default), outputs stay in EPSG:4326. Example: "EPSG:32750"
        for UTM zone 50S.
    :param verbose: Print progress messages (default: True).
    :return: Dictionary with output file paths:
        - "forest_cover": multi-band file (all years)
        - "forest_bands": list of individual band files
        - "fcc": FCC trajectory file
        - "fcc_periods": list of period deforestation files
    """

    # ---- Parse AOI ----
    extent = _parse_aoi(aoi, buff)
    if verbose:
        print(f"AOI extent (with buffer): {extent}")

    # ---- Validate years ----
    if source == "tmf":
        for y in years:
            if y < 2000 or y > 2025:
                raise ValueError(
                    f"TMF year {y} out of range. Valid: 2000-2025."
                )
    elif source == "gfc":
        for y in years:
            if y < 2001 or y > 2025:
                raise ValueError(
                    f"GFC year {y} out of range. Valid: 2001-2025."
                )
    else:
        raise ValueError(f"Unknown source: {source}. Use 'tmf' or 'gfc'.")

    n_bands = len(years)

    # ---- Build forest cover Image ----
    if verbose:
        print(f"Building forest cover from {source.upper()} "
              f"for years {years}")

    if source == "tmf":
        forest_img = ee_tmf(years)
    else:
        forest_img = ee_gfc(years, perc=perc)

    # ---- Auto-calculate safe tile size ----
    # computePixels limit is ~48MB. EE computes internally at higher
    # precision (~6 bytes/pixel/band), so we use a conservative estimate.
    COMPUTE_LIMIT = 50_331_648
    BYTES_PER_PIXEL = 6  # EE internal precision overhead
    safe_pixels = COMPUTE_LIMIT / (n_bands * BYTES_PER_PIXEL)
    safe_side = int(math.sqrt(safe_pixels))
    max_tile_deg = round(safe_side * SCALE, 4)

    if tile_size is None:
        tile_size = max_tile_deg
        if verbose:
            print(f"Auto tile size: {tile_size:.4f}° "
                  f"({safe_side}x{safe_side} px) for {n_bands} bands")
    elif tile_size > max_tile_deg:
        if verbose:
            print(f"WARNING: tile_size={tile_size}° too large for "
                  f"{n_bands} bands. Reducing to {max_tile_deg:.4f}°")
        tile_size = max_tile_deg

    # ---- Snap extent and create tile grid ----
    snapped_extent = _snap_extent(extent)
    tiles = _make_grid(snapped_extent, tile_size=tile_size)
    if verbose:
        print(f"Number of tiles: {len(tiles)}")

    # ---- Prepare output directory ----
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)

    # Save grid for reference
    grid_file = os.path.join(output_dir, "grid.gpkg")
    _save_grid_to_gpkg(tiles, grid_file)

    # Temp directory for tiles
    tile_dir = os.path.join(output_dir, "tiles_tmp")
    os.makedirs(tile_dir, exist_ok=True)

    # ---- Download tiles ----
    download_args = [
        (tile, forest_img, idx, tile_dir, verbose, n_bands, max_retries)
        for idx, tile in enumerate(tiles)
    ]

    if parallel and len(tiles) > 1:
        ncpu = min(len(tiles), max(1, multiprocessing.cpu_count() - 1), 10)
        if verbose:
            print(f"Downloading {len(tiles)} tiles in parallel "
                  f"({ncpu} threads)...")
        tile_files = [None] * len(tiles)
        with ThreadPoolExecutor(max_workers=ncpu) as executor:
            future_to_idx = {
                executor.submit(_download_tile, args): args[2]
                for args in download_args
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                tile_files[idx] = future.result()
    else:
        if verbose:
            print("Downloading tiles sequentially...")
        tile_files = [_download_tile(args) for args in download_args]

    # ---- Check results ----
    n_ok = sum(1 for f in tile_files if f is not None)
    n_fail = len(tiles) - n_ok
    if verbose:
        print(f"Download complete: {n_ok}/{len(tiles)} tiles OK"
              + (f", {n_fail} failed" if n_fail > 0 else ""))

    # ---- Mosaic tiles ----
    if verbose:
        print("Mosaicking tiles...")

    # First mosaic to a temporary file (full extent)
    aoi_is_vector = isinstance(aoi, (str, Path)) and os.path.isfile(str(aoi))

    if crop_to_aoi and aoi_is_vector:
        # Mosaic without cropping first, then clip to vector
        mosaic_tmp = output_file.replace(".tif", "_mosaic_tmp.tif")
        _mosaic_tiles(tile_files, mosaic_tmp, crop_extent=None)

        if verbose:
            print("Clipping to AOI vector boundary...")
        _clip_to_vector(mosaic_tmp, output_file, str(aoi), buff=0.0)

        # Cleanup temp mosaic
        if os.path.exists(mosaic_tmp):
            os.remove(mosaic_tmp)
    else:
        # Crop to rectangular extent only
        crop_extent = _snap_extent(extent) if crop_to_aoi else None
        _mosaic_tiles(tile_files, output_file, crop_extent=crop_extent)

        # Set NoData=255 on the mosaicked output so reprojection
        # fills outside-extent areas with 255 instead of 0
        _set_nodata(output_file, nodata=255)

    # ---- Cleanup tile files ----
    for f in tile_files:
        if f is not None and os.path.exists(f):
            os.remove(f)
    if os.path.exists(tile_dir) and not os.listdir(tile_dir):
        os.rmdir(tile_dir)

    # ---- Build result dictionary ----
    result = {"forest_cover": output_file}

    # ---- Export individual bands ----
    if export_individual:
        if verbose:
            print("Exporting individual forest bands...")
        band_files = export_bands(
            input_file=output_file,
            output_dir=output_dir,
            prefix="forest_t",
            verbose=verbose,
        )
        result["forest_bands"] = band_files

    # ---- Export FCC trajectory raster ----
    if export_fcc:
        if verbose:
            print("Computing FCC trajectory raster...")
        indices = "".join(str(i + 1) for i in range(len(years)))
        fcc_file = os.path.join(output_dir, f"fcc{indices}.tif")
        sum_raster_bands(
            input_file=output_file,
            output_file=fcc_file,
            verbose=verbose,
        )
        result["fcc"] = fcc_file

    # ---- Export period FCC rasters ----
    if export_period and len(years) >= 2:
        if verbose:
            print("Exporting period deforestation rasters...")
        period_files = export_period_fcc(
            input_file=output_file,
            output_dir=output_dir,
            verbose=verbose,
        )
        result["fcc_periods"] = period_files

    # ---- Reproject all outputs ----
    if output_crs is not None:
        if verbose:
            print(f"Reprojecting all outputs to {output_crs}...")

        # Collect all files to reproject
        all_files = [output_file]
        if "forest_bands" in result:
            all_files.extend(result["forest_bands"])
        if "fcc" in result:
            all_files.append(result["fcc"])
        if "fcc_periods" in result:
            all_files.extend(result["fcc_periods"])

        # Reproject in-place (overwrite originals)
        for f in all_files:
            if f and os.path.exists(f):
                tmp = f.replace(".tif", "_4326.tif")
                os.rename(f, tmp)
                reproject_raster(
                    tmp, f, dst_crs=output_crs,
                    resampling="near", verbose=verbose,
                )
                os.remove(tmp)

    if verbose:
        print("=" * 60)
        print("All outputs:")
        crs_label = f" [{output_crs}]" if output_crs else " [EPSG:4326]"
        print(f"  Projection              : {crs_label.strip(' []')}")
        print(f"  Multi-band forest cover : {output_file}")
        if export_individual:
            for bf in result["forest_bands"]:
                print(f"  Individual band         : {bf}")
        if export_fcc:
            print(f"  FCC trajectory          : {result['fcc']}")
        if export_period and "fcc_periods" in result:
            for pf in result["fcc_periods"]:
                print(f"  Period deforestation    : {pf}")
        print("=" * 60)
        print("Done!")

    return result


# ── RunContext-aware entry point ──────────────────────────────

from palmdef_risk.io.run import RunContext


def download_forest(ctx: RunContext, use_cache: bool = True) -> dict:
    """Download forest cover change data for this run.

    Reads all parameters from ctx.config. Writes outputs to
    ctx.raw_dir/forest/. Returns the same dict as get_fcc().
    """
    import json
    import shutil
    from palmdef_risk.cache import CacheManager
    from palmdef_risk.io.helpers import aoi_bbox_4326

    cfg = ctx.config
    out_dir = ctx.raw_dir / "forest"

    _bbox = aoi_bbox_4326(cfg.aoi_source)
    _cm = CacheManager(cfg.cache_dir)
    _fkey = _cm.forest_key(_bbox, cfg.aoi_buffer, cfg.forest_source, cfg.forest_years, cfg.forest_perc)

    if use_cache and _cm.forest_valid(_fkey, list(_bbox)):
        _cache_d = _cm.forest_dir(_fkey)
        out_dir.mkdir(parents=True, exist_ok=True)
        for _f in _cache_d.iterdir():
            if _f.name != "metadata.json":
                shutil.copy2(_f, out_dir / _f.name)
        print("Forest: loaded from cross-run cache.")
        return {"forest_cover": str(out_dir / "forest_cover.tif")}

    if use_cache and (out_dir / "fcc23.tif").exists():
        print("Forest: outputs already present in run folder, skipping.")
        return {"forest_cover": str(out_dir / "forest_cover.tif")}

    ee.Initialize(
        project=cfg.gee_project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # aoi_buffer is in metres; get_fcc expects degrees (~111 320 m/deg at equator)
    buff_deg = cfg.aoi_buffer / 111_320.0

    result = get_fcc(
        aoi=cfg.aoi_source,
        years=cfg.forest_years,
        source=cfg.forest_source,
        buff=buff_deg,
        perc=cfg.forest_perc,
        output_file=str(out_dir / "forest_cover.tif"),
        output_crs=ctx.config.crs,
        verbose=True,
    )
    _cache_d = _cm.forest_dir(_fkey)
    _cache_d.mkdir(parents=True, exist_ok=True)
    for _f in out_dir.iterdir():
        if _f.is_file():
            shutil.copy2(_f, _cache_d / _f.name)
    (_cache_d / "metadata.json").write_text(
        json.dumps({"downloaded_extent": list(_bbox)}), encoding="utf-8"
    )
    return result
