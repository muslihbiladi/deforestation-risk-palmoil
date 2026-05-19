"""variable_downloader: Download spatial data for deforestation modeling.

This module downloads spatial variables required by the forestatrisk
library, EXCEPT forest cover (handled by forest_downloader.py).

Outputs:
    Rasters (GeoTIFF):
        - altitude.tif     : Elevation in meters (NASA SRTM 30m via GEE)
        - slope.tif        : Slope in degrees (NASA SRTM 30m via GEE)

    Vectors (GeoPackage):
        - pa.gpkg          : Protected area polygons (WDPA via GEE)
        - road.gpkg        : Road network lines (OSM via osmnx)
        - river.gpkg       : Waterway lines (OSM via osmnx)
        - town.gpkg        : Settlement points (OSM via osmnx)

Data sources:
    - NASA SRTM 30m (USGS)    -> GEE: USGS/SRTMGL1_003
    - WDPA (Protected Planet) -> GEE: WCMC/WDPA/current/polygons
    - OpenStreetMap            -> osmnx (features_from_polygon)

The AOI is defined by the user as either an extent tuple
(xmin, ymin, xmax, ymax) or a vector file (GPKG/SHP/GeoJSON).
No country codes needed.

Based on the forestatrisk Python package by Ghislain Vieilledent (Cirad).

Usage:
    import ee
    import variable_downloader as vd

    ee.Initialize(project="your-project",
                  opt_url="https://earthengine-highvolume.googleapis.com")

    # Download everything
    vd.get_variables(
        aoi="path/to/aoi.gpkg",
        output_dir="data",
    )

    # Or individually
    vd.get_srtm(aoi="aoi.gpkg", output_dir="data")
    vd.get_wdpa(aoi="aoi.gpkg", output_dir="data")
    vd.get_osm(aoi="aoi.gpkg", output_dir="data")   # all OSM at once
    vd.get_roads(aoi="aoi.gpkg", output_dir="data")  # roads only
    vd.get_rivers(aoi="aoi.gpkg", output_dir="data") # rivers only
    vd.get_towns(aoi="aoi.gpkg", output_dir="data")  # towns only

Dependencies:
    - ee (earthengine-api)
    - numpy
    - osgeo (GDAL/OGR/OSR)
    - osmnx (for OSM feature download)
    - geopandas
    - shapely
"""

import os
import json
import math
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import geopandas as gpd
from shapely.geometry import box as shapely_box
from osgeo import gdal, ogr, osr

gdal.UseExceptions()


# ============================================================
# Constants
# ============================================================

SCALE_30M  = 0.000277777777778   # ~30m at equator
SCALE_90M  = 0.000833333333333   # ~90m at equator
SCALE_10M  = 0.000092592592593   # ~10m at equator (kept for reference)
SCALE_100M = 0.000925925925926   # ~100m at equator (GHSL GHS_BUILT_S native)
COMPUTE_LIMIT = 50_331_648       # computePixels ~48 MB limit

# Available epochs for JRC/GHSL/P2023A/GHS_BUILT_S (5-year steps)
GHSL_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]


def _snap_ghsl_epoch(year):
    """Snap a year to the nearest available GHSL epoch."""
    return min(GHSL_EPOCHS, key=lambda e: abs(e - year))

# OSM feature tags for osmnx queries
OSM_ROAD_TAGS = {
    "highway": [
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link",
        "tertiary", "tertiary_link", "road", "unclassified",
        "residential", "service", "track", "living_street", "path",
    ]
}
OSM_RIVER_TAGS = {"waterway": True}  # all waterway features; geom-type filter keeps lines
OSM_TOWN_TAGS = {
    "place": ["city", "town", "village", "hamlet"]
}


# ============================================================
# AOI parsing  (mirrors forest_downloader.py)
# ============================================================

def _get_extent_from_vector(vector_path):
    """Get bounding box from a vector file in EPSG:4326."""
    ds = ogr.Open(str(vector_path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open vector file: {vector_path}")
    layer = ds.GetLayer()
    srs_src = layer.GetSpatialRef()
    xmin, xmax, ymin, ymax = layer.GetExtent()

    srs_4326 = osr.SpatialReference()
    srs_4326.ImportFromEPSG(4326)
    srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    if srs_src is not None:
        srs_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not srs_src.IsSame(srs_4326):
            transform = osr.CoordinateTransformation(srs_src, srs_4326)
            corners = [(xmin, ymin), (xmin, ymax),
                       (xmax, ymin), (xmax, ymax)]
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
    """Parse AOI into (xmin, ymin, xmax, ymax) with optional buffer."""
    if isinstance(aoi, (tuple, list)) and len(aoi) == 4:
        xmin, ymin, xmax, ymax = aoi
    elif isinstance(aoi, (str, Path)):
        xmin, ymin, xmax, ymax = _get_extent_from_vector(aoi)
    else:
        raise ValueError(
            "aoi must be (xmin, ymin, xmax, ymax) or a vector file path."
        )
    return (xmin - buff, ymin - buff, xmax + buff, ymax + buff)


def _load_aoi_polygon(aoi, buff=0.0):
    """Load AOI as a shapely polygon in EPSG:4326.

    If aoi is a vector file, the actual (dissolved) geometry is returned so
    downstream clipping follows the real admin boundary rather than its bbox.
    """
    if isinstance(aoi, (tuple, list)) and len(aoi) == 4:
        xmin, ymin, xmax, ymax = aoi
        return shapely_box(xmin - buff, ymin - buff, xmax + buff, ymax + buff)
    if isinstance(aoi, (str, Path)):
        gdf = gpd.read_file(str(aoi))
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
        geom = (gdf.geometry.union_all()
                if hasattr(gdf.geometry, "union_all")
                else gdf.geometry.unary_union)
        if buff:
            geom = geom.buffer(buff)
        return geom
    raise ValueError(
        "aoi must be (xmin, ymin, xmax, ymax) or a vector file path."
    )


# ============================================================
# Grid snapping and tiling  (mirrors forest_downloader.py)
# ============================================================

def _snap_extent(extent, scale):
    """Snap extent to a global pixel grid."""
    xmin, ymin, xmax, ymax = extent
    return (
        math.floor(xmin / scale) * scale,
        math.floor(ymin / scale) * scale,
        math.ceil(xmax / scale) * scale,
        math.ceil(ymax / scale) * scale,
    )


def _make_grid(extent, tile_size, scale):
    """Create a list of tile extents snapped to the pixel grid."""
    xmin, ymin, xmax, ymax = extent
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
# GEE tile download via computePixels
# ============================================================

def _download_tile_generic(args):
    """Download one tile using ee.data.computePixels."""
    (tile_extent, ee_image, tile_index, output_dir,
     scale, n_bands, max_retries, verbose) = args
    xmin, ymin, xmax, ymax = tile_extent
    tile_file = os.path.join(output_dir, f"tile_{tile_index:04d}.tif")

    n_cols = round((xmax - xmin) / scale)
    n_rows = round((ymax - ymin) / scale)
    if n_cols == 0 or n_rows == 0:
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
            ds = gdal.Open(tile_file)
            if ds is None:
                raise RuntimeError("GDAL cannot open tile")
            if ds.RasterCount != n_bands:
                raise RuntimeError(
                    f"Expected {n_bands} bands, got {ds.RasterCount}")
            ds = None
            if verbose:
                print(f"  Tile {tile_index} OK ({n_cols}x{n_rows} px)")
            return tile_file
        except Exception as e:
            if verbose:
                print(f"  Tile {tile_index} attempt "
                      f"{attempt}/{max_retries} FAILED: {e}")
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30))

    return None


# ============================================================
# Mosaicking and clipping (mirrors forest_downloader.py)
# ============================================================

def _mosaic_tiles(tile_files, output_file, crop_extent=None):
    """Mosaic tiles into a single GeoTIFF via GDAL VRT."""
    valid = [f for f in tile_files if f and os.path.exists(f)]
    if not valid:
        raise RuntimeError("No valid tiles to mosaic.")

    for f in valid:
        ds = gdal.Open(f, gdal.GA_Update)
        if ds:
            for b in range(1, ds.RasterCount + 1):
                ds.GetRasterBand(b).DeleteNoDataValue()
            ds.FlushCache()
            ds = None

    vrt = output_file.replace(".tif", "_mosaic.vrt")
    vrt_ds = gdal.BuildVRT(
        vrt, valid, options=gdal.BuildVRTOptions(resolution="highest"))
    vrt_ds.FlushCache()
    vrt_ds = None

    opts = {"format": "GTiff",
            "creationOptions": ["COMPRESS=DEFLATE", "TILED=YES"]}
    if crop_extent:
        xmin, ymin, xmax, ymax = crop_extent
        opts["projWin"] = [xmin, ymax, xmax, ymin]
    gdal.Translate(output_file, vrt, **opts)

    if os.path.exists(vrt):
        os.remove(vrt)


def _clip_to_vector(input_file, output_file, vector_path, nodata=255):
    """Clip a raster to a vector boundary.

    Handles CRS mismatch: if the vector and raster are in different
    coordinate systems, the cutlineSRS is set so GDAL reprojects
    the cutline on the fly. Works with both GPKG and SHP.
    """
    ds = ogr.Open(str(vector_path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open vector: {vector_path}")
    layer = ds.GetLayer()
    layer_name = layer.GetName()

    # Get the CRS of the cutline vector
    vec_srs = layer.GetSpatialRef()
    cutline_srs = None
    if vec_srs is not None:
        cutline_srs = vec_srs.ExportToWkt()
    ds = None

    src = gdal.Open(input_file, gdal.GA_Update)
    if src:
        for b in range(1, src.RasterCount + 1):
            src.GetRasterBand(b).DeleteNoDataValue()
        src.FlushCache()
        src = None

    warp_kwargs = dict(
        format="GTiff",
        cutlineDSName=str(vector_path),
        cutlineLayer=layer_name,
        cropToCutline=True,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
        dstNodata=nodata,
    )
    # Tell GDAL the CRS of the cutline so it can reproject if needed
    if cutline_srs is not None:
        warp_kwargs["cutlineSRS"] = cutline_srs

    gdal.Warp(output_file, input_file,
              options=gdal.WarpOptions(**warp_kwargs))


# ============================================================
# Generic GEE raster downloader
# ============================================================

def _download_ee_raster(ee_image, aoi, output_file, scale=SCALE_90M,
                        n_bands=1, bytes_per_pixel=4, buff=0.0,
                        crop_to_aoi=True, nodata=None, tile_size=None,
                        parallel=True, max_retries=3, verbose=True):
    """Download an ee.Image to a local GeoTIFF."""
    extent = _parse_aoi(aoi, buff)

    safe_pixels = COMPUTE_LIMIT / (n_bands * bytes_per_pixel)
    safe_side = int(math.sqrt(safe_pixels))
    max_tile_deg = round(safe_side * scale, 6)
    if tile_size is None:
        tile_size = max_tile_deg
    elif tile_size > max_tile_deg:
        tile_size = max_tile_deg

    if verbose:
        print(f"  Scale: {scale:.10f} | Tile: {tile_size:.4f} "
              f"| Bands: {n_bands}")

    snapped = _snap_extent(extent, scale)
    tiles = _make_grid(snapped, tile_size, scale)
    if verbose:
        print(f"  Tiles: {len(tiles)}")

    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)
    tile_dir = os.path.join(out_dir, "_tiles_tmp")
    os.makedirs(tile_dir, exist_ok=True)

    dl_args = [
        (t, ee_image, i, tile_dir, scale, n_bands, max_retries, verbose)
        for i, t in enumerate(tiles)
    ]

    if parallel and len(tiles) > 1:
        ncpu = min(len(tiles), max(1, multiprocessing.cpu_count() - 1), 10)
        tile_files = [None] * len(tiles)
        with ThreadPoolExecutor(max_workers=ncpu) as pool:
            fmap = {pool.submit(_download_tile_generic, a): a[2]
                    for a in dl_args}
            for fut in as_completed(fmap):
                tile_files[fmap[fut]] = fut.result()
    else:
        tile_files = [_download_tile_generic(a) for a in dl_args]

    n_ok = sum(1 for f in tile_files if f)
    if verbose:
        print(f"  Downloaded: {n_ok}/{len(tiles)} tiles")

    aoi_is_vector = (isinstance(aoi, (str, Path))
                     and os.path.isfile(str(aoi)))
    if crop_to_aoi and aoi_is_vector:
        tmp = output_file.replace(".tif", "_mosaic_tmp.tif")
        _mosaic_tiles(tile_files, tmp)
        nd = nodata if nodata is not None else 255
        _clip_to_vector(tmp, output_file, str(aoi), nodata=nd)
        if os.path.exists(tmp):
            os.remove(tmp)
    else:
        crop = _snap_extent(extent, scale) if crop_to_aoi else None
        _mosaic_tiles(tile_files, output_file, crop_extent=crop)

    for f in tile_files:
        if f and os.path.exists(f):
            os.remove(f)
    if os.path.isdir(tile_dir) and not os.listdir(tile_dir):
        os.rmdir(tile_dir)

    if verbose:
        print(f"  Output: {output_file}")
    return output_file


# ============================================================
# EE image builder: SRTM
# ============================================================

def ee_srtm():
    """Build a 2-band ee.Image: altitude (Int16) + slope (Float32).

    Uses NASA SRTM 1 arc-second (~30m) from USGS/SRTMGL1_003.
    """
    srtm = ee.Image("USGS/SRTMGL1_003").select("elevation")
    altitude = srtm.rename("altitude").toInt16()
    slope = ee.Terrain.slope(srtm).rename("slope").toFloat()
    return ee.Image.cat([altitude, slope])


def ee_ghsl_built(year):
    """Build binary built-up raster from GHSL P2023A GHS_BUILT_S at 100m.

    Uses JRC/GHSL/P2023A/GHS_BUILT_S — multi-epoch 100m collection
    (epochs: 1975–2030 in 5-year steps). The input year is snapped to the
    nearest available epoch.

    Both bands are combined: any non-zero value in either → 1 (built-up):
      - built_surface      : total built-up surface area (m² per cell)
      - built_surface_nres  : non-residential built-up surface area (m² per cell)

    Resampling from 100m to 30m (nearest neighbor) is performed in Notebook 3
    during alignment to the reference raster (forest_t2.tif).

    :param year: Target year — snapped to nearest GHSL epoch.
    :return: ee.Image with band 'built_up' (uint8: 1=built-up, 0=not built-up).
    """
    epoch = _snap_ghsl_epoch(year)
    img = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}")
    built = img.select("built_surface").gt(0)
    nres  = img.select("built_surface_nres").gt(0)
    return built.Or(nres).toUint8().rename("built_up")


# ============================================================
# Reprojection utility
# ============================================================

def _reproject_raster(input_file, output_file, dst_crs,
                      resampling="near", resolution=None, nodata=None):
    """Reproject a raster to a target CRS."""
    resample_map = {
        "near": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "average": gdal.GRA_Average,
        "mode": gdal.GRA_Mode,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    src = gdal.Open(input_file)
    src_nd = src.GetRasterBand(1).GetNoDataValue()
    src = None
    nd = nodata if nodata is not None else src_nd

    opts = dict(
        format="GTiff", dstSRS=dst_crs,
        resampleAlg=resample_map.get(resampling,
                                     gdal.GRA_NearestNeighbour),
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    if nd is not None:
        opts["srcNodata"] = nd
        opts["dstNodata"] = nd
    if resolution is not None:
        opts["xRes"] = resolution
        opts["yRes"] = resolution

    gdal.Warp(output_file, input_file, options=gdal.WarpOptions(**opts))
    return output_file


def _reproject_vector(input_path, output_path, dst_crs, verbose=True):
    """Reproject a vector file to a target CRS.

    :param input_path: Input GPKG or SHP.
    :param output_path: Output GPKG.
    :param dst_crs: Target CRS (e.g. "EPSG:32749").
    :param verbose: Print progress.
    :return: Path to output file, or None on failure.
    """
    ds_in = ogr.Open(input_path)
    if ds_in is None:
        return None
    layer_in = ds_in.GetLayer()

    srs_dst = osr.SpatialReference()
    srs_dst.SetFromUserInput(dst_crs)
    srs_dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    srs_src = layer_in.GetSpatialRef()
    if srs_src is None:
        srs_src = osr.SpatialReference()
        srs_src.ImportFromEPSG(4326)
    srs_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    ct = osr.CoordinateTransformation(srs_src, srs_dst)

    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(output_path):
        drv.DeleteDataSource(output_path)
    ds_out = drv.CreateDataSource(output_path)

    out_layer = ds_out.CreateLayer(
        layer_in.GetName(), srs_dst, layer_in.GetGeomType())

    defn = layer_in.GetLayerDefn()
    for i in range(defn.GetFieldCount()):
        out_layer.CreateField(defn.GetFieldDefn(i))

    for feat in layer_in:
        geom = feat.GetGeometryRef()
        if geom is not None:
            geom = geom.Clone()
            geom.Transform(ct)
        out_feat = ogr.Feature(out_layer.GetLayerDefn())
        out_feat.SetGeometry(geom)
        for i in range(defn.GetFieldCount()):
            out_feat.SetField(i, feat.GetField(i))
        out_layer.CreateFeature(out_feat)

    ds_out.FlushCache()
    ds_out = None
    ds_in = None

    if verbose:
        print(f"  Reprojected vector → {dst_crs}: {output_path}")
    return output_path


# ============================================================
# SRTM downloader (raster)
# ============================================================

def get_srtm(aoi, output_dir="data", buff=0.0, crop_to_aoi=True,
             output_crs=None, parallel=True, max_retries=3, verbose=True):
    """Download SRTM altitude and slope rasters from GEE.

    :param aoi: (xmin, ymin, xmax, ymax) or vector file path.
    :param output_dir: Output directory.
    :param buff: Buffer in degrees around AOI.
    :param crop_to_aoi: Clip to AOI boundary.
    :param output_crs: Reproject outputs to this CRS (e.g. "EPSG:32749").
        If None (default), outputs stay in EPSG:4326.
    :param parallel: Parallel tile downloads.
    :param max_retries: Retry attempts per tile.
    :param verbose: Print progress.
    :return: Dict with 'altitude' and 'slope' file paths.
    """
    if verbose:
        print("=" * 60)
        print("Downloading SRTM (altitude + slope) from GEE...")

    os.makedirs(output_dir, exist_ok=True)
    combined = os.path.join(output_dir, "_srtm_combined.tif")

    _download_ee_raster(
        ee_image=ee_srtm(), aoi=aoi, output_file=combined,
        scale=SCALE_30M, n_bands=2, bytes_per_pixel=6,
        buff=buff, crop_to_aoi=crop_to_aoi, nodata=-32768,
        parallel=parallel, max_retries=max_retries, verbose=verbose,
    )

    # Split into separate files
    ds = gdal.Open(combined)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    drv = gdal.GetDriverByName("GTiff")
    result = {}

    # Band 1: altitude
    alt_path = os.path.join(output_dir, "altitude.tif")
    alt_ds = drv.Create(alt_path, nx, ny, 1, gdal.GDT_Int16,
                        ["COMPRESS=DEFLATE", "TILED=YES"])
    alt_ds.SetGeoTransform(gt)
    alt_ds.SetProjection(proj)
    alt_ds.GetRasterBand(1).WriteArray(ds.GetRasterBand(1).ReadAsArray())
    alt_ds.GetRasterBand(1).SetNoDataValue(-32768)
    alt_ds.FlushCache()
    alt_ds = None
    result["altitude"] = alt_path

    # Band 2: slope
    slp_path = os.path.join(output_dir, "slope.tif")
    slp_ds = drv.Create(slp_path, nx, ny, 1, gdal.GDT_Float32,
                        ["COMPRESS=DEFLATE", "TILED=YES"])
    slp_ds.SetGeoTransform(gt)
    slp_ds.SetProjection(proj)
    slope_arr = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    slope_arr[slope_arr == -32768] = -9999.0  # remap clip sentinel to declared nodata
    slp_ds.GetRasterBand(1).WriteArray(slope_arr)
    slp_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    slp_ds.FlushCache()
    slp_ds = None
    result["slope"] = slp_path

    ds = None
    if os.path.exists(combined):
        os.remove(combined)

    if verbose:
        print(f"  altitude : {alt_path}")
        print(f"  slope    : {slp_path}")

    # Reproject if requested
    if output_crs is not None:
        for var, path in [("altitude", alt_path), ("slope", slp_path)]:
            tmp = path.replace(".tif", "_4326.tif")
            os.rename(path, tmp)
            rsmp = "bilinear" if var in ("altitude", "slope") else "near"
            _reproject_raster(tmp, path, dst_crs=output_crs,
                              resampling=rsmp, resolution=30)
            os.remove(tmp)
            if verbose:
                print(f"  Reprojected {var} → {output_crs} (30m)")

    return result


# ============================================================
# GHSL downloader (raster — alternative to OSM towns)
# ============================================================

def get_ghsl(aoi, years, output_dir="data", buff=0.0, crop_to_aoi=True,
             output_crs=None, parallel=True, max_retries=3, verbose=True):
    """Download GHSL built-up surface from GEE for t2 and t3 years.

    Uses JRC/GHSL/P2023A/GHS_BUILT_S — multi-epoch 100m collection
    (epochs: 1975–2030 in 5-year steps). Downloads two images (one per input
    year), each snapped to the nearest available epoch. Both built_surface and
    nres_built_surface bands are combined into a single binary raster.

    Resampling from 100m to 30m (nearest neighbor) and alignment to the
    reference raster (forest_t2.tif) are performed in Notebook 3.

    :param aoi: (xmin, ymin, xmax, ymax) or vector file path.
    :param years: [t2_year, t3_year] — each snapped to nearest GHSL epoch.
    :param output_dir: Output directory (default: "data").
    :param buff: Buffer in degrees around AOI.
    :param crop_to_aoi: Clip rasters to AOI boundary.
    :param output_crs: Reproject to this CRS (e.g. "EPSG:32749"). If None,
        output stays in EPSG:4326. Native 100m resolution is preserved.
    :param parallel: Parallel tile downloads.
    :param max_retries: Retry attempts per tile.
    :param verbose: Print progress.
    :return: Dict with 'ghsl_built_t2' and 'ghsl_built_t3' keys → file paths.
    """
    if verbose:
        print("=" * 60)
        print("Downloading GHSL built-up surface from GEE (100m, GHS_BUILT_S)...")

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for label, year in zip(["t2", "t3"], years):
        epoch = _snap_ghsl_epoch(year)
        out_path = os.path.join(output_dir, f"ghsl_built_{label}.tif")

        if verbose:
            snap_note = f" → snapped to {epoch}" if epoch != year else ""
            print(f"\n  [{label}] year={year}{snap_note}")

        _download_ee_raster(
            ee_image=ee_ghsl_built(year),
            aoi=aoi,
            output_file=out_path,
            scale=SCALE_100M,
            n_bands=1,
            bytes_per_pixel=4,  # source bands are float32; conservative sizing
            buff=buff,
            crop_to_aoi=crop_to_aoi,
            nodata=255,
            parallel=parallel,
            max_retries=max_retries,
            verbose=verbose,
        )

        if output_crs is not None and os.path.exists(out_path):
            tmp = out_path.replace(".tif", "_4326.tif")
            os.rename(out_path, tmp)
            # Keep native 100m — resampling to 30m happens in Notebook 3
            # via reproject_raster_to_match(resample_alg="near").
            _reproject_raster(tmp, out_path, dst_crs=output_crs,
                              resampling="near", resolution=100)
            os.remove(tmp)
            if verbose:
                print(f"  Reprojected ghsl_built_{label} → {output_crs} (100m)")

        if verbose:
            print(f"  ghsl_built_{label} : {out_path}")
        results[f"ghsl_built_{label}"] = out_path

    return results


# ============================================================
# WDPA downloader (vector)
# ============================================================

def get_wdpa(aoi, output_dir="data", buff=0.0, output_crs=None,
             verbose=True):
    """Download protected area polygons from GEE as a GeoPackage.

    Queries the WCMC/WDPA/current/polygons FeatureCollection on
    GEE, filtered to the AOI bounding box.

    :param aoi: (xmin, ymin, xmax, ymax) or vector file path.
    :param output_dir: Output directory.
    :param buff: Buffer in degrees around AOI.
    :param output_crs: Reproject output to this CRS (e.g. "EPSG:32749").
        If None (default), output stays in EPSG:4326.
    :param verbose: Print progress.
    :return: Dict with 'pa' file path.
    """
    if verbose:
        print("=" * 60)
        print("Downloading WDPA protected areas from GEE (vector)...")

    os.makedirs(output_dir, exist_ok=True)
    extent = _parse_aoi(aoi, buff)
    xmin, ymin, xmax, ymax = extent

    if verbose:
        print(f"  Extent: {xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}")

    # Build bounding box and filter WDPA
    bbox = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
    wdpa = (
        ee.FeatureCollection("WCMC/WDPA/current/polygons")
        .filterBounds(bbox)
    )

    # Select useful attributes to reduce download size.
    # If .select() fails (column names may vary), download all
    # and let _geojson_to_gpkg sanitize the output.
    try:
        wdpa_select = wdpa.select(
            ["WDPAID", "WDPA_PID", "NAME", "ORIG_NAME", "DESIG",
             "DESIG_TYPE", "IUCN_CAT", "INT_CRIT", "STATUS",
             "STATUS_YR", "MANG_AUTH", "MANG_PLAN", "NO_TAKE",
             "REP_AREA", "GIS_AREA", "ISO3"],
        )
    except Exception:
        if verbose:
            print("  Column selection failed, downloading all attributes.")
        wdpa_select = wdpa

    # Count features
    if verbose:
        count = wdpa_select.size().getInfo()
        print(f"  Protected areas found: {count}")
        if count == 0:
            print("  WARNING: No protected areas in this AOI.")
            pa_path = os.path.join(output_dir, "pa.gpkg")
            _create_empty_gpkg(pa_path, ogr.wkbMultiPolygon)
            return {"pa": pa_path}

    # Download as GeoJSON via computeFeatures
    if verbose:
        print("  Downloading features from GEE...")

    fc_dict = wdpa_select.getInfo()

    # Write to GeoPackage
    pa_path = os.path.join(output_dir, "pa.gpkg")
    _geojson_to_gpkg(fc_dict, pa_path, "protected_areas", verbose=verbose)

    # Clip to AOI polygon (filterBounds only intersects — need true clip)
    aoi_polygon = _load_aoi_polygon(aoi, buff)
    try:
        gdf = gpd.read_file(pa_path)
        if not gdf.empty:
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs="EPSG:4326")
            before = len(gdf)
            gdf = gpd.clip(gdf.to_crs("EPSG:4326"), aoi_gdf)
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
            if verbose:
                print(f"  Clipped to AOI: {before} -> {len(gdf)} features")
            os.remove(pa_path)
            if not gdf.empty:
                gdf.to_file(pa_path, driver="GPKG")
            else:
                _create_empty_gpkg(pa_path, ogr.wkbMultiPolygon)
    except Exception as e:
        if verbose:
            print(f"  WARNING: clip step skipped ({e})")

    # Reproject if requested
    if output_crs is not None and os.path.exists(pa_path):
        tmp = pa_path.replace(".gpkg", "_4326.gpkg")
        os.rename(pa_path, tmp)
        _reproject_vector(tmp, pa_path, output_crs, verbose=verbose)
        os.remove(tmp)

    if verbose:
        print(f"  pa : {pa_path}")
    return {"pa": pa_path}


def _create_empty_gpkg(output_path, geom_type=ogr.wkbMultiPolygon):
    """Create an empty GeoPackage with a single layer."""
    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(output_path):
        drv.DeleteDataSource(output_path)
    ds = drv.CreateDataSource(output_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.CreateLayer("empty", srs, geom_type)
    ds.FlushCache()
    ds = None


def _geojson_to_gpkg(fc_dict, output_path, layer_name, verbose=True):
    """Convert a GEE FeatureCollection dict to GeoPackage.

    Sanitizes GEE output by removing properties with empty or
    invalid names (GEE adds system fields like "" and "id" that
    break GeoPackage writers).

    :param fc_dict: Dict from ee.FeatureCollection.getInfo().
    :param output_path: Output GPKG path.
    :param layer_name: Layer name in the GPKG.
    :param verbose: Print progress.
    """
    # Sanitize: remove properties with empty/invalid names
    # and strip the top-level "id" that GEE adds (conflicts
    # with GPKG's built-in fid column)
    valid_features = []
    for feat in fc_dict.get("features", []):
        # Remove top-level "id" (GEE system field)
        feat.pop("id", None)

        props = feat.get("properties", {})
        clean_props = {}
        for k, v in props.items():
            # Skip empty names, "id", "system:index" etc.
            if not k or k.strip() == "" or k.startswith("system:"):
                continue
            clean_props[k] = v
        feat["properties"] = clean_props
        valid_features.append(feat)
    fc_dict["features"] = valid_features

    if not valid_features:
        if verbose:
            print("  WARNING: No valid features, creating empty GPKG.")
        _create_empty_gpkg(output_path)
        return

    # Write sanitized GeoJSON to temp file
    tmp_json = output_path.replace(".gpkg", "_tmp.geojson")
    with open(tmp_json, "w") as f:
        json.dump(fc_dict, f)

    # Convert via OGR
    ds_in = ogr.Open(tmp_json)
    if ds_in is None or ds_in.GetLayerCount() == 0:
        if verbose:
            print("  WARNING: Cannot read GeoJSON, creating empty GPKG.")
        _create_empty_gpkg(output_path)
        if os.path.exists(tmp_json):
            os.remove(tmp_json)
        return

    layer_in = ds_in.GetLayer()

    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(output_path):
        drv.DeleteDataSource(output_path)
    ds_out = drv.CreateDataSource(output_path)

    srs = layer_in.GetSpatialRef()
    if srs is None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

    out_layer = ds_out.CreateLayer(
        layer_name, srs, layer_in.GetGeomType())

    # Copy only fields with valid names
    defn = layer_in.GetLayerDefn()
    valid_fields = []
    for i in range(defn.GetFieldCount()):
        field_defn = defn.GetFieldDefn(i)
        name = field_defn.GetName()
        if name and name.strip() != "":
            out_layer.CreateField(field_defn)
            valid_fields.append(name)

    # Copy features, transferring only valid fields
    count = 0
    for feat in layer_in:
        out_feat = ogr.Feature(out_layer.GetLayerDefn())
        out_feat.SetGeometry(feat.GetGeometryRef().Clone())
        for field_name in valid_fields:
            idx_src = feat.GetFieldIndex(field_name)
            idx_dst = out_feat.GetFieldIndex(field_name)
            if idx_src >= 0 and idx_dst >= 0:
                out_feat.SetField(idx_dst, feat.GetField(idx_src))
        out_layer.CreateFeature(out_feat)
        count += 1

    ds_out.FlushCache()
    ds_out = None
    ds_in = None

    if os.path.exists(tmp_json):
        os.remove(tmp_json)

    if verbose:
        print(f"  Written {count} features to {output_path}")


# ============================================================
# osmnx-based OSM downloader
# ============================================================

def _download_osm_osmnx(name, tags, keep_geom_types,
                        aoi, output_dir, buff, output_crs,
                        timeout, verbose):
    """Download OSM features via osmnx, clip to AOI polygon, write to GPKG."""
    import osmnx as ox

    os.makedirs(output_dir, exist_ok=True)

    polygon = _load_aoi_polygon(aoi, buff)
    xmin, ymin, xmax, ymax = polygon.bounds

    if verbose:
        print(f"  AOI extent: {xmax - xmin:.2f} x {ymax - ymin:.2f} deg")

    # Configure request timeout
    try:
        ox.settings.requests_timeout = timeout   # osmnx >= 2.0
    except AttributeError:
        ox.settings.timeout = timeout             # osmnx < 2.0

    try:
        try:
            gdf = ox.features_from_polygon(polygon, tags=tags)
        except AttributeError:
            gdf = ox.geometries_from_polygon(polygon, tags=tags)  # osmnx < 1.0
    except Exception as e:
        if verbose:
            print(f"  osmnx returned no features: {e}")
        return {}

    if gdf is None or gdf.empty:
        if verbose:
            print(f"  No {name} features found.")
        return {}

    # Filter to required geometry types and drop all OSM attribute columns
    gdf = gdf[gdf.geometry.geom_type.isin(keep_geom_types)].copy()
    if gdf.empty:
        if verbose:
            print(f"  No {name} features with geometry {keep_geom_types}.")
        return {}

    gdf = gpd.GeoDataFrame(geometry=gdf.geometry.values, crs=gdf.crs)

    # Clip to AOI polygon (in EPSG:4326 before reprojection)
    aoi_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    before = len(gdf)
    gdf = gpd.clip(gdf, aoi_gdf)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf = gdf[gdf.geometry.geom_type.isin(keep_geom_types)]
    if verbose:
        print(f"  Clipped to AOI: {before} -> {len(gdf)} features")

    if gdf.empty:
        if verbose:
            print(f"  No {name} features inside AOI.")
        return {}

    if output_crs is not None:
        gdf = gdf.to_crs(output_crs)

    gpkg_path = os.path.join(output_dir, f"{name}.gpkg")
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    gdf.to_file(gpkg_path, driver="GPKG")

    if verbose:
        print(f"  {len(gdf)} features -> {gpkg_path}")
    return {name: gpkg_path}


def get_roads(aoi, output_dir="data", buff=0.0, output_crs=None,
              timeout=180, verbose=True):
    """Download OSM road network via osmnx.

    :param aoi: (xmin, ymin, xmax, ymax) or vector file path.
    :param output_dir: Output directory.
    :param buff: Buffer in degrees around AOI.
    :param output_crs: Reproject to this CRS (e.g. "EPSG:32749").
    :param timeout: Request timeout in seconds.
    :param verbose: Print progress.
    :return: Dict with 'road' file path (or empty dict if no features).
    """
    if verbose:
        print("=" * 60)
        print("Downloading OSM roads...")
    return _download_osm_osmnx(
        "road", OSM_ROAD_TAGS, ["LineString", "MultiLineString"],
        aoi, output_dir, buff, output_crs, timeout, verbose,
    )


def get_rivers(aoi, output_dir="data", buff=0.0, output_crs=None,
               timeout=180, verbose=True):
    """Download OSM waterways via osmnx.

    :param aoi: (xmin, ymin, xmax, ymax) or vector file path.
    :param output_dir: Output directory.
    :param buff: Buffer in degrees around AOI.
    :param output_crs: Reproject to this CRS (e.g. "EPSG:32749").
    :param timeout: Request timeout in seconds.
    :param verbose: Print progress.
    :return: Dict with 'river' file path (or empty dict if no features).
    """
    if verbose:
        print("=" * 60)
        print("Downloading OSM rivers...")
    return _download_osm_osmnx(
        "river", OSM_RIVER_TAGS, ["LineString", "MultiLineString"],
        aoi, output_dir, buff, output_crs, timeout, verbose,
    )


def get_towns(aoi, output_dir="data", buff=0.0, output_crs=None,
              timeout=180, verbose=True):
    """Download OSM settlement nodes via osmnx.

    :param aoi: (xmin, ymin, xmax, ymax) or vector file path.
    :param output_dir: Output directory.
    :param buff: Buffer in degrees around AOI.
    :param output_crs: Reproject to this CRS (e.g. "EPSG:32749").
    :param timeout: Request timeout in seconds.
    :param verbose: Print progress.
    :return: Dict with 'town' file path (or empty dict if no features).
    """
    if verbose:
        print("=" * 60)
        print("Downloading OSM towns/settlements...")
    return _download_osm_osmnx(
        "town", OSM_TOWN_TAGS, ["Point"],
        aoi, output_dir, buff, output_crs, timeout, verbose,
    )


# ============================================================
# Combined OSM downloader (backwards-compatible wrapper)
# ============================================================

def get_osm(aoi, output_dir="data", buff=0.0, output_crs=None,
            timeout=180, verbose=True):
    """Download OSM road, river, and town vectors via osmnx.

    Wrapper around get_roads(), get_rivers(), get_towns(). Use those
    functions directly to download individual feature types.

    :param aoi: (xmin, ymin, xmax, ymax) or vector file path.
    :param output_dir: Output directory.
    :param buff: Buffer in degrees around AOI.
    :param output_crs: Reproject outputs to this CRS (e.g. "EPSG:32749").
        If None (default), outputs stay in EPSG:4326.
    :param timeout: Request timeout in seconds.
    :param verbose: Print progress.
    :return: Dict with 'road', 'river', 'town' file paths.
    """
    if verbose:
        print("=" * 60)
        print("Downloading OSM vector data (roads + rivers + towns)...")

    kwargs = dict(
        output_dir=output_dir, buff=buff, output_crs=output_crs,
        timeout=timeout, verbose=verbose,
    )
    result = {}
    result.update(get_roads(aoi, **kwargs))
    result.update(get_rivers(aoi, **kwargs))
    result.update(get_towns(aoi, **kwargs))
    return result


# ============================================================
# Main function: get_variables
# ============================================================

def get_variables(aoi, output_dir="data", buff=0.0,
                  output_crs=None, crop_to_aoi=True, parallel=True,
                  max_retries=3, osm_timeout=180,
                  use_ghsl_towns=False, ghsl_years=None,
                  verbose=True):
    """Download all spatial data for forestatrisk.

    Raster outputs (GeoTIFF):
        - altitude.tif
        - slope.tif
        - ghsl_built_t2.tif, ghsl_built_t3.tif  (only when use_ghsl_towns=True)

    Vector outputs (GeoPackage):
        - pa.gpkg
        - road.gpkg
        - river.gpkg
        - town.gpkg        (only when use_ghsl_towns=False)

    All outputs are in EPSG:4326 by default. Set output_crs to
    reproject everything to a projected CRS (e.g. "EPSG:32749").

    NOTE: Forest cover and derived variables (dist_edge, dist_defor)
    are handled by forest_downloader.get_fcc().

    :param aoi: (xmin, ymin, xmax, ymax) or vector file path.
    :param output_dir: Output directory (default: "data").
    :param buff: Buffer in degrees around AOI.
    :param output_crs: Reproject all outputs to this CRS
        (e.g. "EPSG:32749"). If None, outputs stay in EPSG:4326.
    :param crop_to_aoi: Clip raster outputs to AOI boundary.
    :param parallel: Parallel tile downloads for GEE.
    :param max_retries: Retries per tile/query.
    :param osm_timeout: OSM/Overpass request timeout in seconds.
    :param use_ghsl_towns: If True, download GHSL built-up surface
        (ghsl_built_t2.tif, ghsl_built_t3.tif) instead of OSM town
        points (town.gpkg). Roads and rivers are always from OSM.
        Default: False.
    :param ghsl_years: [t2_year, t3_year] — required when
        use_ghsl_towns=True. Each year is snapped to the nearest
        available GHSL epoch (1975–2030, 5-year steps).
    :param verbose: Print progress.
    :return: Dict mapping variable names to file paths.
    """
    crs_label = output_crs if output_crs else "EPSG:4326"
    if verbose:
        print("=" * 60)
        print("VARIABLE DOWNLOADER")
        print(f"  AOI        : {aoi}")
        print(f"  Output dir : {output_dir}")
        print(f"  Output CRS : {crs_label}")
        print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    result = {}

    # 1. SRTM (altitude + slope rasters)
    result.update(get_srtm(
        aoi=aoi, output_dir=output_dir, buff=buff,
        crop_to_aoi=crop_to_aoi, output_crs=output_crs,
        parallel=parallel, max_retries=max_retries, verbose=verbose,
    ))

    # 2. WDPA (protected area polygons)
    result.update(get_wdpa(
        aoi=aoi, output_dir=output_dir, buff=buff,
        output_crs=output_crs, verbose=verbose,
    ))

    # 3. OSM roads + rivers (always from OSM)
    osm_kwargs = dict(
        output_dir=output_dir, buff=buff, output_crs=output_crs,
        timeout=osm_timeout, verbose=verbose,
    )
    result.update(get_roads(aoi, **osm_kwargs))
    result.update(get_rivers(aoi, **osm_kwargs))

    # 4. Town settlements: OSM points OR GHSL built-up raster
    if use_ghsl_towns:
        if not ghsl_years or len(ghsl_years) != 2:
            raise ValueError(
                "ghsl_years=[t2_year, t3_year] is required when use_ghsl_towns=True"
            )
        result.update(get_ghsl(
            aoi=aoi, years=ghsl_years, output_dir=output_dir, buff=buff,
            crop_to_aoi=crop_to_aoi, output_crs=output_crs,
            parallel=parallel, max_retries=max_retries, verbose=verbose,
        ))
    else:
        result.update(get_towns(aoi, **osm_kwargs))

    # Summary
    if verbose:
        print("=" * 60)
        print("ALL DATA DOWNLOADED")
        print(f"  CRS: {crs_label}")
        for name, path in result.items():
            ext = os.path.splitext(path)[1]
            fmt = "raster" if ext == ".tif" else "vector"
            print(f"  {name:12s} : {path}  ({fmt})")
        print("=" * 60)

    return result


# ── RunContext-aware entry point ──────────────────────────────

from palmoil_risk.io.run import RunContext


def download_variables(ctx: RunContext) -> dict:
    """Download all spatial covariates for this run.

    Reads parameters from ctx.config. Writes to ctx.raw_dir/variables/.
    Returns the same dict as get_variables().
    """
    cfg = ctx.config
    out_dir = ctx.raw_dir / "variables"

    return get_variables(
        aoi=cfg.aoi_source,
        output_dir=str(out_dir),
        buff=cfg.aoi_buffer,
        output_crs=None,
        crop_to_aoi=True,
        parallel=True,
        max_retries=3,
        osm_timeout=cfg.osm_timeout,
        use_ghsl_towns=cfg.use_ghsl_towns,
        ghsl_years=cfg.ghsl_years,
        verbose=True,
    )
