"""
IO helper functions for forestatrisk data preparation.

Refactored from helper_functions.py (project root).
setup_workspace() has been removed — use RunContext instead.

Includes utilities for:
  - Raster inspection and reprojection
  - Vector inspection, reprojection, and rasterization
  - Masking and alignment verification
  - Batch processing
"""

import os
import glob
import numpy as np
from osgeo import gdal, ogr, osr, gdalconst
from pathlib import Path

from palmdef_risk.constants import NODATA_BYTE, NODATA_FLOAT


# ============================================================
# AOI UTILITIES
# ============================================================

def aoi_bbox_4326(aoi_source: str) -> tuple:
    """Return (xmin, ymin, xmax, ymax) in EPSG:4326 for the AOI source."""
    import geopandas as gpd
    try:
        parts = [float(x) for x in str(aoi_source).split(",")]
        if len(parts) == 4:
            return tuple(parts)
    except ValueError:
        pass
    gdf = gpd.read_file(aoi_source)
    gdf_4326 = gdf.to_crs("EPSG:4326")
    xmin, ymin, xmax, ymax = gdf_4326.total_bounds
    return (xmin, ymin, xmax, ymax)


def raster_shape(path):
    """Return (rows, cols) for a raster, or None if it cannot be opened.

    Single source for the shape-mismatch resumability checks in the process
    stage (align / distances / gravity), which each used to define this locally.
    """
    ds = gdal.Open(str(path))
    if ds is None:
        return None
    shape = (ds.RasterYSize, ds.RasterXSize)
    ds = None
    return shape


# ============================================================
# WORKSPACE UTILITIES
# ============================================================

def remove_if_exists(path):
    """Delete a file if it exists. Safe to call with None."""
    if path and os.path.exists(path):
        os.remove(path)


# ============================================================
# RASTER UTILITIES
# ============================================================

def inspect_raster(filepath):
    """Print key properties of a raster file.

    Args:
        filepath: Path to a raster file (.tif).

    Returns:
        dict with raster properties, or None if file can't be opened.
    """
    ds = gdal.Open(filepath)
    if ds is None:
        print(f"  ERROR: Cannot open {filepath}")
        return None

    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    # Use GDAL statistics API for min/max — avoids loading the full array.
    # SWIG signature is positional: GetStatistics(approx_ok, force).
    stats = band.GetStatistics(0, 1)  # (min, max, mean, std)

    n_pixels = ds.RasterXSize * ds.RasterYSize
    unique_values = None
    if n_pixels < 1e6:
        arr = band.ReadAsArray()
        unique_values = np.unique(arr).tolist()

    info = {
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "bands": ds.RasterCount,
        "pixel_x": gt[1],
        "pixel_y": gt[5],
        "projection": proj,
        "nodata": nodata,
        "dtype": gdal.GetDataTypeName(band.DataType),
        "value_min": stats[0],
        "value_max": stats[1],
        "unique_values": unique_values,
    }

    ds = None

    # Determine CRS type
    if abs(info["pixel_x"]) < 1:
        info["crs_type"] = "Geographic (degrees)"
    else:
        info["crs_type"] = "Projected (meters)"

    return info


def print_raster_info(filepath):
    """Print formatted raster info to console.

    Args:
        filepath: Path to a raster file (.tif).

    Returns:
        dict with raster properties.
    """
    info = inspect_raster(filepath)
    if info is None:
        return None

    print(f"File: {os.path.basename(filepath)}")
    print(f"  Size:        {info['width']} x {info['height']}")
    print(f"  Pixel size:  {info['pixel_x']:.6f} x {info['pixel_y']:.6f}")
    print(f"  NoData:      {info['nodata']}")
    print(f"  Dtype:       {info['dtype']}")
    print(f"  Value range: {info['value_min']} to {info['value_max']}")
    print(f"  CRS type:    {info['crs_type']}")
    return info


def get_mask_properties(mask_file):
    """Read all properties from a reference/mask raster.

    Does NOT load the reference pixel array — the study-area NoData footprint
    is applied later by apply_mask()/apply_mask_float(), which stream it
    block-by-block straight from ``ref_path``. This keeps a single
    full-reference boolean from being materialised and carried across every
    masked layer in align_all (the dominant align-stage OOM source).

    Args:
        mask_file: Path to the reference raster.

    Returns:
        dict with gt, proj, xsize, ysize, nodata, extent, srs, ref_path.
    """
    ds = gdal.Open(mask_file)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    ds = None

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * xsize
    ymin = ymax + gt[5] * ysize

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)

    return {
        "gt": gt,
        "proj": proj,
        "xsize": xsize,
        "ysize": ysize,
        "nodata": nodata,
        "extent": (xmin, ymin, xmax, ymax),
        "srs": srs,
        "ref_path": mask_file,
    }


# ============================================================
# RASTER REPROJECTION
# ============================================================

def reproject_raster(input_file, output_file, target_crs,
                     target_res=None, resample_alg="near"):
    """Reproject a single raster file.

    Args:
        input_file:  Path to input raster.
        output_file: Path to output raster.
        target_crs:  Target CRS string (e.g. "EPSG:32647").
        target_res:  Target pixel resolution in meters (None = auto).
        resample_alg: Resampling algorithm ("near", "bilinear", etc.).

    Returns:
        True if successful, False otherwise.
    """
    src_ds = gdal.Open(input_file)
    if src_ds is None:
        print(f"  ERROR: Cannot open {input_file}")
        return False

    band = src_ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    dtype = band.DataType
    src_ds = None

    warp_options = {
        "dstSRS": target_crs,
        "resampleAlg": resample_alg,
        "format": "GTiff",
        "outputType": dtype,
        "creationOptions": ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    }

    if nodata is not None:
        warp_options["srcNodata"] = nodata
        warp_options["dstNodata"] = nodata

    if target_res is not None:
        warp_options["xRes"] = target_res
        warp_options["yRes"] = target_res

    remove_if_exists(output_file)
    result = gdal.Warp(output_file, input_file, **warp_options)

    if result is None:
        print(f"  ERROR: Reprojection failed for {input_file}")
        return False

    result = None
    return True


def reproject_raster_to_match(input_file, output_file, mask_props,
                              resample_alg="bilinear", output_dtype=None):
    """Reproject and align a raster to exactly match a reference raster.

    Args:
        input_file:  Path to input raster.
        output_file: Path to output raster.
        mask_props:  dict from get_mask_properties().
        resample_alg: Resampling algorithm. Use "bilinear" for continuous
                      data (DEM, slope), "near" for categorical data, or
                      "mode" for majority resampling when downscaling a
                      categorical raster (e.g. GHSL 10m → 30m).
        output_dtype: GDAL data type constant for the output raster
                      (e.g. gdal.GDT_Byte for categorical rasters).
                      Defaults to None → Float32 (existing behaviour).

    Returns:
        True if successful, False otherwise.
    """
    src_ds = gdal.Open(input_file)
    if src_ds is None:
        print(f"  ERROR: Cannot open {input_file}")
        return False

    band = src_ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    src_ds = None

    xmin, ymin, xmax, ymax = mask_props["extent"]

    warp_options = {
        "dstSRS": mask_props["proj"],
        "outputBounds": (xmin, ymin, xmax, ymax),
        "width": mask_props["xsize"],
        "height": mask_props["ysize"],
        "resampleAlg": resample_alg,
        "format": "GTiff",
        "outputType": output_dtype if output_dtype is not None else gdalconst.GDT_Float32,
        "creationOptions": ["COMPRESS=LZW", "TILED=YES"],
    }

    if nodata is not None:
        warp_options["srcNodata"] = nodata

    remove_if_exists(output_file)
    result = gdal.Warp(output_file, input_file, **warp_options)

    if result is None:
        print(f"  ERROR: Alignment failed for {input_file}")
        return False

    result = None
    return True


def batch_reproject_rasters(input_dir, output_dir, target_crs,
                            target_res=None, resample_alg="near",
                            extensions=None):
    """Reproject all rasters in a directory.

    Args:
        input_dir:   Directory with input rasters.
        output_dir:  Directory for output rasters.
        target_crs:  Target CRS string.
        target_res:  Target pixel resolution (None = auto).
        resample_alg: Resampling algorithm.
        extensions:  List of file extensions to process.

    Returns:
        Tuple of (success_count, fail_count).
    """
    if extensions is None:
        extensions = [".tif", ".tiff"]

    os.makedirs(output_dir, exist_ok=True)

    raster_files = []
    for ext in extensions:
        raster_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        raster_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    raster_files = sorted(set(raster_files))

    if not raster_files:
        print(f"No raster files found in {input_dir}")
        return 0, 0

    print(f"Found {len(raster_files)} raster file(s)\n")

    success_count = 0
    fail_count = 0

    for i, input_file in enumerate(raster_files, 1):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)

        print(f"[{i}/{len(raster_files)}] {filename}")

        info = inspect_raster(input_file)
        if info is None:
            fail_count += 1
            continue

        print(f"  Input:  {info['width']}x{info['height']}, "
              f"pixel={info['pixel_x']:.6f}, "
              f"nodata={info['nodata']}, dtype={info['dtype']}")

        ok = reproject_raster(
            input_file, output_file,
            target_crs=target_crs,
            target_res=target_res,
            resample_alg=resample_alg,
        )

        if ok:
            out_info = inspect_raster(output_file)
            if out_info:
                print(f"  Output: {out_info['width']}x{out_info['height']}, "
                      f"pixel={out_info['pixel_x']:.2f}m, "
                      f"nodata={out_info['nodata']}, dtype={out_info['dtype']}")
            print(f"  Saved to {output_file}")
            success_count += 1
        else:
            fail_count += 1

        print()

    print(f"DONE: {success_count} succeeded, {fail_count} failed")
    return success_count, fail_count


# ============================================================
# VECTOR UTILITIES
# ============================================================

def inspect_vector(filepath):
    """Print key properties of a vector file.

    Args:
        filepath: Path to a vector file (.shp, .gpkg, .geojson).

    Returns:
        dict with vector properties, or None if file can't be opened.
    """
    ds = ogr.Open(filepath)
    if ds is None:
        print(f"  ERROR: Cannot open {filepath}")
        return None

    layer = ds.GetLayer()
    srs = layer.GetSpatialRef()
    geom_type = ogr.GeometryTypeToName(layer.GetGeomType())
    feat_count = layer.GetFeatureCount()
    extent = layer.GetExtent()

    info = {
        "features": feat_count,
        "geom_type": geom_type,
        "extent": extent,
        "srs": srs,
    }

    print(f"  Features:  {feat_count}")
    print(f"  Geometry:  {geom_type}")
    print(f"  Extent:    ({extent[0]:.4f}, {extent[2]:.4f}, "
          f"{extent[1]:.4f}, {extent[3]:.4f})")

    if srs:
        crs_name = srs.GetAttrValue("PROJCS") or srs.GetAttrValue("GEOGCS")
        print(f"  CRS:       {crs_name}")
        if srs.IsGeographic():
            print(f"  CRS type:  Geographic (degrees) -> will be reprojected")
        else:
            print(f"  CRS type:  Projected (meters)")
    else:
        print(f"  CRS:       UNDEFINED")

    ds = None
    return info


def reproject_vector(input_path, output_path, target_srs):
    """Reproject a vector file to the target SRS.

    Uses gdal.VectorTranslate (wraps ogr2ogr) for C-level performance —
    significantly faster than a Python feature loop for large files.

    Args:
        input_path:  Path to input vector file (.shp, .gpkg, .geojson, etc.).
        output_path: Path to output vector file.
        target_srs:  osr.SpatialReference object for target CRS.

    Returns:
        Path to the output file (or input_path if already in correct CRS).
    """
    src_ds = ogr.Open(input_path)
    src_layer = src_ds.GetLayer()
    src_srs = src_layer.GetSpatialRef()
    src_ds = None

    if src_srs and src_srs.IsSame(target_srs):
        return input_path

    # Extract EPSG code from osr object; fall back to WKT export
    epsg_code = target_srs.GetAuthorityCode(None)
    dst_srs_str = f"EPSG:{epsg_code}" if epsg_code else target_srs.ExportToWkt()

    # Infer output format from extension
    ext = os.path.splitext(output_path)[1].lower().lstrip(".")
    fmt = {"gpkg": "GPKG", "shp": "ESRI Shapefile",
           "geojson": "GeoJSON", "json": "GeoJSON"}.get(ext, "GPKG")

    if os.path.exists(output_path):
        drv = ogr.GetDriverByName(fmt)
        if drv:
            drv.DeleteDataSource(output_path)

    result = gdal.VectorTranslate(
        output_path, input_path,
        options=gdal.VectorTranslateOptions(
            dstSRS=dst_srs_str, reproject=True, format=fmt,
        ),
    )
    if result is None:
        raise RuntimeError(
            f"VectorTranslate failed: {input_path} -> {output_path}")
    result = None
    return output_path


# ============================================================
# RASTERIZATION
# ============================================================

def rasterize_vector(vector_path, output_path, burn_value,
                     mask_props, nodata_value=NODATA_BYTE):
    """Rasterize a vector file to match the reference raster grid.

    Args:
        vector_path: Path to input vector file.
        output_path: Path to output raster file.
        burn_value:  Pixel value to assign where features exist.
        mask_props:  dict from get_mask_properties().
        nodata_value: NoData value for output raster.

    Returns:
        True if successful, False otherwise.
    """
    driver = gdal.GetDriverByName("GTiff")
    remove_if_exists(output_path)
    out_ds = driver.Create(
        output_path, mask_props["xsize"], mask_props["ysize"], 1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(mask_props["gt"])
    out_ds.SetProjection(mask_props["proj"])

    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(nodata_value)
    band.Fill(0)

    vec_ds = ogr.Open(vector_path)
    if vec_ds is None:
        print(f"  ERROR: Cannot open {vector_path}")
        out_ds = None
        return False

    vec_layer = vec_ds.GetLayer()

    gdal.RasterizeLayer(
        out_ds, [1], vec_layer,
        burn_values=[burn_value],
        options=["ALL_TOUCHED=TRUE"],
    )

    out_ds.FlushCache()
    vec_ds = None
    out_ds = None
    return True


# ============================================================
# MASKING
# ============================================================

def _stamp_reference_nodata(raster_path, ref_path, nodata_value, as_float):
    """Stamp ``nodata_value`` into ``raster_path`` wherever the reference raster
    is NoData, streaming over GDAL blocks.

    Numerically identical to ``arr[ref == ref_nodata] = nodata_value`` on the
    full arrays, but never holds the full reference or the full target in RAM:
    peak is one block of each. ``raster_path`` and ``ref_path`` must share grid
    dimensions (the align/distance pipeline guarantees this). The target's
    NoData value is always (re)set, matching the prior full-array behaviour
    even when the reference itself declares no NoData.
    """
    ref_ds = gdal.Open(str(ref_path))
    ref_band = ref_ds.GetRasterBand(1)
    ref_nd = ref_band.GetNoDataValue()

    ds = gdal.Open(str(raster_path), gdal.GA_Update)
    band = ds.GetRasterBand(1)
    xsize, ysize = band.XSize, band.YSize
    bx, by = band.GetBlockSize()

    if ref_nd is not None:
        for yoff in range(0, ysize, by):
            ywin = min(by, ysize - yoff)
            for xoff in range(0, xsize, bx):
                xwin = min(bx, xsize - xoff)
                ref_blk = ref_band.ReadAsArray(xoff, yoff, xwin, ywin)
                mask = ref_blk == ref_nd
                if not mask.any():
                    continue
                blk = band.ReadAsArray(xoff, yoff, xwin, ywin)
                if as_float:
                    blk = blk.astype(np.float32)
                blk[mask] = nodata_value
                band.WriteArray(blk, xoff, yoff)

    band.SetNoDataValue(nodata_value)
    ds.FlushCache()
    ds = None
    ref_ds = None


def apply_mask(raster_path, ref_path, nodata_value=NODATA_BYTE):
    """Apply study-area mask to a Byte raster — set NoData outside study area.

    Streams the reference NoData footprint block-by-block (see
    _stamp_reference_nodata) instead of carrying a full-reference boolean.

    Args:
        raster_path:  Path to raster file (modified in place).
        ref_path:     Path to the reference raster whose NoData footprint
                      defines the study area.
        nodata_value: NoData value to write.
    """
    _stamp_reference_nodata(raster_path, ref_path, nodata_value, as_float=False)


def apply_mask_float(raster_path, ref_path, nodata_value=NODATA_FLOAT):
    """Apply study-area mask to a float raster (e.g. DEM, slope).

    Streams the reference NoData footprint block-by-block (see
    _stamp_reference_nodata) instead of carrying a full-reference boolean.

    Args:
        raster_path:  Path to raster file (modified in place).
        ref_path:     Path to the reference raster whose NoData footprint
                      defines the study area.
        nodata_value: NoData value to write.
    """
    _stamp_reference_nodata(raster_path, ref_path, nodata_value, as_float=True)


# ============================================================
# ALIGNMENT VERIFICATION
# ============================================================

def verify_alignment(filepath, ref_file, verbose=True):
    """Check if a raster is perfectly aligned with a reference.

    Args:
        filepath: Path to raster to check.
        ref_file: Path to reference raster.
        verbose:  Print results.

    Returns:
        True if all checks pass.
    """
    ds1 = gdal.Open(filepath)
    ds2 = gdal.Open(ref_file)

    if ds1 is None or ds2 is None:
        if verbose:
            print(f"  Cannot open one of the files")
        return False

    gt1, gt2 = ds1.GetGeoTransform(), ds2.GetGeoTransform()

    checks = {
        "Same width":      ds1.RasterXSize == ds2.RasterXSize,
        "Same height":     ds1.RasterYSize == ds2.RasterYSize,
        "Same pixel size": abs(gt1[1] - gt2[1]) < 0.01,
        "Same origin X":   abs(gt1[0] - gt2[0]) < 0.01,
        "Same origin Y":   abs(gt1[3] - gt2[3]) < 0.01,
        "Same CRS":        ds1.GetProjection() == ds2.GetProjection(),
    }

    ds1 = None
    ds2 = None

    all_ok = all(checks.values())
    if verbose:
        for check, passed in checks.items():
            status = "OK" if passed else "FAIL"
            print(f"  {status} {check}")
    return all_ok


def batch_verify_alignment(file_list, ref_file):
    """Verify alignment of multiple rasters against a reference.

    Args:
        file_list: List of raster file paths.
        ref_file:  Reference raster path.
    """
    print(f"Alignment check against {os.path.basename(ref_file)}")
    print("-" * 50)

    for f in file_list:
        if os.path.exists(f):
            ok = verify_alignment(f, ref_file, verbose=False)
            status = "OK" if ok else "FAIL"
            print(f"  {status} {os.path.basename(f)}")
        else:
            print(f"  ? {os.path.basename(f)} - not found")


# ============================================================
# PIXEL SIZE UTILITY
# ============================================================

def get_pixel_size_m(raster_path: str | Path) -> float:
    """Return pixel size in metres for a projected raster.

    Reads the geotransform and assumes square pixels and a CRS in metres.
    Raises ValueError if the pixel size appears to be in degrees (< 1).
    """
    ds = gdal.Open(str(raster_path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {raster_path}")
    gt = ds.GetGeoTransform()
    ds = None
    px = abs(gt[1])
    if px < 1:
        raise ValueError(
            f"Pixel size {px:.6f} appears to be in degrees. "
            "Reproject to a projected CRS before calling get_pixel_size_m()."
        )
    return px
