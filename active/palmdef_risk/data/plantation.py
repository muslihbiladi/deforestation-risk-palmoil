"""plantation: Download oil-palm plantation rasters from the Descals dataset.

Source: Descals et al., "Global oil palm extent and planting year 1990-2021"
(Zenodo 10.5281/zenodo.13379129). Two global layers:

    - OP-extent (10 m) : classes [0] non-palm, [1] industrial, [2] smallholder
    - OP-YoP    (30 m) : per-pixel planting year (1990-2021)

This module provides the "download" source for plantation data (the alternative
to user-supplied rasters). The global dataset is downloaded once into a shared,
AOI-independent cache (cache/plantation_global/) and reused across runs; only the
per-run AOI clip is recomputed.

Accumulation semantics: plantation at a given year Y is the cumulative extent of
all pixels planted from the first year of planting (1990) through Y, i.e.
    class = extent  where (1990 <= YoP <= Y)
            0       otherwise
A year > 2021 is clamped to 2021 (the dataset's final year) with a caution notice.

Outputs match the GHSL format produced elsewhere in Stage 1: one GeoTIFF per
year label (plantation_t2.tif / plantation_t3.tif), Byte, NoData 255, reprojected
to the run CRS at native resolution. Alignment to forest_t2.tif happens in Stage 2.
"""

from __future__ import annotations

import os
import glob
import json
import time
import zipfile
import logging
from pathlib import Path

import requests
import numpy as np
from osgeo import gdal

from palmdef_risk.data.variables import _parse_aoi, _clip_to_vector, _reproject_raster
from palmdef_risk.constants import NODATA_BYTE

gdal.UseExceptions()
log = logging.getLogger(__name__)

# ── Dataset constants ────────────────────────────────────────────────────────
_ZENODO_RECORD = "13379129"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"
_EXTENT_ZIP = "GlobalOilPalm_OP-extent.zip"
_YOP_ZIP = "GlobalOilPalm_OP-YoP.zip"
_YOP_MIN = 1990   # first planting year in the dataset
_YOP_MAX = 2021   # last planting year in the dataset
_UA = "palmdef_risk/1.0 (deforestation risk research; +https://www.wri.org)"


# ============================================================
# Accumulation core (pure numpy — unit-tested without network)
# ============================================================

def accumulate_classes(extent, yop, year, industrial_value=1,
                       smallholder_value=2, yop_min=_YOP_MIN, yop_max=_YOP_MAX,
                       nodata=NODATA_BYTE):
    """Cumulative plantation class array up to `year`.

    Keep the extent class (industrial/smallholder) only where the pixel was
    planted in [yop_min, min(year, yop_max)]; everything else becomes 0.
    Descals class 1 -> industrial_value, class 2 -> smallholder_value.

    :param extent: 2-D array of Descals extent classes (0/1/2).
    :param yop: 2-D array of planting years (same shape as extent).
    :param year: target accumulation cutoff year.
    :return: (out, eff_year, clamped) where out is a Byte ndarray.
    """
    extent = np.asarray(extent)
    yop = np.asarray(yop)
    clamped = year > yop_max
    eff_year = min(int(year), yop_max)

    planted = (yop >= yop_min) & (yop <= eff_year)
    out = np.zeros(extent.shape, dtype=np.uint8)
    out[planted & (extent == 1)] = industrial_value
    out[planted & (extent == 2)] = smallholder_value
    return out, eff_year, clamped


# ============================================================
# Global cache (download + extract once, shared across runs)
# ============================================================

def _download_file(url, dest, max_retries=3, verbose=True):
    """Stream a (large) file to disk with retries. Resumes nothing — atomic via .part."""
    part = str(dest) + ".part"
    for attempt in range(1, max_retries + 1):
        try:
            if verbose:
                print(f"  Downloading {os.path.basename(str(dest))} "
                      f"(attempt {attempt}/{max_retries})...")
            with requests.get(url, headers={"User-Agent": _UA},
                              stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            done += len(chunk)
            if total and done < total:
                raise IOError(f"incomplete download: {done}/{total} bytes")
            os.replace(part, dest)
            if verbose:
                print(f"    done ({done / 1e6:.1f} MB)")
            return dest
        except Exception as e:
            if verbose:
                print(f"    FAILED: {e}")
            if os.path.exists(part):
                os.remove(part)
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Could not download {url} after {max_retries} attempts")


def _extract_zip(zip_path, dest_dir, verbose=True):
    """Extract a zip into dest_dir; return the list of extracted .tif paths."""
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    tifs = sorted(glob.glob(os.path.join(dest_dir, "**", "*.tif"), recursive=True))
    if verbose:
        print(f"    extracted {len(tifs)} GeoTIFF tiles -> {dest_dir}")
    if not tifs:
        raise RuntimeError(f"No GeoTIFF tiles found after extracting {zip_path}")
    return tifs


def ensure_descals_cache(cache_dir, verbose=True) -> dict:
    """Make sure the global Descals dataset is present in the shared cache.

    Availability check first: if extent.vrt + yop.vrt + metadata.json already
    exist, nothing is downloaded. Otherwise both Zenodo zips are downloaded,
    extracted, and mosaicked into VRTs. Idempotent — safe to call every run.

    :return: {"extent_vrt": Path, "yop_vrt": Path}
    """
    from palmdef_risk.cache import CacheManager

    cm = CacheManager(cache_dir)
    gdir = cm.plantation_global_dir()
    extent_vrt = gdir / "extent.vrt"
    yop_vrt = gdir / "yop.vrt"

    if cm.plantation_global_valid():
        if verbose:
            print(f"Plantation (Descals): global cache hit -> {gdir}")
        return {"extent_vrt": extent_vrt, "yop_vrt": yop_vrt}

    if verbose:
        print("=" * 60)
        print("Plantation (Descals): global cache miss — downloading from Zenodo")
        print(f"  Record: 10.5281/zenodo.{_ZENODO_RECORD}")
    gdir.mkdir(parents=True, exist_ok=True)

    for zip_name, sub, vrt in (
        (_EXTENT_ZIP, "extent", extent_vrt),
        (_YOP_ZIP, "yop", yop_vrt),
    ):
        sub_dir = gdir / sub
        zip_path = gdir / zip_name
        tifs = sorted(glob.glob(str(sub_dir / "**" / "*.tif"), recursive=True))
        if not tifs:
            _download_file(f"{_ZENODO_BASE}/{zip_name}?download=1", zip_path,
                           verbose=verbose)
            tifs = _extract_zip(zip_path, sub_dir, verbose=verbose)
            if zip_path.exists():
                zip_path.unlink()
        vrt_ds = gdal.BuildVRT(str(vrt), tifs,
                               options=gdal.BuildVRTOptions(resolution="highest"))
        vrt_ds.FlushCache()
        vrt_ds = None
        if verbose:
            print(f"  built {vrt.name} from {len(tifs)} tiles")

    (gdir / "metadata.json").write_text(json.dumps({
        "record": _ZENODO_RECORD,
        "source": "Descals et al. Global oil palm extent and planting year 1990-2021",
        "yop_min": _YOP_MIN, "yop_max": _YOP_MAX,
    }, indent=2))
    if verbose:
        print(f"Plantation (Descals): global cache ready -> {gdir}")
    return {"extent_vrt": extent_vrt, "yop_vrt": yop_vrt}


# ============================================================
# Per-run AOI clip + accumulation
# ============================================================

def _clip_vrt_to_bbox(vrt_path, out_path, bbox, width=None, height=None,
                      resample="near"):
    """Warp a global VRT down to the AOI bbox (EPSG:4326).

    When width/height are given, warp onto exactly that grid (used to put YoP on
    the extent grid). Otherwise GDAL picks the native-resolution window.
    """
    resample_map = {"near": gdal.GRA_NearestNeighbour,
                    "bilinear": gdal.GRA_Bilinear}
    opts = dict(
        format="GTiff", outputBounds=list(bbox),
        resampleAlg=resample_map.get(resample, gdal.GRA_NearestNeighbour),
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    if width is not None and height is not None:
        opts["width"] = width
        opts["height"] = height
    gdal.Warp(out_path, str(vrt_path), options=gdal.WarpOptions(**opts))
    return out_path


def _build_year_raster(extent_vrt, yop_vrt, aoi, year, out_path,
                       buff=0.0, output_crs=None, industrial_value=1,
                       smallholder_value=2, verbose=True):
    """Clip + accumulate Descals layers for one year into a Byte GeoTIFF.

    Writes accumulation (NoData 255) at native ~10 m in EPSG:4326, optionally
    reprojected to output_crs (nearest, native resolution). Returns out_path.
    """
    out_path = str(out_path)
    work_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(work_dir, exist_ok=True)
    bbox = _parse_aoi(aoi, buff)

    # 1. Clip extent to AOI bbox at native resolution; this defines the grid.
    extent_clip = os.path.join(work_dir, "_extent_clip.tif")
    _clip_vrt_to_bbox(extent_vrt, extent_clip, bbox)

    eds = gdal.Open(extent_clip)
    gt = eds.GetGeoTransform()
    proj = eds.GetProjection()
    nx, ny = eds.RasterXSize, eds.RasterYSize
    extent_arr = eds.GetRasterBand(1).ReadAsArray()
    eds = None

    # 2. Warp YoP (30 m) onto the exact extent grid so arrays align cell-by-cell.
    yop_clip = os.path.join(work_dir, "_yop_clip.tif")
    _clip_vrt_to_bbox(yop_vrt, yop_clip, bbox, width=nx, height=ny)
    yds = gdal.Open(yop_clip)
    yop_arr = yds.GetRasterBand(1).ReadAsArray()
    yds = None

    # 3. Accumulate.
    out_arr, eff_year, clamped = accumulate_classes(
        extent_arr, yop_arr, year,
        industrial_value=industrial_value, smallholder_value=smallholder_value,
    )
    if clamped and verbose:
        print(f"  CAUTION: requested year {year} > {_YOP_MAX} (dataset max). "
              f"Using cumulative extent 1990-{_YOP_MAX}.")

    # 4. Write Byte raster (NoData 255) on the extent grid.
    acc_4326 = os.path.join(work_dir, "_acc_4326.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(acc_4326, nx, ny, 1, gdal.GDT_Byte,
                    ["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(out_arr)
    ds.GetRasterBand(1).SetNoDataValue(NODATA_BYTE)
    ds.FlushCache()
    ds = None

    # 5. Clip to AOI polygon when AOI is a vector (matches the other variables).
    aoi_is_vector = isinstance(aoi, (str, Path)) and os.path.isfile(str(aoi))
    src = acc_4326
    if aoi_is_vector:
        clipped = os.path.join(work_dir, "_acc_clip.tif")
        _clip_to_vector(acc_4326, clipped, str(aoi), nodata=NODATA_BYTE)
        src = clipped

    # 6. Reproject to run CRS (native resolution) or keep EPSG:4326.
    if output_crs is not None:
        _reproject_raster(src, out_path, dst_crs=output_crs,
                          resampling="near", nodata=NODATA_BYTE)
    else:
        os.replace(src, out_path)

    for tmp in (extent_clip, yop_clip, acc_4326,
                os.path.join(work_dir, "_acc_clip.tif")):
        if os.path.exists(tmp) and os.path.abspath(tmp) != os.path.abspath(out_path):
            os.remove(tmp)

    if verbose:
        print(f"  plantation ({year}, cum to {eff_year}) -> {out_path}")
    return out_path


def get_plantation_descals(aoi, years, cache_dir, output_dir, buff=0.0,
                           output_crs=None, industrial_value=1,
                           smallholder_value=2, verbose=True) -> dict:
    """Download + clip Descals plantation rasters for t2 and t3 years.

    Mirrors get_ghsl's signature/behaviour: one Byte GeoTIFF per label
    (plantation_t2.tif, plantation_t3.tif) in output_dir.

    :param years: [t2_year, t3_year] accumulation cutoffs.
    :return: {"plantation_t2": path, "plantation_t3": path}
    """
    if verbose:
        print("=" * 60)
        print("Downloading plantation (Descals Global Oil Palm)...")

    cache = ensure_descals_cache(cache_dir, verbose=verbose)
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for label, year in zip(["t2", "t3"], years):
        out_path = os.path.join(output_dir, f"plantation_{label}.tif")
        if verbose:
            print(f"\n  [{label}] accumulation year = {year}")
        _build_year_raster(
            cache["extent_vrt"], cache["yop_vrt"], aoi, int(year), out_path,
            buff=buff, output_crs=output_crs,
            industrial_value=industrial_value,
            smallholder_value=smallholder_value, verbose=verbose,
        )
        results[f"plantation_{label}"] = out_path
    return results
