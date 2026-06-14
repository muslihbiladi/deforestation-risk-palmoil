"""Alignment, rasterization, masking, and distance computation.

All rasters are aligned to forest_t2.tif (the reference raster).
Distances are computed via forestatrisk.data.compute.compute_distance().
"""
from __future__ import annotations
from pathlib import Path
import logging
import shutil

import numpy as np
from osgeo import gdal, gdalconst

from palmdef_risk.io.run import RunContext
from palmdef_risk.io.helpers import (
    get_mask_properties, reproject_raster, reproject_raster_to_match,
    reproject_vector, rasterize_vector, apply_mask, apply_mask_float,
    remove_if_exists, get_pixel_size_m,
)

log = logging.getLogger(__name__)

_PROTECTED_FILENAME = "protected"   # never "pa" — causes patsy formula errors


def _ensure_forest_utm(ctx: RunContext) -> None:
    """Reproject raw/forest/*.tif from EPSG:4326 to UTM in-place if needed.

    The GEE download writes EPSG:4326 when output_crs was None at download
    time. This self-heals that run by reprojecting before align copies files.
    The resulting shape change cascades to invalidate all downstream rasters
    (distances, gravity) via the existing shape-mismatch checks.
    """
    if not ctx.config.crs:
        return
    ref = ctx.raw_dir / "forest" / "forest_t2.tif"
    if not ref.exists():
        return
    ds = gdal.Open(str(ref))
    if ds is None:
        return
    gt = ds.GetGeoTransform()
    ds = None
    if abs(gt[1]) >= 1.0:
        return  # already in projected CRS (metres)

    log.warning(
        "Forest rasters are EPSG:4326 — reprojecting to %s in-place",
        ctx.config.crs,
    )
    forest_dir = ctx.raw_dir / "forest"
    for fname in [
        "forest_cover.tif", "forest_t1.tif", "forest_t2.tif", "forest_t3.tif",
        "fcc12.tif", "fcc23.tif", "fcc123.tif",
    ]:
        src = forest_dir / fname
        if not src.exists():
            continue
        tmp = forest_dir / f"{src.stem}_4326_tmp.tif"
        src.rename(tmp)
        reproject_raster(str(tmp), str(src), target_crs=ctx.config.crs,
                         resample_alg="near")
        tmp.unlink()
        log.info("  Reprojected raw forest: %s → %s", fname, ctx.config.crs)


def _remap_srtm_voids(path: str, nodata: float = -9999.0) -> None:
    """Replace SRTM void sentinel (-32768) that leaks through bilinear reprojection."""
    ds = gdal.Open(path, gdal.GA_Update)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    arr[arr < -9000] = nodata  # -32768 and any near-sentinel bleed from bilinear
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds.FlushCache()
    ds = None


def _discover_inputs(ctx: RunContext) -> dict:
    """Build an inputs dict by scanning ctx.raw_dir when no dict is passed."""
    ui = ctx.raw_dir / "user_inputs"
    d: dict = {}
    for name in ("peatland", "hgu", "river"):
        for ext in (".gpkg", ".tif", ".shp"):
            p = ui / f"{name}{ext}"
            if p.exists():
                d[name] = p
                break
    # plantation may come from user_inputs/ (source=user) or variables/ (source=download)
    variables = ctx.raw_dir / "variables"
    for name in ("plantation_t2", "plantation_t3"):
        for base in (ui, variables):
            for ext in (".tif", ".gpkg"):
                p = base / f"{name}{ext}"
                if p.exists():
                    d[name] = p
                    break
            if name in d:
                break
    mill = ctx.raw_dir / "mill" / "mill_t2.gpkg"
    if mill.exists():
        d["mill"] = mill
    return d


def align_all(ctx: RunContext, inputs: dict | None = None, force: bool = False) -> dict[str, Path]:
    """Align all raw inputs to the reference raster (forest_t2.tif).

    `inputs` is the merged dict from download_forest(), download_variables(),
    download_mill(), and ingest_user_inputs(). When omitted, files are
    auto-discovered from ctx.raw_dir (useful when resuming in notebook 02).

    Skips any output that already exists unless force=True.

    Returns dict mapping variable names to aligned raster paths in ctx.data_dir.
    """
    def _raster_shape(p: Path):
        ds = gdal.Open(str(p))
        if ds is None:
            return None
        s = (ds.RasterYSize, ds.RasterXSize)
        ds = None
        return s

    def _skip(p: Path, src: Path | None = None) -> bool:
        if not force and p.exists():
            if src is not None and src.exists():
                if _raster_shape(p) != _raster_shape(src):
                    log.warning(
                        "  %s shape mismatch vs source — overwriting", p.name)
                    p.unlink()
                    return False
            log.info("  skip (exists): %s", p.name)
            return True
        return False

    _ensure_forest_utm(ctx)

    # After forest reprojection to UTM, purge any data-dir rasters that are
    # still in EPSG:4326 (pixel < 1°) or have wrong shape. This triggers
    # re-alignment for altitude, slope, road, etc. whose raw source is also
    # in 4326 (so the shape-mismatch check against source would not catch them).
    _ref_path = ctx.raw_dir / "forest" / "forest_t2.tif"
    _ref_ds = gdal.Open(str(_ref_path))
    if _ref_ds is not None:
        _ref_gt = _ref_ds.GetGeoTransform()
        _ref_shape = (_ref_ds.RasterYSize, _ref_ds.RasterXSize)
        _ref_ds = None
        if abs(_ref_gt[1]) >= 1.0:  # reference is UTM (metres)
            for _tif in list(ctx.data_dir.glob("*.tif")):
                try:
                    _ds = gdal.Open(str(_tif))
                except Exception:
                    _ds = None
                if _ds is None:
                    log.warning("Deleting corrupt raster: %s", _tif.name)
                    _tif.unlink()
                    continue
                _gt = _ds.GetGeoTransform()
                _shape = (_ds.RasterYSize, _ds.RasterXSize)
                _ds = None
                if _shape != _ref_shape or abs(_gt[1]) < 1.0:
                    log.info("Deleting stale raster (shape/CRS mismatch): %s", _tif.name)
                    _tif.unlink()

    if inputs is None or isinstance(inputs, Path):
        inputs = _discover_inputs(ctx)
    raw_forest = ctx.raw_dir / "forest"
    ref_file = raw_forest / "forest_t2.tif"
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference raster not found: {ref_file}")

    (ctx.data_dir / "intermediate").mkdir(parents=True, exist_ok=True)

    mask_props = get_mask_properties(str(ref_file))
    result: dict[str, Path] = {}

    # 1. Copy forest rasters (already in final CRS from download, just move to data_dir)
    for name in ["forest_t1", "forest_t2", "forest_t3", "fcc12", "fcc23", "fcc123"]:
        src = raw_forest / f"{name}.tif"
        if src.exists():
            dst = ctx.data_dir / f"{name}.tif"
            if not _skip(dst, src=src):
                shutil.copy2(src, dst)
            result[name] = dst

    # 2. Align SRTM (altitude + slope) — Float32
    for name in ["altitude", "slope"]:
        raw_path = ctx.raw_dir / "variables" / f"{name}.tif"
        if raw_path.exists():
            out = ctx.data_dir / f"{name}.tif"
            if not _skip(out):
                reproject_raster_to_match(str(raw_path), str(out), mask_props,
                                          resample_alg="bilinear")
                apply_mask_float(str(out), mask_props["ref_path"])
                if name == "altitude":
                    _remap_srtm_voids(str(out))
            result[name] = out

    # 3. Rasterize vectors — Byte
    vec_dir = ctx.raw_dir / "variables"
    for name, burn in [(_PROTECTED_FILENAME, 1), ("road", 1), ("river", 1), ("town", 1)]:
        # User-supplied river overrides the OSM download
        vec_path = (inputs.get("river") or _find_vector(vec_dir, name)) if name == "river" \
            else _find_vector(vec_dir, name)
        if vec_path:
            out = ctx.data_dir / f"{name}.tif"
            if not _skip(out):
                proj_vec = ctx.data_dir / "intermediate" / f"{name}_proj.gpkg"
                vec_to_burn = reproject_vector(str(vec_path), str(proj_vec), mask_props["srs"])
                rasterize_vector(str(vec_to_burn), str(out), burn, mask_props)
                apply_mask(str(out), mask_props["ref_path"])
            result[name] = out

    # 4. Peatland — branch on type
    peat_src = inputs.get("peatland")
    if peat_src:
        out = ctx.data_dir / "peatland.tif"
        if not _skip(out):
            if ctx.config.peatland_type == "binary":
                proj_peat = ctx.data_dir / "intermediate" / "peatland_proj.gpkg"
                peat_to_burn = reproject_vector(str(peat_src), str(proj_peat), mask_props["srs"])
                rasterize_vector(str(peat_to_burn), str(out), 1, mask_props)
                apply_mask(str(out), mask_props["ref_path"])
            else:
                reproject_raster_to_match(str(peat_src), str(out), mask_props,
                                          resample_alg="bilinear")
                apply_mask_float(str(out), mask_props["ref_path"])
        result["peatland"] = out

    # 5. HGU signed-distance raster
    hgu_src = inputs.get("hgu")
    if hgu_src:
        out = ctx.data_dir / "hgu_signed_dist.tif"
        if not _skip(out):
            compute_hgu_signed_distance(
                hgu_gpkg=str(hgu_src),
                ref_tif=str(ref_file),
                out_tif=str(out),
            )
        result["hgu_signed_dist"] = out

    # 6. Plantation — merge two classes → single presence raster.
    #    t2 → plantation.tif (model period); t3 → plantation_t3.tif (forecast).
    for src_key, out_name, res_key in (
        ("plantation_t2", "plantation.tif", "plantation"),
        ("plantation_t3", "plantation_t3.tif", "plantation_t3"),
    ):
        plant_src = inputs.get(src_key)
        if not plant_src:
            continue
        out = ctx.data_dir / out_name
        if not _skip(out):
            merged = ctx.data_dir / "intermediate" / f"{src_key}_merged.tif"
            merge_plantation(str(plant_src), str(merged),
                             ctx.config.plantation_industrial_value,
                             ctx.config.plantation_smallholder_value)
            reproject_raster_to_match(str(merged), str(out), mask_props,
                                      resample_alg="near",
                                      output_dtype=gdalconst.GDT_Byte)
            apply_mask(str(out), mask_props["ref_path"])
        result[res_key] = out

    # 7. Mill — rasterize presence raster
    mill_gpkg = inputs.get("mill")
    if mill_gpkg:
        out = ctx.data_dir / "mill.tif"
        if not _skip(out):
            proj_mill = ctx.data_dir / "intermediate" / "mill_proj.gpkg"
            mill_to_burn = reproject_vector(str(mill_gpkg), str(proj_mill), mask_props["srs"])
            rasterize_vector(str(mill_to_burn), str(out), 1, mask_props)
            apply_mask(str(out), mask_props["ref_path"])
        result["mill"] = out

    return result


def merge_plantation(
    src_path: str | Path,
    dst_path: str | Path,
    industrial_value: int,
    smallholder_value: int,
) -> Path:
    """Merge two-class plantation raster into single binary presence raster.

    Pixels equal to industrial_value OR smallholder_value become 1 (present).
    All other pixels become 0. NoData (255) pixels are preserved.
    """
    ds = gdal.Open(str(src_path))
    arr = ds.GetRasterBand(1).ReadAsArray()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None

    nodata_mask = (arr == int(nd)) if nd is not None else np.zeros(arr.shape, dtype=bool)
    merged = np.where(
        (arr == industrial_value) | (arr == smallholder_value), 1, 0
    ).astype(np.uint8)
    merged[nodata_mask] = 255

    ny, nx = merged.shape
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(str(dst_path), nx, ny, 1, gdal.GDT_Byte,
                           ["COMPRESS=DEFLATE", "TILED=YES"])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(merged)
    out_ds.GetRasterBand(1).SetNoDataValue(255)
    out_ds.FlushCache()
    out_ds = None
    return Path(dst_path)


def compute_hgu_signed_distance(
    hgu_gpkg: str,
    ref_tif: str,
    out_tif: str,
) -> None:
    """Write signed distance to HGU boundary: negative inside, positive outside."""
    from palmdef_risk.io.helpers import get_mask_properties, rasterize_vector

    mask_props = get_mask_properties(ref_tif)
    hgu_mask_path = Path(out_tif).parent / "_hgu_mask_tmp.tif"
    rasterize_vector(hgu_gpkg, str(hgu_mask_path), burn_value=1, mask_props=mask_props)

    ds_mask = gdal.Open(str(hgu_mask_path))
    inside = ds_mask.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    gt = ds_mask.GetGeoTransform()
    proj = ds_mask.GetProjection()
    ny, nx = inside.shape
    ds_mask = None

    def _proximity(arr: np.ndarray) -> np.ndarray:
        drv = gdal.GetDriverByName("MEM")
        src_ds = drv.Create("", nx, ny, 1, gdal.GDT_Byte)
        src_ds.SetGeoTransform(gt)
        src_ds.SetProjection(proj)
        src_ds.GetRasterBand(1).WriteArray(arr)
        out_ds = drv.Create("", nx, ny, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        gdal.ComputeProximity(
            src_ds.GetRasterBand(1),
            out_ds.GetRasterBand(1),
            options=["DISTUNITS=GEO"],
        )
        return out_ds.GetRasterBand(1).ReadAsArray()

    dist_from_inside = _proximity(inside)
    outside = (1 - inside).astype(np.uint8)
    dist_from_outside = _proximity(outside)
    signed = dist_from_inside.astype(np.float32) - dist_from_outside.astype(np.float32)

    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(
        str(out_tif), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(signed)
    out_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    out_ds = None
    Path(str(hgu_mask_path)).unlink(missing_ok=True)


def _find_vector(directory: Path, name: str) -> Path | None:
    for ext in (".gpkg", ".shp", ".geojson"):
        p = directory / f"{name}{ext}"
        if p.exists():
            return p
    return None
