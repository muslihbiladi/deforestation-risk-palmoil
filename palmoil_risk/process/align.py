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

import forestatrisk as far
from palmoil_risk.io.run import RunContext
from palmoil_risk.io.helpers import (
    get_mask_properties, reproject_raster_to_match, reproject_vector,
    rasterize_vector, apply_mask, apply_mask_float, remove_if_exists,
    get_pixel_size_m,
)

log = logging.getLogger(__name__)


def align_all(ctx: RunContext, inputs: dict) -> dict[str, Path]:
    """Align all raw inputs to the reference raster (forest_t2.tif).

    `inputs` is the merged dict from download_forest(), download_variables(),
    download_mill(), and ingest_user_inputs().

    Returns dict mapping variable names to aligned raster paths in ctx.data_dir.
    """
    raw_forest = ctx.raw_dir / "forest"
    ref_file = raw_forest / "forest_t2.tif"
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference raster not found: {ref_file}")

    mask_props = get_mask_properties(str(ref_file))
    result: dict[str, Path] = {}

    # 1. Copy forest rasters (already in final CRS from download, just move to data_dir)
    for name in ["forest_t1", "forest_t2", "forest_t3", "fcc12", "fcc23", "fcc123"]:
        src = raw_forest / f"{name}.tif"
        if src.exists():
            dst = ctx.data_dir / f"{name}.tif"
            shutil.copy2(src, dst)
            result[name] = dst

    # 2. Align SRTM (altitude + slope) — Float32
    for name in ["altitude", "slope"]:
        raw_path = ctx.raw_dir / "variables" / f"{name}.tif"
        if raw_path.exists():
            out = ctx.data_dir / f"{name}.tif"
            reproject_raster_to_match(str(raw_path), str(out), mask_props,
                                      resample_alg="bilinear")
            apply_mask_float(str(out), mask_props["invalid_mask"])
            result[name] = out

    # 3. Rasterize vectors — Byte
    vec_dir = ctx.raw_dir / "variables"
    for name, burn in [("pa", 1), ("road", 1), ("river", 1), ("town", 1)]:
        vec_path = _find_vector(vec_dir, name)
        if vec_path:
            proj_vec = ctx.data_dir / "intermediate" / f"{name}_proj.gpkg"
            reproject_vector(str(vec_path), str(proj_vec), mask_props["srs"])
            out = ctx.data_dir / f"{name}.tif"
            rasterize_vector(str(proj_vec), str(out), burn, mask_props)
            apply_mask(str(out), mask_props["invalid_mask"])
            result[name] = out

    # 4. Peatland — branch on type
    peat_src = inputs.get("peatland")
    if peat_src:
        out = ctx.data_dir / "peatland.tif"
        if ctx.config.peatland_type == "binary":
            proj_peat = ctx.data_dir / "intermediate" / "peatland_proj.gpkg"
            reproject_vector(str(peat_src), str(proj_peat), mask_props["srs"])
            rasterize_vector(str(proj_peat), str(out), 1, mask_props)
            apply_mask(str(out), mask_props["invalid_mask"])
        else:
            reproject_raster_to_match(str(peat_src), str(out), mask_props,
                                      resample_alg="bilinear")
            apply_mask_float(str(out), mask_props["invalid_mask"])
        result["peatland"] = out

    # 5. HGU — vector → raster
    hgu_src = inputs.get("hgu")
    if hgu_src:
        proj_hgu = ctx.data_dir / "intermediate" / "hgu_proj.gpkg"
        reproject_vector(str(hgu_src), str(proj_hgu), mask_props["srs"])
        out = ctx.data_dir / "hgu.tif"
        rasterize_vector(str(proj_hgu), str(out), 1, mask_props)
        apply_mask(str(out), mask_props["invalid_mask"])
        result["hgu"] = out

    # 6. Plantation — merge two classes → single presence raster
    plant_t2 = inputs.get("plantation_t2")
    if plant_t2:
        merged = ctx.data_dir / "intermediate" / "plantation_merged.tif"
        merge_plantation(str(plant_t2), str(merged),
                         ctx.config.plantation_industrial_value,
                         ctx.config.plantation_smallholder_value)
        out = ctx.data_dir / "plantation.tif"
        reproject_raster_to_match(str(merged), str(out), mask_props,
                                  resample_alg="near",
                                  output_dtype=gdalconst.GDT_Byte)
        apply_mask(str(out), mask_props["invalid_mask"])
        result["plantation"] = out

    # 7. Mill — rasterize presence raster
    mill_gpkg = inputs.get("mill")
    if mill_gpkg:
        proj_mill = ctx.data_dir / "intermediate" / "mill_proj.gpkg"
        reproject_vector(str(mill_gpkg), str(proj_mill), mask_props["srs"])
        out = ctx.data_dir / "mill.tif"
        rasterize_vector(str(proj_mill), str(out), 1, mask_props)
        apply_mask(str(out), mask_props["invalid_mask"])
        result["mill"] = out

    return result


def compute_all_distances(ctx: RunContext) -> dict[str, Path]:
    """Compute all distance rasters via forestatrisk.data.compute.compute_distance().

    Reads aligned rasters from ctx.data_dir. Writes dist_*.tif back to ctx.data_dir.
    """
    d = ctx.data_dir
    result: dict[str, Path] = {}

    def _dist(input_file: Path, dist_file: Path, values: int = 0) -> Path:
        log.info("  Computing distance: %s", dist_file.name)
        far.data.compute.compute_distance(
            input_file=str(input_file),
            dist_file=str(dist_file),
            values=values,
            input_nodata=True,
            verbose=False,
        )
        return dist_file

    if (d / "forest_t2.tif").exists():
        result["dist_edge"] = _dist(d / "forest_t2.tif", d / "dist_edge.tif")
    if (d / "fcc12.tif").exists():
        result["dist_defor"] = _dist(d / "fcc12.tif", d / "dist_defor.tif")
    if (d / "road.tif").exists():
        result["dist_road"] = _dist(d / "road.tif", d / "dist_road.tif")
    if (d / "river.tif").exists():
        result["dist_river"] = _dist(d / "river.tif", d / "dist_river.tif")
    if (d / "town.tif").exists():
        result["dist_town"] = _dist(d / "town.tif", d / "dist_town.tif")
    elif (d / "pa.tif").exists():
        result["dist_town"] = _dist(d / "pa.tif", d / "dist_town.tif")
    if (d / "mill.tif").exists():
        result["dist_mill"] = _dist(d / "mill.tif", d / "dist_mill.tif")
    if (d / "plantation.tif").exists():
        result["dist_plantation_edge"] = _dist(
            d / "plantation.tif", d / "dist_plantation_edge.tif")

    if (d / "forest_t3.tif").exists():
        result["dist_edge_forecast"] = _dist(
            d / "forest_t3.tif", d / "dist_edge_forecast.tif")
    if (d / "fcc23.tif").exists():
        result["dist_defor_forecast"] = _dist(
            d / "fcc23.tif", d / "dist_defor_forecast.tif")

    plant_t3_aligned = ctx.data_dir / "plantation_t3.tif"
    if plant_t3_aligned.exists():
        result["dist_plantation_edge_forecast"] = _dist(
            plant_t3_aligned, d / "dist_plantation_edge_forecast.tif")
    else:
        log.info("  dist_plantation_edge_forecast skipped (no plantation_t3)")

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


def _find_vector(directory: Path, name: str) -> Path | None:
    for ext in (".gpkg", ".shp", ".geojson"):
        p = directory / f"{name}{ext}"
        if p.exists():
            return p
    return None
