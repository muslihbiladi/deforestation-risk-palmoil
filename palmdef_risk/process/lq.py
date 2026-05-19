"""LQ (Location Quotient) surface computation for palm oil supply chain analysis.

Phase 1 of the analytical pipeline:
  1.1 Prepare plantation surface P(i,j) via focal mean
  1.2 Compute mill surface M(i,j) via Gaussian KDE
  1.3 Compute LQ_MP and LQ_PM surfaces
  1.4 Compute LQ² surface
  1.5 Classify LQ zones (1-5)
"""
from __future__ import annotations
from pathlib import Path
import logging
import shutil

import numpy as np
from osgeo import gdal
from scipy.ndimage import uniform_filter
from scipy.stats import gaussian_kde

from palmdef_risk.io.run import RunContext
from palmdef_risk.io.helpers import get_pixel_size_m

log = logging.getLogger(__name__)

_ZONE_THRESHOLDS = [0.50, 0.80, 1.20, 1.50]


def run_lq_pipeline(ctx: RunContext) -> dict[str, Path]:
    """Run the full LQ pipeline for this run. Returns dict of output paths."""
    d = ctx.data_dir

    p_surface = compute_plantation_surface(
        d / "plantation.tif",
        d / "intermediate" / "P_surface.tif",
    )

    m_surface = compute_mill_kde(
        mill_gpkg=ctx.raw_dir / "mill" / "mill.gpkg",
        reference_raster=d / "forest_t2.tif",
        bandwidth_km=ctx.config.kde_bandwidth_km,
        output_path=d / "intermediate" / "M_surface.tif",
    )

    shutil.copy2(m_surface, d / "M.tif")
    shutil.copy2(p_surface, d / "P.tif")

    lq_mp, lq_pm = compute_lq(
        m_raster=m_surface,
        p_raster=p_surface,
        output_mp=d / "lq_mp.tif",
        output_pm=d / "lq_pm.tif",
        epsilon=ctx.config.lq_epsilon,
    )

    active_lq = lq_mp if ctx.config.lq_direction == "mp" else lq_pm
    shutil.copy2(active_lq, d / "lq.tif")
    log.info("Active LQ direction: %s -> lq.tif", ctx.config.lq_direction)

    lq_sq = compute_lq_squared(d / "lq.tif", d / "lq_sq.tif")
    lq_zones = classify_lq_zones(d / "lq.tif", d / "lq_zones.tif")

    return {
        "M": d / "M.tif",
        "P": d / "P.tif",
        "lq_mp": lq_mp,
        "lq_pm": lq_pm,
        "lq": d / "lq.tif",
        "lq_sq": lq_sq,
        "lq_zones": lq_zones,
    }


def compute_plantation_surface(
    plantation_raster: Path,
    output_path: Path,
    focal_radius_px: int = 33,
) -> Path:
    """Convert binary plantation raster to continuous proportion surface via focal mean."""
    ds = gdal.Open(str(plantation_raster))
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None

    invalid = (arr == nd) if nd is not None else np.zeros(arr.shape, dtype=bool)
    arr[invalid] = 0.0

    size = 2 * focal_radius_px + 1
    P = uniform_filter(arr.astype(np.float64), size=size).astype(np.float32)
    P[invalid] = -9999.0

    _write_float32_raster(output_path, P, gt, proj)
    log.info("  Plantation surface written: %s", output_path)
    return output_path


def compute_mill_kde(
    mill_gpkg: Path,
    reference_raster: Path,
    bandwidth_km: float,
    output_path: Path,
) -> Path:
    """Compute Gaussian KDE of mill locations aligned to reference raster grid."""
    import geopandas as gpd
    from osgeo import osr

    ds = gdal.Open(str(reference_raster))
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    proj = ds.GetProjection()
    nd_arr = ds.GetRasterBand(1).ReadAsArray() == 255
    ds = None

    ref_srs = osr.SpatialReference()
    ref_srs.ImportFromWkt(proj)
    epsg = ref_srs.GetAuthorityCode(None)
    mills = gpd.read_file(str(mill_gpkg))
    if epsg:
        mills = mills.to_crs(f"EPSG:{epsg}")

    xs = np.array(mills.geometry.x, dtype=np.float64)
    ys = np.array(mills.geometry.y, dtype=np.float64)

    px = gt[0] + (np.arange(nx) + 0.5) * gt[1]
    py = gt[3] + (np.arange(ny) + 0.5) * gt[5]
    grid_x, grid_y = np.meshgrid(px, py)

    if len(xs) < 2:
        kde_grid = np.zeros((ny, nx), dtype=np.float32)
        log.warning("  <2 mills in AOI -- KDE surface is all zeros")
    else:
        bw_m = bandwidth_km * 1000.0
        std_x = xs.std() if xs.std() > 0 else 1.0
        kde = gaussian_kde(np.vstack([xs, ys]), bw_method=bw_m / std_x)
        positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
        kde_grid = kde(positions).reshape(ny, nx).astype(np.float32)

    kde_grid[nd_arr] = -9999.0
    _write_float32_raster(output_path, kde_grid, gt, proj)
    log.info("  Mill KDE surface written: %s", output_path)
    return output_path


def compute_lq(
    m_raster: Path,
    p_raster: Path,
    output_mp: Path,
    output_pm: Path,
    epsilon: float = 0.001,
) -> tuple[Path, Path]:
    """Compute LQ_MP and LQ_PM surfaces.

    LQ_MP(i,j) = [M(i,j)/P(i,j)] / [M_global/P_global]
    LQ_PM(i,j) = [P(i,j)/M(i,j)] / [P_global/M_global]
    """
    ds_m = gdal.Open(str(m_raster))
    ds_p = gdal.Open(str(p_raster))
    M = ds_m.GetRasterBand(1).ReadAsArray().astype(np.float32)
    P = ds_p.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nd_m = ds_m.GetRasterBand(1).GetNoDataValue()
    nd_p = ds_p.GetRasterBand(1).GetNoDataValue()
    gt = ds_m.GetGeoTransform()
    proj = ds_m.GetProjection()
    ds_m = None
    ds_p = None

    invalid = np.zeros(M.shape, dtype=bool)
    if nd_m is not None:
        invalid |= M == nd_m
    if nd_p is not None:
        invalid |= P == nd_p

    M_v = M.copy()
    P_v = P.copy()
    M_v[invalid] = 0.0
    P_v[invalid] = 0.0

    valid_mask = ~invalid
    M_global = float(M_v[valid_mask].mean()) if valid_mask.any() else 1.0
    P_global = float(P_v[valid_mask].mean()) if valid_mask.any() else 1.0

    eps_m = epsilon * max(M_global, 1e-10)
    eps_p = epsilon * max(P_global, 1e-10)

    LQ_MP = (M_v / (P_v + eps_p)) / (M_global / (P_global + eps_p))
    LQ_PM = (P_v / (M_v + eps_m)) / (P_global / (M_global + eps_m))

    LQ_MP[invalid] = -9999.0
    LQ_PM[invalid] = -9999.0

    _write_float32_raster(output_mp, LQ_MP, gt, proj)
    _write_float32_raster(output_pm, LQ_PM, gt, proj)
    log.info("  LQ_MP written: %s", output_mp)
    log.info("  LQ_PM written: %s", output_pm)
    return output_mp, output_pm


def compute_lq_squared(lq_raster: Path, output_path: Path) -> Path:
    """Compute LQ^2 = LQ ** 2, preserving NoData."""
    ds = gdal.Open(str(lq_raster))
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None

    invalid = (arr == nd) if nd is not None else np.zeros(arr.shape, dtype=bool)
    sq = arr ** 2
    sq[invalid] = -9999.0
    _write_float32_raster(output_path, sq, gt, proj)
    return output_path


def classify_lq_zones(lq_raster: Path, output_path: Path) -> Path:
    """Classify LQ into 5 zones: 1=critical gap ... 5=overcapacity."""
    ds = gdal.Open(str(lq_raster))
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ny, nx = arr.shape
    ds = None

    invalid = (arr == nd) if nd is not None else np.zeros(arr.shape, dtype=bool)
    zones = np.ones((ny, nx), dtype=np.uint8)

    thresholds = [0.50, 0.80, 1.20, 1.50]
    for cls, thr in enumerate(thresholds, start=2):
        zones[arr >= thr] = cls

    zones[invalid] = 255

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(str(output_path), nx, ny, 1, gdal.GDT_Byte,
                           ["COMPRESS=DEFLATE", "TILED=YES"])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(zones)
    out_ds.GetRasterBand(1).SetNoDataValue(255)
    out_ds.FlushCache()
    out_ds = None
    return output_path


def _write_float32_raster(path: Path, arr: np.ndarray, gt: tuple, proj: str) -> None:
    ny, nx = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), nx, ny, 1, gdal.GDT_Float32,
                       ["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
    ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    ds.FlushCache()
    ds = None
