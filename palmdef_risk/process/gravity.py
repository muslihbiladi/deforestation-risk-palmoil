from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from osgeo import gdal
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)


def _apply_gaussian_filter(
    mill_raster: Path | str,
    out_path: Path | str,
    sigma_km: float,
    radius_km: float,
) -> None:
    """Gaussian kernel accessibility: convolution of mill density with Gaussian kernel."""
    ds = gdal.Open(str(mill_raster))
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    pixel_size_m = abs(gt[1])
    ds = None

    sigma_px = (sigma_km * 1000.0) / pixel_size_m
    truncate = radius_km * 1000.0 / (sigma_km * 1000.0)

    result = gaussian_filter(arr.astype(float), sigma=sigma_px,
                             truncate=truncate).astype(np.float32)

    ny, nx = result.shape
    out_ds = gdal.GetDriverByName("GTiff").Create(
        str(out_path), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(result)
    out_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    out_ds = None


def orthogonalize_gravity(
    gravity_path: Path | str,
    dist_road_path: Path | str,
    dist_town_path: Path | str,
    out_path: Path | str,
) -> float:
    """OLS: A_i ~ dist_road + dist_town. Residual → out_path. Returns R²."""
    def _load_flat(p):
        ds = gdal.Open(str(p))
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        nd = ds.GetRasterBand(1).GetNoDataValue()
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        shape = arr.shape
        ds = None
        return arr, nd, gt, proj, shape

    g_arr, g_nd, gt, proj, shape = _load_flat(gravity_path)
    r_arr, r_nd, *_ = _load_flat(dist_road_path)
    t_arr, t_nd, *_ = _load_flat(dist_town_path)

    mask = (g_arr != g_nd) & (r_arr != r_nd) & (t_arr != t_nd)
    g = g_arr[mask]
    r = r_arr[mask]
    t = t_arr[mask]

    X = np.column_stack([np.ones(len(g)), r, t])
    beta, *_ = np.linalg.lstsq(X, g, rcond=None)
    g_hat = X @ beta
    residual_flat = g - g_hat

    ss_res = np.sum(residual_flat ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if r2 > 0.85:
        logger.warning(
            "Gravity R²=%.3f > 0.85: accessibility largely collinear with infrastructure.",
            r2,
        )

    ny, nx = shape
    resid_arr = np.full(shape, -9999.0, dtype=np.float32)
    resid_arr[mask] = residual_flat.astype(np.float32)

    out_ds = gdal.GetDriverByName("GTiff").Create(
        str(out_path), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(resid_arr)
    out_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    out_ds = None
    logger.info("Gravity orthogonalized R²=%.3f; residual → %s", r2, out_path)
    return r2


def compute_gravity_accessibility(ctx: "RunContext") -> Path:
    """Rasterize mill_t2.gpkg, apply Gaussian filter → data/gravity_raw.tif."""
    from palmdef_risk.io.helpers import get_mask_properties, rasterize_vector
    d = ctx.data_dir
    ref = d / "forest_t2.tif"
    mill_gpkg = ctx.raw_dir / "mill" / "mill_t2.gpkg"

    mask_props = get_mask_properties(str(ref))
    mill_raster = d / "_mill_density_tmp.tif"
    rasterize_vector(str(mill_gpkg), str(mill_raster), burn_value=1,
                     mask_props=mask_props)

    out = d / "gravity_raw.tif"
    _apply_gaussian_filter(mill_raster, out,
                           sigma_km=ctx.config.sigma_km,
                           radius_km=ctx.config.radius_km)
    mill_raster.unlink(missing_ok=True)
    return out


def orthogonalize_gravity_ctx(ctx: "RunContext") -> Path:
    """Run orthogonalize_gravity using run context paths."""
    d = ctx.data_dir
    out = d / "gravity_resid.tif"
    orthogonalize_gravity(d / "gravity_raw.tif", d / "dist_road.tif",
                          d / "dist_town.tif", out)
    return out
