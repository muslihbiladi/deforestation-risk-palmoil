from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from osgeo import gdal

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

from palmdef_risk.constants import NODATA_FLOAT

logger = logging.getLogger(__name__)


def _load_flat(p):
    ds = gdal.Open(str(p))
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    shape = arr.shape
    ds = None
    return arr, nd, gt, proj, shape


def orthogonalize_plantation(
    dist_plant_path: Path | str,
    dist_edge_path: Path | str,
    dist_defor_path: Path | str,
    dist_road_path: Path | str,
    out_path: Path | str,
) -> float:
    """OLS: log(dist_plant+1) ~ log(dist_edge+1)+log(dist_defor+1)+log(dist_road+1).

    Writes the residual to out_path (Float32, NoData -9999). Returns R².
    The residual is the `plantation_resid` covariate (already in log space — it must
    NOT be re-logged downstream).
    """
    p_arr, p_nd, gt, proj, shape = _load_flat(dist_plant_path)
    e_arr, e_nd, *_ = _load_flat(dist_edge_path)
    f_arr, d_nd, *_ = _load_flat(dist_defor_path)
    r_arr, r_nd, *_ = _load_flat(dist_road_path)

    mask = (
        (p_arr != p_nd) & (e_arr != e_nd) & (f_arr != d_nd) & (r_arr != r_nd)
    )
    y = np.log(p_arr[mask] + 1.0)
    Xe = np.log(e_arr[mask] + 1.0)
    Xf = np.log(f_arr[mask] + 1.0)
    Xr = np.log(r_arr[mask] + 1.0)

    X = np.column_stack([np.ones(len(y)), Xe, Xf, Xr])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual_flat = y - X @ beta

    ss_res = np.sum(residual_flat ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r2 > 0.85:
        logger.warning(
            "Plantation R²=%.3f > 0.85: plantation proximity largely collinear "
            "with edge/defor/road.", r2,
        )

    ny, nx = shape
    resid_arr = np.full(shape, NODATA_FLOAT, dtype=np.float32)
    resid_arr[mask] = residual_flat.astype(np.float32)

    out_ds = gdal.GetDriverByName("GTiff").Create(
        str(out_path), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(resid_arr)
    out_ds.GetRasterBand(1).SetNoDataValue(NODATA_FLOAT)
    out_ds.FlushCache()
    out_ds = None
    logger.info("Plantation orthogonalized R²=%.3f; residual → %s", r2, out_path)
    return r2


def compute_plantation_resid(ctx: "RunContext", force: bool = False) -> float:
    """Compute model-period plantation_resid → data/plantation_resid.tif.

    Returns R²; 0.0 (skipped) when dist_plantation_edge.tif is absent.
    """
    d = ctx.data_dir
    dist_plant = d / "dist_plantation_edge.tif"
    out_resid = d / "plantation_resid.tif"
    if not dist_plant.exists():
        logger.warning("dist_plantation_edge.tif absent — skipping plantation_resid")
        return 0.0
    if out_resid.exists() and not force:
        logger.info("plantation_resid.tif exists — skipping")
        return 0.0
    regressors = [d / "dist_edge.tif", d / "dist_defor.tif", d / "dist_road.tif"]
    missing = [p.name for p in regressors if not p.exists()]
    if missing:
        logger.warning("plantation_resid skipped — missing regressor raster(s): %s", missing)
        return 0.0
    return orthogonalize_plantation(
        dist_plant, d / "dist_edge.tif", d / "dist_defor.tif",
        d / "dist_road.tif", out_resid,
    )


def compute_plantation_resid_forecast(ctx: "RunContext", force: bool = False) -> float:
    """Compute forecast (t3) plantation_resid → data/forecast/plantation_resid.tif.

    Orthogonalizes against t3 dist_edge/dist_defor (data/forecast/) and the static
    t2 dist_road (data/dist_road.tif). Returns R²; 0.0 when t3 plantation absent.
    """
    d = ctx.data_dir
    fcast = d / "forecast"
    fcast.mkdir(parents=True, exist_ok=True)
    dist_plant = fcast / "dist_plantation_edge.tif"
    out_resid = fcast / "plantation_resid.tif"
    if not dist_plant.exists():
        logger.warning("forecast/dist_plantation_edge.tif absent — skipping forecast plantation_resid")
        return 0.0
    if out_resid.exists() and not force:
        logger.info("forecast/plantation_resid.tif exists — skipping")
        return 0.0
    regressors = [fcast / "dist_edge.tif", fcast / "dist_defor.tif", d / "dist_road.tif"]
    missing = [p.name for p in regressors if not p.exists()]
    if missing:
        logger.warning("forecast plantation_resid skipped — missing regressor raster(s): %s", missing)
        return 0.0
    return orthogonalize_plantation(
        dist_plant, fcast / "dist_edge.tif", fcast / "dist_defor.tif",
        d / "dist_road.tif", out_resid,
    )
