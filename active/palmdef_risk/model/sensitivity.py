from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from osgeo import gdal

from palmdef_risk.process.gravity import orthogonalize_gravity_ctx
from palmdef_risk.model.icar import _build_and_fit

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)


def compute_gravity_raw(ctx: "RunContext", sigma_km: float) -> Path:
    """Compute gravity_raw.tif at a given sigma (overwrites ctx.data_dir/gravity_raw.tif).

    Reprojects the mill GPKG to the reference raster's CRS before burning, so the
    rasterization step doesn't silently produce zero burns when the cached mill
    file's CRS differs from the run's UTM grid. Matches the production path in
    `process.gravity._compute_gravity_for_period`.
    """
    from palmdef_risk.process.gravity import _apply_gaussian_filter, _rasterize_points_numpy
    from palmdef_risk.io.helpers import get_mask_properties, reproject_vector
    d = ctx.data_dir
    ref = d / "forest_t2.tif"
    mill_gpkg = ctx.raw_dir / "mill" / "mill_t2.gpkg"

    mask_props = get_mask_properties(str(ref))
    proj_mill = d / "_mill_proj_sensitivity.gpkg"
    mill_src = Path(reproject_vector(str(mill_gpkg), str(proj_mill), mask_props["srs"]))

    tmp = d / "_mill_density_sensitivity_tmp.tif"
    n_burned = _rasterize_points_numpy(mill_src, ref, tmp)
    if mill_src == proj_mill:
        proj_mill.unlink(missing_ok=True)

    if n_burned == 0:
        logger.error("No mill points burned for sigma=%.0f km", sigma_km)
        tmp.unlink(missing_ok=True)
        return d / "gravity_raw.tif"
    out = d / "gravity_raw.tif"
    _apply_gaussian_filter(tmp, out, sigma_km=sigma_km, radius_km=ctx.config.radius_km)
    tmp.unlink(missing_ok=True)
    return out


def _resample_gravity_resid(grav_resid_path: Path, df_xy: pd.DataFrame) -> np.ndarray:
    """Look up gravity_resid raster values at the sample's (X, Y) UTM coords.

    Returns a float64 array aligned with df_xy. NoData → NaN, out-of-bounds → NaN.
    """
    ds = gdal.Open(str(grav_resid_path))
    arr = ds.GetRasterBand(1).ReadAsArray()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ds = None

    col = ((df_xy["X"].values - gt[0]) / gt[1]).astype(np.int64)
    row = ((df_xy["Y"].values - gt[3]) / gt[5]).astype(np.int64)
    in_bounds = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)

    out = np.full(len(df_xy), np.nan, dtype=np.float64)
    vals = arr[row[in_bounds], col[in_bounds]].astype(np.float64)
    if nd is not None:
        vals[vals == nd] = np.nan
    out[in_bounds] = vals
    return out


def run_gravity_sensitivity(ctx: "RunContext") -> Path:
    """
    For each sigma in config.sensitivity_sigmas: rebuild gravity_resid at that
    bandwidth, re-sample gravity_resid into the training sample's coordinates,
    refit Model B in-memory (no pickle clobber), and record accessibility
    coefficient + mean deviance. Writes gravity_sensitivity.json.
    """
    out_dir = ctx.output_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "gravity_sensitivity.json"

    d = ctx.data_dir
    backup_raw = d / "_gravity_raw_backup.tif"
    backup_resid = d / "_gravity_resid_backup.tif"
    if (d / "gravity_raw.tif").exists():
        shutil.copy2(d / "gravity_raw.tif", backup_raw)
    if (d / "gravity_resid.tif").exists():
        shutil.copy2(d / "gravity_resid.tif", backup_resid)

    sample = pd.read_csv(ctx.output_dir / "sample.csv")
    if "gravity_resid" not in sample.columns or {"X", "Y"} - set(sample.columns):
        raise RuntimeError(
            "sample.csv missing gravity_resid / X / Y columns — rerun the sampling step first."
        )

    from tqdm.auto import tqdm
    results = []
    sigmas = list(ctx.config.sensitivity_sigmas)
    for sigma in tqdm(sigmas, desc="Gravity sensitivity (σ sweep)", unit="σ"):
        logger.info("Gravity sensitivity: sigma=%.0f km", sigma)
        compute_gravity_raw(ctx, sigma_km=sigma)
        # force=False: keep the gravity_raw.tif we just wrote at the sensitivity sigma.
        # _compute_gravity_for_period still re-runs orthogonalize() unconditionally.
        orthogonalize_gravity_ctx(ctx, force=False)

        data = sample.copy()
        data["gravity_resid"] = _resample_gravity_resid(d / "gravity_resid.tif", data)

        state = _build_and_fit("B", ctx, data)
        formula = state["formula"]
        coef_idx = _gravity_coef_index(formula)
        betas = state["betas"]
        deviance = state["deviance"]
        results.append({
            "sigma_km": sigma,
            "accessibility_coef": (
                float(betas[coef_idx])
                if coef_idx is not None and coef_idx < len(betas)
                else None
            ),
            "mean_deviance": (
                float(np.mean(deviance))
                if hasattr(deviance, "__len__")
                else float(deviance)
            ),
        })

    if backup_raw.exists():
        shutil.move(str(backup_raw), d / "gravity_raw.tif")
    if backup_resid.exists():
        shutil.move(str(backup_resid), d / "gravity_resid.tif")

    out_json.write_text(json.dumps(results, indent=2))
    logger.info("Gravity sensitivity written to %s", out_json)
    return out_json


def _gravity_coef_index(formula: str) -> int | None:
    """Find index of gravity_resid beta in the betas array (intercept = 0)."""
    try:
        terms = formula.split("~")[1].split("+")
        terms = [t.strip() for t in terms if "cell" not in t]
        for i, t in enumerate(terms):
            if "gravity_resid" in t:
                return i + 1  # +1 for intercept
    except Exception:
        pass
    return None
