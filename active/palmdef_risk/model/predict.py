from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import pickle

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)


def _create_log_dist_rasters(data_dir: Path, formula: str) -> None:
    """Write log_dist_*.tif rasters to data_dir for each log_dist_* term in formula.

    forestatrisk's predict_raster_binomial_iCAR reads rasters by filename, so
    pre-computed log-distance covariates must exist as .tif files alongside the
    original dist_*.tif rasters.  Only missing files are created.
    """
    from osgeo import gdal

    needed = re.findall(r"log_(dist_\w+)", formula)
    for col in needed:
        log_path = data_dir / f"log_{col}.tif"
        if log_path.exists():
            continue
        src_path = data_dir / f"{col}.tif"
        if not src_path.exists():
            logger.warning("Source raster missing, cannot create %s", log_path.name)
            continue

        ds = gdal.Open(str(src_path))
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        arr = band.ReadAsArray().astype(np.float32)

        out_arr = np.full_like(arr, -9999.0)
        valid = arr != nodata if nodata is not None else np.ones_like(arr, dtype=bool)
        out_arr[valid] = np.log(arr[valid] + 1)

        drv = gdal.GetDriverByName("GTiff")
        out_ds = drv.Create(
            str(log_path), ds.RasterXSize, ds.RasterYSize, 1,
            gdal.GDT_Float32, ["COMPRESS=LZW", "TILED=YES"],
        )
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(out_arr)
        out_band.SetNoDataValue(-9999.0)
        out_band.FlushCache()
        out_ds = None
        ds = None
        logger.info("Created log raster: %s", log_path.name)


def _create_hgu_spline_rasters(data_dir: Path, formula: str, sample_path: Path) -> None:
    """Write hgu_b1.tif and hgu_b2.tif from hgu_signed_dist.tif for variant C prediction.

    forestatrisk reads covariates by filename, so the spline basis columns used
    during training must exist as raster files.  Only missing files are created.

    The cr() basis is rebuilt from the SAME training sample patsy memorized at fit
    time (boundary + interior knots), then applied to the full raster via
    build_design_matrices.  This guarantees the prediction basis matches the fitted
    betas, and — because the memorized knots are reused — lets us evaluate the raster
    in chunks without patsy re-deriving (and rejecting) knots per chunk.  Evaluating
    all ~288M valid pixels in one dmatrix call allocates a ~10 GB (n_knots, N) temp
    and OOMs; chunking caps the temp at a few hundred MB.
    """
    if "hgu_b1" not in formula and "hgu_b2" not in formula:
        return
    from osgeo import gdal
    from patsy import dmatrix, build_design_matrices

    src_path = data_dir / "hgu_signed_dist.tif"
    if not src_path.exists():
        logger.warning("hgu_signed_dist.tif missing — cannot create hgu spline rasters")
        return

    needed = [n for n in ("hgu_b1", "hgu_b2") if n in formula and not (data_dir / f"{n}.tif").exists()]
    if not needed:
        return

    # Recover the trained spline state: patsy cr() memorizes boundary knots from the
    # sample's data range during fit. Rebuilding from sample.csv reproduces exactly
    # that state so build_design_matrices reuses it (no per-chunk knot re-derivation).
    train_hgu = pd.read_csv(sample_path)["hgu_signed_dist"].to_numpy(dtype=np.float64)
    train_hgu = train_hgu[~np.isnan(train_hgu)]
    design_info = dmatrix(
        "cr(x, knots=(-5000, 0, 5000)) - 1", {"x": train_hgu}, return_type="matrix"
    ).design_info
    n_basis = len(design_info.column_names)

    ds = gdal.Open(str(src_path))
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float64)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ny, nx = arr.shape
    ds = None

    valid = (arr != nodata) if nodata is not None else np.ones((ny, nx), dtype=bool)
    x_valid = arr[valid].ravel()
    del arr  # free full raster; valid mask is sufficient for scatter-write
    n_valid = len(x_valid)

    # Only hgu_b1/hgu_b2 (basis cols 0/1) feed the model — materialise just those,
    # not all n_basis full-length columns. Apply the memorized basis in 5M-pixel
    # chunks to avoid the ~10 GB monolithic temp.
    want_idx = sorted({min(("hgu_b1", "hgu_b2").index(n), n_basis - 1) for n in needed})
    _CHUNK = 5_000_000
    dm_cols = {j: np.empty(n_valid, dtype=np.float32) for j in want_idx}
    for start in range(0, n_valid, _CHUNK):
        end = min(start + _CHUNK, n_valid)
        chunk = np.asarray(
            build_design_matrices([design_info], {"x": x_valid[start:end]})[0]
        )
        for j in want_idx:
            dm_cols[j][start:end] = chunk[:, j].astype(np.float32)
    del x_valid

    for i, name in enumerate(("hgu_b1", "hgu_b2")):
        if name not in needed:
            continue
        out_path = data_dir / f"{name}.tif"
        col_idx = min(i, n_basis - 1)
        out_arr = np.full((ny, nx), -9999.0, dtype=np.float32)
        out_arr[valid] = dm_cols[col_idx]

        out_ds = gdal.GetDriverByName("GTiff").Create(
            str(out_path), nx, ny, 1, gdal.GDT_Float32,
            ["COMPRESS=LZW", "TILED=YES"],
        )
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(out_arr)
        out_band.SetNoDataValue(-9999.0)
        out_band.FlushCache()
        out_ds = None
        logger.info("Created spline raster: %s", out_path.name)


def predict_risk(ctx: RunContext, model_path: Path, variant: str) -> Path:
    """Run spatial risk prediction for a fitted ICAR variant.

    Loads the safe-state dict (no patsy objects), rebuilds patsy DesignInfo
    from sample.csv (per CLAUDE.md), constructs icarModelPred, interpolates
    rho to 1 km, then calls far.predict_raster_binomial_iCAR.

    Returns path to risk_<variant>.tif.
    """
    import forestatrisk as far
    from patsy import dmatrices
    from palmdef_risk.model.icar import prepare_sample

    with open(model_path, "rb") as fh:
        state = pickle.load(fh)

    # Rebuild patsy DesignInfo from sample.csv (never pickle DesignInfo directly).
    # Must dropna on scaled columns BEFORE dmatrices so scale() statistics match
    # what fit_model used.  Any NaN row causes scale() to store NaN as its mean,
    # which makes build_design_matrices return 0 rows at prediction time.
    sample_path = ctx.output_dir / "sample.csv"
    data = pd.read_csv(sample_path)
    data = prepare_sample(data)
    scaled_cols = re.findall(r"scale\((\w+)\)", state["formula"])
    if scaled_cols:
        data = data.dropna(subset=scaled_cols)
    y, x = dmatrices(state["formula"], data, return_type="matrix")

    pred_mod = far.icarModelPred(
        formula=state["formula"],
        _y_design_info=y.design_info,
        _x_design_info=x.design_info,
        betas=state["betas"],
        rho=state["rho"],
    )

    model_dir = model_path.parent
    rho_path = str(model_dir / "rho.tif")

    # Interpolate posterior-mean rho (cell resolution) to 1 km
    far.interpolate_rho(
        rho=state["rho"],
        input_raster=str(ctx.data_dir / "fcc23.tif"),
        output_file=rho_path,
        csize_orig=ctx.config.csize,
        csize_new=1,
    )

    out_dir = ctx.output_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_path = out_dir / f"risk_{variant}.tif"

    # forestatrisk reads variable rasters by filename; create derived rasters
    # (log-distances, HGU spline basis) before calling predict.
    _create_log_dist_rasters(ctx.data_dir, state["formula"])
    _create_hgu_spline_rasters(ctx.data_dir, state["formula"], sample_path)

    # Validate that every covariate raster exists before handing off to forestatrisk.
    # A missing raster causes forestatrisk to silently process 0 pixels → cryptic
    # numpy broadcast error "(0,) (49202,)".
    scaled_vars = re.findall(r"scale\((\w+)\)", state["formula"])
    bare_covs = {"protected"}  # non-scaled covariates forestatrisk reads from disk
    missing_rasters = [
        v for v in set(scaled_vars) | bare_covs
        if v != "cell" and not (ctx.data_dir / f"{v}.tif").exists()
    ]
    if missing_rasters:
        raise FileNotFoundError(
            f"Prediction for variant {variant} requires rasters that are missing from "
            f"{ctx.data_dir}: {sorted(missing_rasters)}. "
            "Re-run the data processing steps (align_all / compute_all_distances / "
            "compute_gravity_accessibility) to recreate them, then refit the models."
        )

    far.predict_raster_binomial_iCAR(
        pred_mod,
        var_dir=str(ctx.data_dir),
        input_cell_raster=rho_path,
        input_forest_raster=str(ctx.data_dir / "fcc23.tif"),
        output_file=str(risk_path),
    )

    logger.info("Risk raster written: %s", risk_path)
    return risk_path


def predict_all(ctx: RunContext) -> list[Path]:
    """Predict risk for all fitted model variants."""
    from tqdm.auto import tqdm
    results = []
    variants = list(ctx.config.model_variants)
    for variant in tqdm(variants, desc="Predicting risk", unit="variant"):
        model_path = ctx.output_dir / "models" / f"model_{variant}" / f"mod_{variant}.pkl"
        if not model_path.exists():
            logger.warning("Model pkl not found, skipping variant %s: %s", variant, model_path)
            continue
        risk_path = ctx.output_dir / "predictions" / f"risk_{variant}.tif"
        if risk_path.exists():
            logger.info("risk_%s.tif exists — skipping prediction", variant)
            results.append(risk_path)
            continue
        try:
            risk_path = predict_risk(ctx, model_path, variant)
            results.append(risk_path)
        except Exception:
            import traceback
            logger.error("Prediction failed for variant %s:\n%s", variant, traceback.format_exc())
            continue
        try:
            future_path = project_future(ctx, risk_path, variant)
            if future_path is not None:
                results.append(future_path)
        except Exception:
            import traceback
            logger.error("Future projection failed for variant %s:\n%s", variant, traceback.format_exc())
    return results


def project_future(ctx: RunContext, risk_path: Path, variant: str) -> Optional[Path]:
    """Project future forest cover by extrapolating the historical defor rate.

    Steps:
      1. n_years = projection_year - forest_years[-1] (must be > 0).
      2. Annual hectares deforested in the historical window
         (forest_years[-2] → forest_years[-1]) = (area_t2 - area_t3) / Δyears.
      3. Target hectares to deforest = annual_ha × n_years.
      4. far.deforest selects the highest-risk pixels (by risk_<v>.tif) until
         that target hectarage is reached and writes a binary forest mask.

    Writes <output_dir>/predictions/forest_future_<variant>.tif.
    Returns None when project_future is disabled or n_years ≤ 0.
    """
    if not ctx.config.project_future:
        return None

    import forestatrisk as far

    years = ctx.config.forest_years
    n_years = ctx.config.projection_year - years[-1]
    if n_years <= 0:
        return None

    t2_path = ctx.data_dir / "forest_t2.tif"
    t3_path = ctx.data_dir / "forest_t3.tif"
    if not t2_path.exists() or not t3_path.exists():
        logger.warning(
            "Cannot project future for %s: forest_t2.tif or forest_t3.tif missing",
            variant,
        )
        return None

    # Historical annual deforestation rate (hectares/year) from t2 → t3.
    area_t2 = far.countpix(input_raster=str(t2_path), value=1)["area"]
    area_t3 = far.countpix(input_raster=str(t3_path), value=1)["area"]
    hist_span = years[-1] - years[-2]
    if hist_span <= 0:
        logger.warning("Invalid forest_years span (%s) — skipping projection", years)
        return None
    annual_ha = max(area_t2 - area_t3, 0.0) / hist_span
    target_ha = annual_ha * n_years

    out_dir = ctx.output_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    future_path = out_dir / f"forest_future_{variant}.tif"

    stats = far.deforest(
        input_raster=str(risk_path),
        hectares=target_ha,
        output_file=str(future_path),
        blk_rows=128,
    )
    logger.info(
        "Projected %s: %.0f ha over %d yr (annual=%.0f ha/yr, t2→t3 span=%d yr); "
        "threshold=%s, error=%.2f%%",
        variant, target_ha, n_years, annual_ha, hist_span,
        stats.get("threshold"), stats.get("error_perc", float("nan")),
    )
    return future_path


def classify_risk(risk_array: np.ndarray, thresholds: list) -> np.ndarray:
    """Classify a continuous risk array into integer zones (1-based).

    thresholds is an ascending list of N-1 break values; result has N zones.
    """
    out = np.ones(risk_array.shape, dtype=np.uint8)
    for i, t in enumerate(thresholds):
        out[risk_array > t] = i + 2
    return out


def _write_risk_raster(
    prob_arr: np.ndarray,
    ref_tif: str,
    out_tif: str,
) -> None:
    """Write probability [0,1] as UInt16. NoData=0, valid range=[1, 65535]."""
    from osgeo import gdal
    ds = gdal.Open(ref_tif)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ny, nx = prob_arr.shape
    ds = None

    scaled = np.round(prob_arr * 65535).astype(np.uint16)
    scaled = np.clip(scaled, 1, 65535)
    scaled[prob_arr == 0.0] = 0

    out_ds = gdal.GetDriverByName("GTiff").Create(
        out_tif, nx, ny, 1, gdal.GDT_UInt16,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(scaled)
    out_ds.GetRasterBand(1).SetNoDataValue(0)
    out_ds.FlushCache()
    out_ds = None
