from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import pickle

import numpy as np
import pandas as pd

from palmdef_risk.parallel import run_parallel
from palmdef_risk.constants import NODATA_FLOAT

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)

# Static covariates reused from the t2 grid for forecast prediction (no t3 source).
_FORECAST_STATIC_RASTERS = (
    "altitude.tif", "slope.tif", "dist_road.tif", "dist_river.tif",
    "protected.tif", "hgu_signed_dist.tif",
)


def build_forecast_vardir(ctx: "RunContext") -> Path:
    """Assemble a clean data/forecast/ holding the full model-named covariate set.

    Copies static t2 rasters in; the t3 dynamics (dist_edge, dist_defor, dist_town,
    gravity_resid, plantation_resid) are produced upstream in Stage 2 and already
    live under data/forecast/. Returns the forecast directory path.
    """
    import shutil
    d = ctx.data_dir
    fcast = d / "forecast"
    fcast.mkdir(parents=True, exist_ok=True)
    for name in _FORECAST_STATIC_RASTERS:
        src = d / name
        if src.exists():
            shutil.copy2(src, fcast / name)
        else:
            logger.warning(
                "Static covariate %s missing — forecast var_dir may be incomplete", name
            )
    return fcast


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

        # Stream the elementwise log transform over GDAL blocks: read → log →
        # write per window so a full source + full result never coexist in RAM.
        # Bit-identical to the prior full-array path (float32 in, log(x+1) out).
        ds = gdal.Open(str(src_path))
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        nx, ny = band.XSize, band.YSize
        bx, by = band.GetBlockSize()

        drv = gdal.GetDriverByName("GTiff")
        out_ds = drv.Create(
            str(log_path), nx, ny, 1,
            gdal.GDT_Float32, ["COMPRESS=LZW", "TILED=YES"],
        )
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(NODATA_FLOAT)

        for yoff in range(0, ny, by):
            ywin = min(by, ny - yoff)
            for xoff in range(0, nx, bx):
                xwin = min(bx, nx - xoff)
                blk = band.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.float32)
                out_blk = np.full((ywin, xwin), NODATA_FLOAT, dtype=np.float32)
                valid = (blk != nodata) if nodata is not None else np.ones_like(blk, dtype=bool)
                out_blk[valid] = np.log(blk[valid] + 1)
                out_band.WriteArray(out_blk, xoff, yoff)

        out_band.FlushCache()
        out_ds = None
        ds = None
        logger.info("Created log raster: %s", log_path.name)


def _create_hgu_spline_rasters(data_dir: Path, formula: str, sample_path: Path) -> None:
    """Write hgu_b1.tif and hgu_b2.tif from hgu_signed_dist.tif for variant C prediction.

    forestatrisk reads covariates by filename, so the spline basis columns used
    during training must exist as raster files.  Only missing files are created.

    The cr() basis is rebuilt from the SAME training sample patsy memorized at fit
    time (boundary + interior knots), then applied via build_design_matrices.  This
    guarantees the prediction basis matches the fitted betas, and — because the
    memorized knots are reused — lets us evaluate the raster in GDAL-block windows
    without patsy re-deriving (and rejecting) knots per window.  cr() is a pure
    per-pixel function of the memorized knots, so a windowed evaluation is bit-
    identical to evaluating the whole raster at once, while never holding the full
    source array, the full valid mask, or the full basis columns in RAM (evaluating
    all ~288M valid pixels in one dmatrix call allocates a ~10 GB temp and OOMs).
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
    # that state so build_design_matrices reuses it (no per-window knot re-derivation).
    train_hgu = pd.read_csv(sample_path)["hgu_signed_dist"].to_numpy(dtype=np.float64)
    train_hgu = train_hgu[~np.isnan(train_hgu)]
    design_info = dmatrix(
        "cr(x, knots=(-5000, 0, 5000)) - 1", {"x": train_hgu}, return_type="matrix"
    ).design_info
    n_basis = len(design_info.column_names)

    ds = gdal.Open(str(src_path))
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    nx, ny = band.XSize, band.YSize
    bx, by = band.GetBlockSize()

    # Create one output raster per needed basis column up front; write per window.
    drv = gdal.GetDriverByName("GTiff")
    out_bands = {}  # name -> (out_ds, out_band, basis_col_idx)
    for i, name in enumerate(("hgu_b1", "hgu_b2")):
        if name not in needed:
            continue
        out_ds = drv.Create(
            str(data_dir / f"{name}.tif"), nx, ny, 1, gdal.GDT_Float32,
            ["COMPRESS=LZW", "TILED=YES"],
        )
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        ob = out_ds.GetRasterBand(1)
        ob.SetNoDataValue(NODATA_FLOAT)
        out_bands[name] = (out_ds, ob, min(i, n_basis - 1))

    for yoff in range(0, ny, by):
        ywin = min(by, ny - yoff)
        for xoff in range(0, nx, bx):
            xwin = min(bx, nx - xoff)
            blk = band.ReadAsArray(xoff, yoff, xwin, ywin).astype(np.float64)
            valid = (blk != nodata) if nodata is not None else np.ones_like(blk, dtype=bool)
            basis = None
            if valid.any():
                basis = np.asarray(
                    build_design_matrices([design_info], {"x": blk[valid].ravel()})[0]
                )
            for _, ob, col_idx in out_bands.values():
                out_blk = np.full((ywin, xwin), NODATA_FLOAT, dtype=np.float32)
                if basis is not None:
                    out_blk[valid] = basis[:, col_idx].astype(np.float32)
                ob.WriteArray(out_blk, xoff, yoff)

    ds = None
    for name, (out_ds, ob, _) in out_bands.items():
        ob.FlushCache()
        out_ds.FlushCache()
        logger.info("Created spline raster: %s.tif", name)
    out_bands.clear()


def predict_risk(ctx: RunContext, model_path: Path, variant: str) -> Path:
    """Run spatial risk prediction for a fitted ICAR variant.

    Loads the safe-state dict (no patsy objects), rebuilds patsy DesignInfo
    from sample.csv (per CLAUDE.md), constructs icarModelPred, interpolates
    rho to 1 km, then calls far.predict_raster_binomial_iCAR.

    Returns path to risk_<variant>.tif.
    """
    import forestatrisk as far
    from palmdef_risk.model.icar import load_design_matrix

    with open(model_path, "rb") as fh:
        state = pickle.load(fh)

    # Rebuild patsy DesignInfo from sample.csv (never pickle DesignInfo directly).
    # dropna="scaled" drops NaN on the scale()-wrapped columns BEFORE dmatrices so
    # scale() statistics match what fit_model used. Any NaN row would make scale()
    # store NaN as its mean → build_design_matrices returns 0 rows at predict time.
    sample_path = ctx.output_dir / "sample.csv"
    _, y, x = load_design_matrix(ctx, variant, state["formula"], dropna="scaled")

    pred_mod = far.icarModelPred(
        formula=state["formula"],
        _y_design_info=y.design_info,
        _x_design_info=x.design_info,
        betas=state["betas"],
        rho=state["rho"],
    )

    model_dir = model_path.parent
    rho_path = str(model_dir / "rho.tif")

    # Interpolate posterior-mean rho (cell resolution) to 1 km — skip if already
    # done (resumability invariant; interpolate_rho is otherwise re-run on rerun).
    if not Path(rho_path).exists():
        far.interpolate_rho(
            rho=state["rho"],
            input_raster=str(ctx.data_dir / "fcc23.tif"),
            output_file=rho_path,
            csize_orig=ctx.config.csize,
            csize_new=1,
        )
    else:
        logger.info("rho.tif exists — skipping interpolate_rho")

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


def predict_forecast(ctx: RunContext, model_path: Path, variant: str) -> Optional[Path]:
    """Predict t3 forecast risk for a fitted variant from data/forecast/ covariates.

    Reuses the model's interpolated rho.tif (spatial effect is location-based and
    time-agnostic). Returns risk_<variant>_forecast.tif, or None when the forecast
    var_dir lacks a required covariate raster.
    """
    import forestatrisk as far
    from palmdef_risk.model.icar import load_design_matrix

    with open(model_path, "rb") as fh:
        state = pickle.load(fh)

    fcast = ctx.data_dir / "forecast"
    rho_path = model_path.parent / "rho.tif"
    if not rho_path.exists():
        logger.warning(
            "rho.tif missing for variant %s — run predict_risk before predict_forecast",
            variant,
        )
        return None

    # Rebuild DesignInfo from sample.csv (never pickle DesignInfo).
    sample_path = ctx.output_dir / "sample.csv"
    _, y, x = load_design_matrix(ctx, variant, state["formula"], dropna="scaled")
    scaled_cols = re.findall(r"scale\((\w+)\)", state["formula"])  # reused below

    pred_mod = far.icarModelPred(
        formula=state["formula"],
        _y_design_info=y.design_info,
        _x_design_info=x.design_info,
        betas=state["betas"],
        rho=state["rho"],
    )

    # Build derived rasters INTO the forecast dir (from t3 dist_* + copied statics).
    _create_log_dist_rasters(fcast, state["formula"])
    _create_hgu_spline_rasters(fcast, state["formula"], sample_path)

    # Guard: every scaled covariate + protected must exist in the forecast var_dir.
    bare_covs = {"protected"}
    needed = {v for v in set(scaled_cols) | bare_covs if v != "cell"}
    missing = [v for v in needed if not (fcast / f"{v}.tif").exists()]
    if missing:
        logger.warning(
            "Forecast prediction for variant %s skipped — missing forecast rasters %s",
            variant, sorted(missing),
        )
        return None

    out_dir = ctx.output_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_path = out_dir / f"risk_{variant}_forecast.tif"
    forest_t3 = ctx.data_dir / "forest_t3.tif"
    if not forest_t3.exists():
        logger.warning("forest_t3.tif missing — cannot predict forecast for %s", variant)
        return None

    far.predict_raster_binomial_iCAR(
        pred_mod,
        var_dir=str(fcast),
        input_cell_raster=str(rho_path),
        input_forest_raster=str(forest_t3),
        output_file=str(risk_path),
    )
    logger.info("Forecast risk raster written: %s", risk_path)
    return risk_path


def _prewarm_derived_rasters(ctx: RunContext) -> None:
    """Pre-create the shared derived covariate rasters (log-distance, HGU spline)
    once, before the parallel pool, into both the t2 data dir and the forecast
    dir. predict_risk / predict_forecast otherwise create these on demand into the
    *shared* data dir; running variants concurrently would race two processes
    writing the same log_dist_*.tif / hgu_b*.tif. Building them here (idempotent,
    existence-guarded) makes every per-variant worker a no-op for these files.
    """
    sample_path = ctx.output_dir / "sample.csv"
    fcast = ctx.data_dir / "forecast"
    for variant in ctx.config.model_variants:
        model_path = ctx.output_dir / "models" / f"model_{variant}" / f"mod_{variant}.pkl"
        if not model_path.exists():
            continue
        with open(model_path, "rb") as fh:
            formula = pickle.load(fh)["formula"]
        _create_log_dist_rasters(ctx.data_dir, formula)
        _create_hgu_spline_rasters(ctx.data_dir, formula, sample_path)
        if fcast.exists():
            _create_log_dist_rasters(fcast, formula)
            _create_hgu_spline_rasters(fcast, formula, sample_path)


def _predict_one_variant(task: tuple) -> list[str]:
    """Module-level worker: predict + project + forecast for one variant.

    Mirrors the previous per-variant try/except/continue and skip-guards so a
    single variant's failure is logged but does not abort the others — the
    exception must not reach the pool (run_parallel would otherwise propagate it
    and kill the run). Returns the output paths produced (as strings). The run
    dir is reloaded into a RunContext (paths cross the pool, not live objects).
    """
    variant, run_dir, area_t2, area_t3 = task
    from palmdef_risk.io.run import load_run
    ctx = load_run(run_dir)
    out: list[str] = []

    model_path = ctx.output_dir / "models" / f"model_{variant}" / f"mod_{variant}.pkl"
    if not model_path.exists():
        logger.warning("Model pkl not found, skipping variant %s: %s", variant, model_path)
        return out

    risk_path = ctx.output_dir / "predictions" / f"risk_{variant}.tif"
    if risk_path.exists():
        logger.info("risk_%s.tif exists — skipping prediction", variant)
        out.append(str(risk_path))
    else:
        try:
            risk_path = predict_risk(ctx, model_path, variant)
            out.append(str(risk_path))
        except Exception:
            import traceback
            logger.error("Prediction failed for variant %s:\n%s", variant, traceback.format_exc())
            return out  # no risk raster → skip projection + forecast (was `continue`)

    try:
        future_path = project_future(ctx, risk_path, variant,
                                     area_t2=area_t2, area_t3=area_t3)
        if future_path is not None:
            out.append(str(future_path))
    except Exception:
        import traceback
        logger.error("Future projection failed for variant %s:\n%s", variant, traceback.format_exc())

    # t3 forecast risk (decision deferred: does NOT yet feed project_future)
    fc_path = ctx.output_dir / "predictions" / f"risk_{variant}_forecast.tif"
    if fc_path.exists():
        logger.info("risk_%s_forecast.tif exists — skipping forecast", variant)
        out.append(str(fc_path))
    else:
        try:
            fc = predict_forecast(ctx, model_path, variant)
            if fc is not None:
                out.append(str(fc))
        except Exception:
            import traceback
            logger.error("Forecast prediction failed for variant %s:\n%s", variant, traceback.format_exc())
    return out


def predict_all(ctx: RunContext) -> list[Path]:
    """Predict risk for all fitted model variants in parallel via run_parallel.

    build_forecast_vardir and the shared derived rasters are prepared once
    (variant-invariant); each variant is then predicted in its own process,
    bounded by ram_per_predict_gb. Per-variant failures are isolated (logged,
    others proceed), matching the previous sequential loop.
    """
    cfg = ctx.config
    variants = list(cfg.model_variants)
    build_forecast_vardir(ctx)
    _prewarm_derived_rasters(ctx)

    # Forest t2/t3 areas are variant-invariant — compute once here rather than
    # once per variant inside project_future (countpix scans the full raster).
    area_t2 = area_t3 = None
    if cfg.project_future:
        t2_path = ctx.data_dir / "forest_t2.tif"
        t3_path = ctx.data_dir / "forest_t3.tif"
        if t2_path.exists() and t3_path.exists():
            import forestatrisk as far
            area_t2 = far.countpix(input_raster=str(t2_path), value=1)["area"]
            area_t3 = far.countpix(input_raster=str(t3_path), value=1)["area"]

    tasks = [(v, str(ctx.run_dir), area_t2, area_t3) for v in variants]
    nested = run_parallel(
        _predict_one_variant, tasks,
        ram_per_task_gb=cfg.ram_per_predict_gb, cfg=cfg,
        desc="Predicting risk",
    )
    results: list[Path] = []
    for paths in nested:
        results.extend(Path(p) for p in (paths or []))
    return results


def project_future(
    ctx: RunContext,
    risk_path: Path,
    variant: str,
    area_t2: Optional[float] = None,
    area_t3: Optional[float] = None,
) -> Optional[Path]:
    """Project future forest cover by extrapolating the historical defor rate.

    Steps:
      1. n_years = projection_year - forest_years[-1] (must be > 0).
      2. Annual hectares deforested in the historical window
         (forest_years[-2] → forest_years[-1]) = (area_t2 - area_t3) / Δyears.
      3. Target hectares to deforest = annual_ha × n_years.
      4. far.deforest selects the highest-risk pixels (by risk_<v>.tif) until
         that target hectarage is reached and writes a binary forest mask.

    `area_t2`/`area_t3` (forest hectares at t2/t3) are variant-invariant; pass
    them in to avoid recomputing far.countpix once per variant. When omitted
    they are computed here (backward compatible).

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
    # Areas are variant-invariant — reuse precomputed values when supplied.
    if area_t2 is None:
        area_t2 = far.countpix(input_raster=str(t2_path), value=1)["area"]
    if area_t3 is None:
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
