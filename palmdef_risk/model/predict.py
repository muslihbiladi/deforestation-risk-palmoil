from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional
import pickle

import numpy as np

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext


def predict_risk(ctx: RunContext, model_path: Path, variant: str) -> Path:
    """Run spatial risk prediction for a fitted ICAR variant.

    Loads the pickled model, calls far.predict.predict_raster() on
    ctx.data_dir, and writes <output_dir>/predictions/risk_<variant>.tif.
    Returns the output path.
    """
    import forestatrisk as far

    with open(model_path, "rb") as fh:
        mod = pickle.load(fh)

    out_dir = ctx.output_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_path = out_dir / f"risk_{variant}.tif"

    far.predict.predict_raster(
        mod,
        var_dir=str(ctx.data_dir),
        input_raster=str(ctx.data_dir / "fcc23.tif"),
        output_file=str(risk_path),
    )

    return risk_path


def project_future(ctx: RunContext, risk_path: Path, variant: str) -> Optional[Path]:
    """Project future deforestation if config.project_future is True.

    Uses forest_t3 as starting forest cover, applies annual deforestation
    probability from the risk raster over (projection_year - forest_years[-1])
    years. Writes <output_dir>/predictions/forest_future_<variant>.tif.
    Returns output path or None when projection is disabled.
    """
    if not ctx.config.project_future:
        return None

    import forestatrisk as far

    n_years = ctx.config.projection_year - ctx.config.forest_years[-1]
    if n_years <= 0:
        return None

    out_dir = ctx.output_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    future_path = out_dir / f"forest_future_{variant}.tif"

    far.deforest(
        input_raster=str(ctx.data_dir / "forest_t3.tif"),
        hectares=None,
        output_file=str(future_path),
        blk_rows=128,
        probability_file=str(risk_path),
        time_interval=n_years,
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
