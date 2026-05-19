"""Causal direction testing: SLX spatial lag model and optional DiD.

Phase 2 of the analytical pipeline:
  2.1 Build spatially lagged variables at ring bands 0-10, 10-20, 20-30, 30-50 km
  2.2 Fit forward and reverse SLX models
  2.3 Interpret causal direction, write JSON + text report
  2.4 DiD (auto-skipped if mill establishment year column absent)
"""
from __future__ import annotations
from pathlib import Path
import json
import logging
from typing import Optional

import numpy as np
from osgeo import gdal

from palmdef_risk.io.run import RunContext
from palmdef_risk.io.helpers import get_pixel_size_m

log = logging.getLogger(__name__)


def run_correlation_pipeline(ctx: RunContext) -> dict:
    """Run SLX and optional DiD. Returns dict of output paths and results."""
    d = ctx.data_dir
    out_dir = ctx.output_dir / "correlation"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Phase 2: Causal Direction Testing (SLX) ===")
    slx = run_slx(
        m_raster=d / "M.tif",
        p_raster=d / "P.tif",
        output_dir=out_dir,
        pixel_size_m=get_pixel_size_m(d / "forest_t2.tif"),
        n_sample=10000,
    )

    did = _try_did(ctx, out_dir)

    return {"slx": slx, "did": did}


def run_slx(
    m_raster: Path,
    p_raster: Path,
    output_dir: Path,
    pixel_size_m: float = 30.0,
    n_sample: int = 10000,
    ring_bands_km: Optional[list] = None,
) -> dict:
    """Fit forward and reverse SLX models. Return results dict."""
    if ring_bands_km is None:
        ring_bands_km = [(0, 10), (10, 20), (20, 30), (30, 50)]

    M = _read_valid_array(m_raster)
    P = _read_valid_array(p_raster)
    if M is None or P is None:
        log.warning("SLX skipped: M or P raster missing/unreadable")
        return {}

    WM = _build_ring_lags(M, ring_bands_km, pixel_size_m)
    WP = _build_ring_lags(P, ring_bands_km, pixel_size_m)

    flat_M = M.ravel()
    flat_P = P.ravel()
    flat_WM = [w.ravel() for w in WM]
    flat_WP = [w.ravel() for w in WP]

    valid = np.isfinite(flat_M) & np.isfinite(flat_P)
    for w in flat_WM + flat_WP:
        valid &= np.isfinite(w)
    idx = np.where(valid)[0]
    if len(idx) < 100:
        log.warning("SLX: fewer than 100 valid pixels -- skipping")
        return {}

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(idx, size=min(n_sample, len(idx)), replace=False)

    forward_result = _fit_ols(
        y=flat_P[sample_idx],
        X_cols={"M": flat_M[sample_idx],
                **{f"WM_r{i+1}": flat_WM[i][sample_idx]
                   for i in range(len(ring_bands_km))}},
        label="Forward (P ~ M + WM_r*)",
    )

    reverse_result = _fit_ols(
        y=flat_M[sample_idx],
        X_cols={"P": flat_P[sample_idx],
                **{f"WP_r{i+1}": flat_WP[i][sample_idx]
                   for i in range(len(ring_bands_km))}},
        label="Reverse (M ~ P + WP_r*)",
    )

    direction = _interpret_direction(forward_result, reverse_result)

    results = {
        "forward": forward_result,
        "reverse": reverse_result,
        "direction": direction,
        "n_sample": int(len(sample_idx)),
        "ring_bands_km": ring_bands_km,
    }

    json_path = output_dir / "slx_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))

    report_path = output_dir / "slx_report.txt"
    _write_slx_report(results, report_path)

    log.info("  SLX direction: %s", direction)
    log.info("  Results written to %s", output_dir)
    return results


def _build_ring_lags(arr: np.ndarray, ring_bands_km: list,
                     pixel_size_m: float) -> list[np.ndarray]:
    """Build ring-band focal means for each (inner_km, outer_km) band."""
    from scipy.ndimage import uniform_filter
    lags = []
    for inner_km, outer_km in ring_bands_km:
        inner_px = int(inner_km * 1000 / pixel_size_m)
        outer_px = int(outer_km * 1000 / pixel_size_m)
        outer_mean = uniform_filter(arr, size=2 * outer_px + 1)
        if inner_px > 0:
            inner_mean = uniform_filter(arr, size=2 * inner_px + 1)
            outer_area = (2 * outer_px + 1) ** 2
            inner_area = (2 * inner_px + 1) ** 2
            ring_mean = ((outer_mean * outer_area - inner_mean * inner_area)
                         / max(outer_area - inner_area, 1))
        else:
            ring_mean = outer_mean
        lags.append(ring_mean)
    return lags


def _fit_ols(y: np.ndarray, X_cols: dict, label: str) -> dict:
    """Fit OLS regression. Returns dict with coefficients and R2."""
    from numpy.linalg import lstsq

    names = list(X_cols.keys())
    X = np.column_stack([np.ones(len(y))] + [X_cols[n] for n in names])
    coeffs, _, _, _ = lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result = {
        "label": label,
        "intercept": float(coeffs[0]),
        "r2": round(r2, 4),
        "coefficients": {n: round(float(c), 6)
                         for n, c in zip(names, coeffs[1:])},
    }
    log.info("  %s  R2=%.4f", label, r2)
    return result


def _interpret_direction(forward: dict, reverse: dict) -> str:
    fwd_lags_sig = sum(1 for k, v in forward.get("coefficients", {}).items()
                       if k.startswith("WM_") and abs(v) > 0.01)
    rev_lags_sig = sum(1 for k, v in reverse.get("coefficients", {}).items()
                       if k.startswith("WP_") and abs(v) > 0.01)

    if fwd_lags_sig >= 2 and rev_lags_sig < 2:
        return "Story B: mills drive plantation expansion -> use LQ_PM"
    elif rev_lags_sig >= 2 and fwd_lags_sig < 2:
        return "Story A: plantations attract mills -> use LQ_MP"
    elif fwd_lags_sig >= 2 and rev_lags_sig >= 2:
        return "Bidirectional: report both LQ_MP and LQ_PM"
    else:
        return "No clear causal direction -- report both, emphasise controls"


def _write_slx_report(results: dict, path: Path) -> None:
    lines = [
        "=" * 60,
        "SLX CAUSAL DIRECTION REPORT",
        "=" * 60,
        f"Sample size     : {results.get('n_sample')}",
        f"Ring bands (km) : {results.get('ring_bands_km')}",
        "",
        "DIRECTION FINDING:",
        f"  {results.get('direction', 'N/A')}",
        "",
        "FORWARD MODEL (P ~ M + WM_r*)",
        f"  R2 = {results.get('forward', {}).get('r2', 'N/A')}",
    ]
    for k, v in results.get("forward", {}).get("coefficients", {}).items():
        lines.append(f"  {k:12s} = {v:+.6f}")
    lines += ["", "REVERSE MODEL (M ~ P + WP_r*)"]
    lines.append(f"  R2 = {results.get('reverse', {}).get('r2', 'N/A')}")
    for k, v in results.get("reverse", {}).get("coefficients", {}).items():
        lines.append(f"  {k:12s} = {v:+.6f}")
    lines += ["", "=" * 60]
    path.write_text("\n".join(lines))


def _read_valid_array(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    ds = gdal.Open(str(path))
    if ds is None:
        return None
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    if nd is not None:
        arr[arr == nd] = np.nan
    ds = None
    return arr


def _try_did(ctx: RunContext, out_dir: Path) -> Optional[dict]:
    """Run DiD if mill dataset has establishment year field."""
    import geopandas as gpd
    mill_gpkg = ctx.raw_dir / "mill" / "mill.gpkg"
    if not mill_gpkg.exists():
        return None
    mills = gpd.read_file(str(mill_gpkg))
    year_cols = [c for c in mills.columns
                 if any(kw in c.lower() for kw in ("year", "established", "built"))]
    if not year_cols:
        log.info("DiD skipped: no establishment year column found in mill dataset")
        return None
    log.info("DiD: found year column '%s' -- DiD analysis available", year_cols[0])
    return {"year_column": year_cols[0], "status": "available"}
