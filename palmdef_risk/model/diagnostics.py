from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_vif(
    covariates: list[str],
    sample_csv: Path | str,
    out_json: Path | str,
) -> dict[str, float]:
    """Compute VIF for each covariate. Writes results to out_json. Warns for VIF > 5."""
    data = pd.read_csv(sample_csv)[covariates].dropna()
    X = data.values
    vif = {}
    for j, col in enumerate(covariates):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        others = np.column_stack([np.ones(len(y)), others])
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        y_hat = others @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        v = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        vif[col] = round(v, 3)
        if v > 10:
            logger.warning("VIF(%s)=%.1f > 10 (high multicollinearity)", col, v)
        elif v > 5:
            logger.warning("VIF(%s)=%.1f > 5 (moderate multicollinearity)", col, v)

    Path(out_json).write_text(json.dumps(vif, indent=2))
    return vif


def compute_morans_i(
    residuals: dict[str, np.ndarray],
    coords: list[tuple[float, float]],
    out_json: Path | str,
) -> dict:
    """Compute Moran's I on deviance residuals for each model variant. Uses k=8 KNN weights."""
    try:
        from libpysal.weights import KNN
        from esda.moran import Moran
    except ImportError:
        logger.warning("libpysal/esda not installed — Moran's I skipped")
        results = {v: {"I": None, "p_value": None, "note": "esda not installed"}
                   for v in residuals}
        Path(out_json).write_text(json.dumps(results, indent=2))
        return results

    import geopandas as gpd
    from shapely.geometry import Point
    pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coords])
    w = KNN.from_dataframe(pts, k=min(8, len(coords) - 1))
    w.transform = "R"

    results = {}
    for variant, resid in residuals.items():
        mi = Moran(resid, w)
        results[variant] = {
            "I": round(float(mi.I), 4),
            "p_value": round(float(mi.p_norm), 4),
        }
        logger.info("Moran's I [%s]: I=%.4f p=%.4f", variant, mi.I, mi.p_norm)

    Path(out_json).write_text(json.dumps(results, indent=2))
    return results
