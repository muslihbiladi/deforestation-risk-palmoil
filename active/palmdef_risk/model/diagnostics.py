from __future__ import annotations
import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

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
    coords: dict[str, list[tuple[float, float]]],
    out_json: Path | str,
) -> dict:
    """Compute Moran's I on deviance residuals for each model variant. Uses k=8 KNN weights.

    `coords` is per-variant because each variant may have a different training
    subset (variant B drops gravity_resid NaNs, variant C also drops HGU NaNs).
    """
    try:
        from libpysal.weights import KNN
        from esda.moran import Moran
    except ImportError:
        logger.warning("libpysal/esda not installed — Moran's I skipped")
        results = {v: {"I": None, "p_value": None, "note": "esda not installed"}
                   for v in residuals}
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(results, indent=2))
        return results

    import geopandas as gpd
    from shapely.geometry import Point

    results = {}
    for variant, resid in residuals.items():
        var_coords = coords[variant]
        pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in var_coords])
        w = KNN.from_dataframe(pts, k=min(8, len(var_coords) - 1))
        w.transform = "R"
        mi = Moran(resid, w)
        results[variant] = {
            "I": round(float(mi.I), 4),
            "p_value": round(float(mi.p_norm), 4),
            "n": int(len(resid)),
        }
        logger.info("Moran's I [%s]: I=%.4f p=%.4f n=%d",
                    variant, mi.I, mi.p_norm, len(resid))

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(results, indent=2))
    return results


def compute_residuals_all(
    ctx: "RunContext",
) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[float, float]]]]:
    """Compute per-sample deviance residuals for every fitted model variant.

    For each variant, reloads its pickled state (betas, rho, formula), rebuilds
    the design matrix from sample.csv using the same dropna subset as fit_model
    (so patsy's scale() statistics match training), computes
    p_hat = sigmoid(X_fixed @ betas + rho[cell]), and returns Bernoulli
    deviance residuals alongside the (X, Y) coords of the surviving rows.
    """
    from patsy import dmatrices
    from palmdef_risk.model.icar import prepare_sample, _LOG_DIST_COLS, variant_extra_cols

    sample_path = ctx.output_dir / "sample.csv"
    base_data = pd.read_csv(sample_path)
    base_data = prepare_sample(base_data)

    base_cols = ["fcc23", "altitude", "slope", "protected", "cell"] + [
        f"log_{c}" for c in _LOG_DIST_COLS
    ]

    from tqdm.auto import tqdm
    residuals: dict[str, np.ndarray] = {}
    coords: dict[str, list[tuple[float, float]]] = {}

    variants = list(ctx.config.model_variants)
    for variant in tqdm(variants, desc="Computing deviance residuals", unit="variant"):
        pkl_path = ctx.output_dir / "models" / f"model_{variant}" / f"mod_{variant}.pkl"
        if not pkl_path.exists():
            logger.warning("Model pkl missing for variant %s — skipping residuals", variant)
            continue
        with open(pkl_path, "rb") as fh:
            state = pickle.load(fh)

        extra_cols = variant_extra_cols(variant)
        data = base_data.dropna(subset=base_cols + extra_cols)
        if len(data) == 0:
            logger.warning("No rows survive dropna for variant %s — skipping", variant)
            continue

        y, x = dmatrices(state["formula"], data, return_type="matrix")
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        col_names = list(x.design_info.column_names)
        if "cell" not in col_names:
            raise ValueError(f"'cell' column missing from design matrix for variant {variant}")
        cell_pos = col_names.index("cell")
        cell_idx = x_arr[:, cell_pos].astype(int)
        x_fixed = np.delete(x_arr, cell_pos, axis=1)

        betas = np.asarray(state["betas"]).ravel()
        rho = np.asarray(state["rho"]).ravel()
        if x_fixed.shape[1] != betas.shape[0]:
            raise ValueError(
                f"Variant {variant}: fixed-effect columns ({x_fixed.shape[1]}) "
                f"do not match betas length ({betas.shape[0]})"
            )

        eta = x_fixed @ betas + rho[cell_idx]
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        p_hat = np.clip(p_hat, 1e-12, 1.0 - 1e-12)

        y_obs = y_arr[:, 0]  # I(1 - fcc23): 1 = deforested, 0 = remained
        d2 = np.where(y_obs == 1, -2.0 * np.log(p_hat), -2.0 * np.log(1.0 - p_hat))
        dresid = np.sign(y_obs - p_hat) * np.sqrt(d2)

        residuals[variant] = dresid
        coords[variant] = list(zip(data["X"].to_numpy(), data["Y"].to_numpy()))
        logger.info("Variant %s: computed %d deviance residuals", variant, len(dresid))

    return residuals, coords
