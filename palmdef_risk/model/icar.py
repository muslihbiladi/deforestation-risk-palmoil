from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)

_LOG_DIST_COLS = ["dist_defor", "dist_edge", "dist_road", "dist_town", "dist_river"]

# Scaled covariates for each model variant (order determines column order in X)
_VARIANT_SCALED_COLS: dict[str, list[str]] = {
    "A": ["altitude", "slope"] + [f"log_{c}" for c in _LOG_DIST_COLS],
    "B": ["altitude", "slope"] + [f"log_{c}" for c in _LOG_DIST_COLS] + ["gravity_resid"],
    "C": ["altitude", "slope"] + [f"log_{c}" for c in _LOG_DIST_COLS] + ["gravity_resid", "hgu_b1", "hgu_b2"],
}


def build_formula(variant: str, data: pd.DataFrame) -> str:
    """Build the forestatrisk formula for variant A, B, or C.

    Constant columns (min == max) are excluded automatically: patsy's scale()
    divides by std, which for a near-zero std produces extreme / NaN values that
    cause NA_action='drop' to remove all rows.
    """
    all_scaled = _VARIANT_SCALED_COLS.get(variant)
    if all_scaled is None:
        raise ValueError(f"Unknown variant: {variant!r}. Valid variants: A, B, C")

    excluded, active = [], []
    for col in all_scaled:
        if col not in data.columns or data[col].min() == data[col].max():
            excluded.append(col)
        else:
            active.append(col)

    if excluded:
        logger.warning("Excluding constant/missing covariates (all-NoData or absent): %s", excluded)
    if not active:
        raise ValueError(f"No valid numeric covariates remain for variant {variant!r}")

    rhs = " + ".join(f"scale({c})" for c in active) + " + protected"
    return f"I(1 - fcc23) + trial ~ {rhs} + cell"


def _add_hgu_spline_cols(data: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute hgu_b1, hgu_b2 spline basis columns from hgu_signed_dist."""
    if "hgu_signed_dist" not in data.columns:
        return data
    from patsy import dmatrix
    import numpy as np
    x = data["hgu_signed_dist"].values
    dm = dmatrix("cr(x, knots=(-5000, 0, 5000)) - 1", {"x": x}, return_type="matrix")
    dm_arr = np.asarray(dm)
    data = data.copy()
    data["hgu_b1"] = dm_arr[:, 0]
    if dm_arr.shape[1] > 1:
        data["hgu_b2"] = dm_arr[:, 1]
    else:
        data["hgu_b2"] = 0.0
    return data


def prepare_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (trial, log distances, HGU spline basis) to the sample DataFrame."""
    data = data.copy()
    data["trial"] = 1  # binomial trial count required by forestatrisk formula
    for col in _LOG_DIST_COLS:
        data[f"log_{col}"] = np.log(data[col] + 1)
    data = _add_hgu_spline_cols(data)
    return data


def _build_and_fit(variant: str, ctx: "RunContext", data: pd.DataFrame) -> dict:
    """Build formula + run iCAR MCMC on the provided sample DataFrame.

    Returns the model state dict (no file I/O). Used by both fit_model (which
    pickles the result) and the gravity sensitivity loop (which needs to
    refit on data with a freshly resampled gravity_resid column without
    clobbering the baseline pickle).
    """
    import forestatrisk as far

    data = prepare_sample(data)

    # patsy's scale() sums all values including NaN → mean becomes NaN → all rows dropped.
    # Drop NaN rows on formula columns BEFORE building the formula.
    base_cols = ["fcc23", "altitude", "slope", "protected", "cell"] + [
        f"log_{c}" for c in _LOG_DIST_COLS
    ]
    extra_cols = (
        (["gravity_resid"] if variant in ("B", "C") else [])
        + (["hgu_b1", "hgu_b2"] if variant == "C" else [])
    )
    n_before = len(data)
    data = data.dropna(subset=base_cols + extra_cols)
    if (n_dropped := n_before - len(data)):
        logger.warning("Dropped %d rows with NaN in formula columns", n_dropped)

    formula = build_formula(variant, data)

    cfg = ctx.config
    nneigh, adj = far.cellneigh(
        raster=str(ctx.data_dir / "fcc23.tif"),
        csize=cfg.csize,
        rank=1,
    )
    mod = far.model_binomial_iCAR(
        suitability_formula=formula,
        data=data,
        n_neighbors=nneigh,
        neighbors=adj,
        Vbeta=cfg.Vbeta,
        beta_start=-99,
        burnin=cfg.burnin,
        mcmc=cfg.mcmc,
        thin=cfg.thin,
        seed=cfg.seed,
        save_rho=0,  # 0 = posterior mean (1D vector), 1 = full MCMC chains
        verbose=0,
    )
    return {
        "betas": mod.betas,
        "rho": mod.rho,        # 1D posterior mean (length = n_cells)
        "formula": formula,
        "mcmc": mod.mcmc,      # Full MCMC chains (nsamp × (npar+2))
        "deviance": mod.deviance,
        "variant": variant,
    }


def fit_model(variant: str, ctx: "RunContext") -> Path:
    """Fit one iCAR model variant. Returns path to saved .pkl file."""
    sample_path = ctx.output_dir / "sample.csv"
    data = pd.read_csv(sample_path)
    state = _build_and_fit(variant, ctx, data)

    model_dir = ctx.output_dir / "models" / f"model_{variant}"
    model_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = model_dir / f"mod_{variant}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(state, f)
    logger.info("Model %s fitted and saved to %s", variant, pkl_path)
    return pkl_path


def fit_all(ctx: "RunContext") -> list[Path]:
    """Fit all configured model variants sequentially."""
    from tqdm.auto import tqdm
    results = []
    variants = list(ctx.config.model_variants)
    for v in tqdm(variants, desc="Fitting iCAR models", unit="variant"):
        try:
            path = fit_model(v, ctx)
            results.append(path)
        except Exception:
            import traceback
            logger.error("Model %s failed:\n%s", v, traceback.format_exc())
            raise  # Re-raise so the notebook shows the real error
    return results
