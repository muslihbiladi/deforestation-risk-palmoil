from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)

_BASELINE_RHS = (
    "scale(altitude)"
    " + scale(slope)"
    " + scale(log(dist_defor + 1))"
    " + scale(log(dist_edge + 1))"
    " + scale(log(dist_road + 1))"
    " + scale(log(dist_town + 1))"
    " + scale(log(dist_river + 1))"
    " + protected"
)

_HGU_SPLINE = "scale(hgu_b1) + scale(hgu_b2)"


def build_formula(variant: str, ctx: "RunContext") -> str:
    """Return the forestatrisk suitab_formula string for model variant A, B, or C."""
    if variant == "A":
        rhs = _BASELINE_RHS
    elif variant == "B":
        rhs = _BASELINE_RHS + " + scale(gravity_resid)"
    elif variant == "C":
        rhs = _BASELINE_RHS + " + scale(gravity_resid) + " + _HGU_SPLINE
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Valid variants: A, B, C")
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
    """Add derived columns (HGU spline basis) to the sample DataFrame."""
    data = _add_hgu_spline_cols(data)
    return data


def fit_model(variant: str, ctx: "RunContext") -> Path:
    """Fit one iCAR model variant. Returns path to saved .pkl file."""
    import forestatrisk as far
    formula = build_formula(variant, ctx)
    sample_path = ctx.output_dir / "sample.csv"
    data = pd.read_csv(sample_path)
    data = prepare_sample(data)

    model_dir = ctx.output_dir / "models" / f"model_{variant}"
    model_dir.mkdir(parents=True, exist_ok=True)

    cfg = ctx.config
    mod = far.model_icar(
        suitab_formula=formula,
        data=data,
        n_neighbors=1,
        Vbeta=cfg.Vbeta,
        beta_start=-99,
        burnin=cfg.burnin,
        mcmc=cfg.mcmc,
        thin=cfg.thin,
        seed=cfg.seed,
        save_rho=True,
        verbose=False,
    )

    pkl_path = model_dir / f"mod_{variant}.pkl"
    safe_state = {
        "betas": mod.betas,
        "rho": mod.rho,
        "formula": formula,
        "betas_mcmc": mod.betas_mcmc,
        "deviance": mod.deviance,
        "variant": variant,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(safe_state, f)
    logger.info("Model %s fitted and saved to %s", variant, pkl_path)
    return pkl_path


def fit_all(ctx: "RunContext") -> list[Path]:
    """Fit all configured model variants in parallel."""
    from palmdef_risk.parallel import run_parallel
    variants = ctx.config.model_variants
    tasks = [(v, ctx) for v in variants]
    results = run_parallel(_fit_worker, tasks,
                           ram_per_task_gb=ctx.config.ram_per_icar_gb, cfg=ctx.config)
    return [r for r in results if r is not None]


def _fit_worker(args: tuple) -> Path | None:
    variant, ctx = args
    try:
        return fit_model(variant, ctx)
    except Exception as e:
        logger.error("Model %s failed: %s", variant, e)
        return None
