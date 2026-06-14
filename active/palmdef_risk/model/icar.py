from __future__ import annotations
import logging
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from palmdef_risk.parallel import run_parallel

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)

_LOG_DIST_COLS = ["dist_defor", "dist_edge", "dist_road", "dist_town", "dist_river"]

_BASE_SCALED_COLS = ["altitude", "slope"] + [f"log_{c}" for c in _LOG_DIST_COLS]

# Covariates each variant adds beyond the biophysical base. Single source of truth
# for both the formula RHS and the NaN-drop subset used in fit/residuals/predict.
_VARIANT_EXTRA_COLS: dict[str, list[str]] = {
    "A": [],
    "B": ["gravity_resid"],
    "C": ["plantation_resid"],
    "D": ["gravity_resid", "plantation_resid"],
    "E": ["gravity_resid", "plantation_resid", "hgu_b1", "hgu_b2"],
}

# Full scaled-covariate list per variant (order determines column order in X).
_VARIANT_SCALED_COLS: dict[str, list[str]] = {
    v: _BASE_SCALED_COLS + extra for v, extra in _VARIANT_EXTRA_COLS.items()
}


def variant_extra_cols(variant: str) -> list[str]:
    """Covariates a variant adds beyond the biophysical base.

    Single source of truth for the NaN-drop subset consumed by _build_and_fit,
    diagnostics.compute_residuals_all, and reports._predict_in_sample.
    """
    if variant not in _VARIANT_EXTRA_COLS:
        raise ValueError(
            f"Unknown variant: {variant!r}. Valid variants: A, B, C, D, E"
        )
    return list(_VARIANT_EXTRA_COLS[variant])


def base_dropna_cols() -> list[str]:
    """Columns that must be non-NaN before the design matrix is built.

    Single source for the base dropna subset shared by fit (_build_and_fit),
    diagnostics, and reports — previously hard-coded identically in each.
    Variant-specific extras come from variant_extra_cols().
    """
    return ["fcc23", "altitude", "slope", "protected", "cell"] + [
        f"log_{c}" for c in _LOG_DIST_COLS
    ]


def build_formula(variant: str, data: pd.DataFrame) -> str:
    """Build the forestatrisk formula for variant A, B, C, D, or E.

    Constant columns (min == max) are excluded automatically: patsy's scale()
    divides by std, which for a near-zero std produces extreme / NaN values that
    cause NA_action='drop' to remove all rows.
    """
    all_scaled = _VARIANT_SCALED_COLS.get(variant)
    if all_scaled is None:
        raise ValueError(f"Unknown variant: {variant!r}. Valid variants: A, B, C, D, E")

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


def load_design_matrix(
    ctx: "RunContext",
    variant: str,
    formula: str,
    dropna: str = "base",
) -> tuple[pd.DataFrame, "np.ndarray", "np.ndarray"]:
    """Rebuild the patsy design matrices for a fitted variant from sample.csv.

    Reads ``<ctx.output_dir>/sample.csv``, applies prepare_sample(), drops NaN
    rows per ``dropna``, then builds (y, x) from ``formula``. This is the
    canonical "rebuild from sample.csv at predict time" path — it never pickles
    or loads a patsy DesignInfo (project rule).

    Single source for the sample-load + design-matrix logic previously
    triplicated across diagnostics, reports, and predict (×2).

    dropna:
      ``"base"``   – drop NaN on base_dropna_cols() + variant_extra_cols(variant);
                     matches fit_model. Used by diagnostics and reports.
      ``"scaled"`` – drop NaN only on the scale()-wrapped columns parsed from
                     ``formula``, so scale() statistics match what fit_model
                     computed. Used by predict (the row subset must equal the
                     fit-time subset for the scale means/stds to line up).

    Returns ``(data, y, x)``: the surviving DataFrame and the patsy y / x
    matrices (each carrying a fresh ``.design_info``).
    """
    from patsy import dmatrices

    data = pd.read_csv(ctx.output_dir / "sample.csv")
    data = prepare_sample(data)

    if dropna == "base":
        subset = base_dropna_cols() + variant_extra_cols(variant)
        data = data.dropna(subset=subset)
    elif dropna == "scaled":
        scaled_cols = re.findall(r"scale\((\w+)\)", formula)
        if scaled_cols:
            data = data.dropna(subset=scaled_cols)
    else:
        raise ValueError(
            f"Unknown dropna mode: {dropna!r}. Use 'base' or 'scaled'."
        )

    y, x = dmatrices(formula, data, return_type="matrix")
    return data, y, x


def _build_and_fit(
    variant: str,
    ctx: "RunContext",
    data: pd.DataFrame,
    nneigh: np.ndarray | None = None,
    adj: np.ndarray | None = None,
) -> dict:
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
    extra_cols = variant_extra_cols(variant)
    n_before = len(data)
    data = data.dropna(subset=base_dropna_cols() + extra_cols)
    if (n_dropped := n_before - len(data)):
        logger.warning("Dropped %d rows with NaN in formula columns", n_dropped)

    formula = build_formula(variant, data)

    cfg = ctx.config
    if nneigh is None or adj is None:
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


def fit_model(
    variant: str,
    ctx: "RunContext",
    nneigh: np.ndarray | None = None,
    adj: np.ndarray | None = None,
) -> Path:
    """Fit one iCAR model variant. Returns path to saved .pkl file."""
    sample_path = ctx.output_dir / "sample.csv"
    data = pd.read_csv(sample_path)
    state = _build_and_fit(variant, ctx, data, nneigh=nneigh, adj=adj)

    model_dir = ctx.output_dir / "models" / f"model_{variant}"
    model_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = model_dir / f"mod_{variant}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(state, f)
    logger.info("Model %s fitted and saved to %s", variant, pkl_path)
    return pkl_path


def _fit_one_variant(task: tuple) -> str:
    """Module-level worker: fit one variant in its own process.

    Data crosses the pool boundary as paths, never live objects (CLAUDE.md
    "parallelism"): the run dir is reloaded into a RunContext, and the shared,
    variant-invariant cellneigh adjacency is loaded from .npy rather than pickled
    through the pool. Honors the pkl skip-guard. Re-raises on failure so fit_all
    stays fail-fast (run_parallel propagates the exception via future.result()).
    """
    variant, run_dir, nneigh_path, adj_path = task
    from palmdef_risk.io.run import load_run
    ctx = load_run(run_dir)
    pkl_path = ctx.output_dir / "models" / f"model_{variant}" / f"mod_{variant}.pkl"
    if pkl_path.exists():
        logger.info("mod_%s.pkl exists — skipping fit", variant)
        return str(pkl_path)
    nneigh = np.load(nneigh_path)
    adj = np.load(adj_path)
    try:
        return str(fit_model(variant, ctx, nneigh=nneigh, adj=adj))
    except Exception:
        import traceback
        logger.error("Model %s failed:\n%s", variant, traceback.format_exc())
        raise  # Re-raise so the notebook shows the real error (fail-fast)


def fit_all(ctx: "RunContext") -> list[Path]:
    """Fit all configured model variants in parallel via run_parallel.

    cellneigh is computed once (variant-invariant) and persisted to .npy so each
    worker loads the shared adjacency by path. One process per variant, bounded
    by ram_per_icar_gb. A worker failure propagates (fail-fast), matching the
    previous sequential loop.
    """
    import forestatrisk as far
    cfg = ctx.config
    variants = list(cfg.model_variants)
    nneigh, adj = far.cellneigh(
        raster=str(ctx.data_dir / "fcc23.tif"),
        csize=cfg.csize,
        rank=1,
    )
    models_dir = ctx.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    nneigh_path = models_dir / "_cellneigh_nneigh.npy"
    adj_path = models_dir / "_cellneigh_adj.npy"
    np.save(nneigh_path, np.asarray(nneigh))
    np.save(adj_path, np.asarray(adj))

    tasks = [
        (v, str(ctx.run_dir), str(nneigh_path), str(adj_path)) for v in variants
    ]
    try:
        paths = run_parallel(
            _fit_one_variant, tasks,
            ram_per_task_gb=cfg.ram_per_icar_gb, cfg=cfg,
            desc="Fitting iCAR models",
        )
    finally:
        nneigh_path.unlink(missing_ok=True)
        adj_path.unlink(missing_ok=True)
    return [Path(p) for p in paths]
