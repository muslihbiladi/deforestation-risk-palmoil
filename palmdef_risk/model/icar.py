from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import pickle

import pandas as pd

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

_BASELINE_RHS = (
    "scale(altitude) + scale(slope)"
    " + scale(log(dist_defor + 1))"
    " + scale(log(dist_edge + 1))"
    " + scale(log(dist_road + 1))"
    " + scale(log(dist_town + 1))"
    " + scale(log(dist_river + 1))"
)


def build_formula(variant: str, ctx: RunContext) -> str:
    """Return the forestatrisk suitab_formula string for the given variant."""
    d = ctx.data_dir

    def ex(name: str) -> bool:
        return (d / name).exists()

    hgu_term = "hgu" if ex("hgu.tif") else None
    if ex("peatland.tif"):
        pt = ctx.config.peatland_type
        peatland_term = "scale(peatland_depth)" if pt == "continuous" else "peatland"
    else:
        peatland_term = None

    lq_col = "lq_mp" if ctx.config.lq_direction == "mp" else "lq_pm"
    dm = "scale(log(dist_mill + 1))" if ex("dist_mill.tif") else None
    dp = "scale(log(dist_plantation_edge + 1))" if ex("dist_plantation_edge.tif") else None
    M_term = "scale(mill_kde)" if ex("mill_kde.tif") else None
    P_term = "scale(plantation_surface)" if ex("plantation_surface.tif") else None
    lq_term = f"scale({lq_col})" if ex(f"{lq_col}.tif") else None
    lq_sq = "scale(lq_sq)" if ex("lq_sq.tif") else None

    def _join(*terms):
        return " + ".join(t for t in terms if t)

    if variant == "A":
        extra = ""
    elif variant == "B":
        extra = _join(hgu_term, peatland_term)
    elif variant == "C":
        extra = _join(dm, dp, M_term, P_term)
    elif variant == "D":
        extra = _join(hgu_term, peatland_term, dm, dp, M_term, P_term)
    elif variant == "E":
        extra = _join(hgu_term, peatland_term, dm, dp, lq_term)
    elif variant == "F":
        extra = _join(hgu_term, peatland_term, dm, dp, lq_term, lq_sq)
    elif variant == "G":
        iact = []
        if lq_term and hgu_term:
            iact.append("scale(lq_hgu)")
        if lq_term and peatland_term and peatland_term == "peatland":
            iact.append("scale(lq_peatland)")
        extra = _join(hgu_term, peatland_term, dm, dp, lq_term, lq_sq, *iact)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    rhs = _BASELINE_RHS + (f" + {extra}" if extra else "")
    return f"I(1 - fcc23) + trial ~ {rhs} + protected + cell"


def _add_interactions(data: pd.DataFrame, lq_col: str) -> pd.DataFrame:
    """Pre-compute LQ x policy interaction columns needed by variant G."""
    if lq_col not in data.columns:
        return data
    data = data.copy()
    lq = data[lq_col]
    if "hgu" in data.columns:
        data["lq_hgu"] = lq * data["hgu"]
    if "peatland" in data.columns:
        data["lq_peatland"] = lq * data["peatland"]
    return data


def fit_icar(ctx: RunContext, variant: str, data: pd.DataFrame) -> object:
    """Fit a forestatrisk ICAR model for one variant and pickle it."""
    import forestatrisk as far

    lq_col = "lq_mp" if ctx.config.lq_direction == "mp" else "lq_pm"
    if variant == "G":
        data = _add_interactions(data, lq_col)

    formula = build_formula(variant, ctx)
    cfg = ctx.config

    model_dir = ctx.output_dir / "models" / f"model_{variant}"
    model_dir.mkdir(parents=True, exist_ok=True)

    mod = far.model_binomial_iCAR(
        suitab_formula=formula,
        data=data,
        n_iter=cfg.mcmc,
        n_burn=cfg.burnin,
        thin=cfg.thin,
        save_rho=True,
    )

    mod_path = model_dir / f"mod_{variant}.pkl"
    with open(mod_path, "wb") as fh:
        pickle.dump(mod, fh)

    return mod
