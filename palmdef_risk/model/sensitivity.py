from __future__ import annotations
import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from palmdef_risk.process.gravity import orthogonalize_gravity_ctx
from palmdef_risk.model.icar import fit_model, build_formula

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)


def compute_gravity_raw(ctx: "RunContext", sigma_km: float) -> Path:
    """Compute gravity_raw.tif at a given sigma (overwrites ctx.data_dir/gravity_raw.tif)."""
    from palmdef_risk.process.gravity import _apply_gaussian_filter
    from palmdef_risk.io.helpers import get_mask_properties, rasterize_vector
    d = ctx.data_dir
    ref = d / "forest_t2.tif"
    mill_gpkg = ctx.raw_dir / "mill" / "mill_t2.gpkg"
    mask_props = get_mask_properties(str(ref))
    tmp = d / "_mill_density_sensitivity_tmp.tif"
    rasterize_vector(str(mill_gpkg), str(tmp), burn_value=1, mask_props=mask_props)
    out = d / "gravity_raw.tif"
    _apply_gaussian_filter(tmp, out, sigma_km=sigma_km, radius_km=ctx.config.radius_km)
    tmp.unlink(missing_ok=True)
    return out


def _load_model(pkl_path: Path) -> object:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def run_gravity_sensitivity(ctx: "RunContext") -> Path:
    """
    For each sigma in config.sensitivity_sigmas: refit Model B, extract
    accessibility coefficient + deviance. Writes gravity_sensitivity.json.
    """
    out_dir = ctx.output_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "gravity_sensitivity.json"

    d = ctx.data_dir
    backup_raw = d / "_gravity_raw_backup.tif"
    backup_resid = d / "_gravity_resid_backup.tif"
    if (d / "gravity_raw.tif").exists():
        shutil.copy2(d / "gravity_raw.tif", backup_raw)
    if (d / "gravity_resid.tif").exists():
        shutil.copy2(d / "gravity_resid.tif", backup_resid)

    results = []
    for sigma in ctx.config.sensitivity_sigmas:
        logger.info("Gravity sensitivity: sigma=%.0f km", sigma)
        compute_gravity_raw(ctx, sigma_km=sigma)
        orthogonalize_gravity_ctx(ctx)
        pkl = fit_model("B", ctx)
        state = _load_model(pkl)
        formula = build_formula("B", ctx)
        coef_idx = _gravity_coef_index(formula, state)
        betas = state.betas if hasattr(state, "betas") else state.get("betas", [])
        deviance = state.deviance if hasattr(state, "deviance") else state.get("deviance", [])
        entry = {
            "sigma_km": sigma,
            "accessibility_coef": float(betas[coef_idx]) if coef_idx is not None and coef_idx < len(betas) else None,
            "mean_deviance": float(np.mean(deviance)) if hasattr(deviance, "__len__") else float(deviance),
        }
        results.append(entry)

    if backup_raw.exists():
        shutil.move(str(backup_raw), d / "gravity_raw.tif")
    if backup_resid.exists():
        shutil.move(str(backup_resid), d / "gravity_resid.tif")

    out_json.write_text(json.dumps(results, indent=2))
    logger.info("Gravity sensitivity written to %s", out_json)
    return out_json


def _gravity_coef_index(formula: str, state: object) -> int | None:
    """Find index of gravity_resid beta in the betas array."""
    try:
        terms = formula.split("~")[1].split("+")
        terms = [t.strip() for t in terms if "cell" not in t]
        for i, t in enumerate(terms):
            if "gravity_resid" in t:
                return i + 1  # +1 for intercept
    except Exception:
        pass
    return None
