from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any
import json

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext


def compute_moran(
    residuals: np.ndarray, coords: np.ndarray, ctx: RunContext
) -> Dict[str, Any]:
    """Compute Moran's I on ICAR residuals. coords: (n, 2) array."""
    from scipy.spatial.distance import cdist

    D = cdist(coords, coords).astype(float)
    np.fill_diagonal(D, np.inf)
    k = min(8, len(coords) - 1)
    kth_dist = np.sort(D, axis=1)[:, k - 1]
    W = (D <= kth_dist[:, None]).astype(float)
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums

    n = len(residuals)
    r = residuals - residuals.mean()
    num = n * float(r @ W @ r)
    den = float(W.sum()) * float(r @ r)
    moran_i = num / den if den != 0 else 0.0

    interpretation = "positive" if moran_i > 0.05 else "negligible"
    result: Dict[str, Any] = {"moran_i": moran_i, "interpretation": interpretation}

    out = ctx.output_dir / "diagnostics" / "moran.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


def check_beta_stability(
    mod_with: object, mod_without: object, coef_name: str
) -> Dict[str, Any]:
    """Compare beta for coef_name between model with and without cell term."""
    beta_with = _extract_beta(mod_with, coef_name)
    beta_without = _extract_beta(mod_without, coef_name)
    if beta_without == 0:
        shift_pct = float("inf")
    else:
        shift_pct = abs(beta_with - beta_without) / abs(beta_without) * 100.0
    return {
        "coef": coef_name,
        "beta_with_cell": beta_with,
        "beta_without_cell": beta_without,
        "shift_pct": shift_pct,
        "confounder_warning": shift_pct > 20,
    }


def _extract_beta(mod: object, coef_name: str) -> float:
    names = getattr(mod, "betas_names", [])
    betas = getattr(mod, "betas", [])
    if coef_name in names:
        return float(betas[names.index(coef_name)])
    return 0.0


def check_vif(data: pd.DataFrame, terms: list) -> Dict[str, float]:
    """Variance Inflation Factor for each column in terms."""
    cols = [t for t in terms if t in data.columns]
    X = data[cols].dropna().values.astype(float)
    vifs: Dict[str, float] = {}
    for i, col in enumerate(cols):
        y = X[:, i]
        Xr = np.delete(X, i, axis=1)
        Xr = np.column_stack([np.ones(len(Xr)), Xr])
        beta, _, _, _ = np.linalg.lstsq(Xr, y, rcond=None)
        y_hat = Xr @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
    return vifs


def compile_dic_table(results: Dict[str, Dict], ctx: RunContext) -> pd.DataFrame:
    """Build DIC comparison table across all fitted variants and save to CSV."""
    rows = []
    for variant, info in results.items():
        mod = info.get("model")
        dic = float(mod.DIC) if (mod is not None and hasattr(mod, "DIC")) else None
        rows.append({"variant": variant, "DIC": dic})
    df = pd.DataFrame(rows).sort_values("DIC").reset_index(drop=True)
    out = ctx.output_dir / "diagnostics" / "dic_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
