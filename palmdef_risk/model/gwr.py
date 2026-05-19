from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, List
import logging

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

log = logging.getLogger(__name__)


def run_gwr(
    ctx: RunContext,
    data: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    coords_cols: tuple = ("x", "y"),
    bandwidth: Optional[float] = None,
) -> Optional[Dict]:
    """Fit GWR on sample data. Returns results dict or None if disabled.

    Skipped when ctx.config.run_gwr is False. Requires mgwr package.
    bandwidth: fixed in metres; if None, golden-section search is used.
    """
    if not ctx.config.run_gwr:
        log.info("GWR skipped (run_gwr=False)")
        return None

    try:
        from mgwr.gwr import GWR
        from mgwr.sel_bw import Sel_BW
    except ImportError as exc:
        log.warning("GWR skipped: mgwr not installed (%s)", exc)
        return None

    cols_needed = [y_col] + list(x_cols) + list(coords_cols)
    df = data[cols_needed].dropna()
    if len(df) < 50:
        log.warning("GWR skipped: fewer than 50 valid rows after dropna")
        return None

    y = df[[y_col]].values
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in x_cols])
    coords = list(zip(df[coords_cols[0]], df[coords_cols[1]]))

    if bandwidth is None:
        selector = Sel_BW(coords, y, X)
        bandwidth = selector.search(criterion="AICc")

    model = GWR(coords, y, X, bw=bandwidth)
    results = model.fit()

    out_dir = ctx.output_dir / "gwr"
    out_dir.mkdir(parents=True, exist_ok=True)

    coef_cols = ["intercept"] + list(x_cols)
    coef_df = pd.DataFrame(results.params, columns=coef_cols)
    coef_df["x"] = df[coords_cols[0]].values
    coef_df["y"] = df[coords_cols[1]].values
    coef_path = out_dir / "gwr_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    summary = {
        "bandwidth": float(bandwidth),
        "aicc": float(results.aicc),
        "r2": float(results.R2),
        "n": int(len(df)),
        "coef_path": str(coef_path),
    }

    import json
    (out_dir / "gwr_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("GWR done: bandwidth=%.1f AICc=%.2f R2=%.4f", bandwidth, results.aicc, results.R2)
    return summary
