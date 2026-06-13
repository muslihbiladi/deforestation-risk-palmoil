import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def test_compute_vif_writes_json(tmp_path):
    from palmdef_risk.model.diagnostics import compute_vif
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "altitude": rng.normal(0, 1, n),
        "slope": rng.normal(0, 1, n),
        "gravity_resid": rng.normal(0, 1, n),
    })
    sample = tmp_path / "sample.csv"
    df.to_csv(sample, index=False)
    out = tmp_path / "vif.json"
    compute_vif(["altitude", "slope", "gravity_resid"], sample, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "altitude" in data
    assert data["altitude"] < 3.0


def test_compute_vif_flags_high_vif(tmp_path, caplog):
    import logging
    from palmdef_risk.model.diagnostics import compute_vif
    rng = np.random.default_rng(42)
    n = 200
    base = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "x1": base,
        "x2": base + rng.normal(0, 0.01, n),
    })
    sample = tmp_path / "sample.csv"
    df.to_csv(sample, index=False)
    out = tmp_path / "vif.json"
    with caplog.at_level(logging.WARNING, logger="palmdef_risk"):
        compute_vif(["x1", "x2"], sample, out)
    assert any("VIF" in r.message for r in caplog.records)


def test_morans_i_output_has_required_keys(tmp_path):
    from palmdef_risk.model.diagnostics import compute_morans_i
    rng = np.random.default_rng(0)
    residuals = {
        "A": rng.normal(0, 1, 100),
        "B": rng.normal(0, 0.5, 100),
    }
    # coords is per-variant (each variant may drop a different NaN subset)
    coords_list = [(i * 30, j * 30) for i in range(10) for j in range(10)]
    coords = {"A": coords_list, "B": coords_list}
    out = tmp_path / "moran.json"
    compute_morans_i(residuals, coords, out)
    assert out.exists()
    data = json.loads(out.read_text())
    for v in ["A", "B"]:
        assert v in data
        assert "I" in data[v]
        assert "p_value" in data[v]
