import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class _FakeCtx:
    def __init__(self, tmp_path):
        self.output_dir = Path(tmp_path)


class _FakeMod:
    def __init__(self, betas_names, betas, dic=None):
        self.betas_names = betas_names
        self.betas = betas
        if dic is not None:
            self.DIC = dic


def test_compute_moran_positive_autocorrelation(tmp_path):
    from palmoil_risk.model.diagnostics import compute_moran

    ctx = _FakeCtx(tmp_path)
    # 9 points in a 3x3 grid; residuals identical within each row → positive autocorrelation
    coords = np.array([[i, j] for i in range(3) for j in range(3)], dtype=float)
    residuals = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

    result = compute_moran(residuals, coords, ctx)

    assert "moran_i" in result
    assert "interpretation" in result
    moran_json = (tmp_path / "diagnostics" / "moran.json").read_text()
    import json
    parsed = json.loads(moran_json)
    assert parsed["moran_i"] == pytest.approx(result["moran_i"])


def test_check_beta_stability_flags_large_shift(tmp_path):
    from palmoil_risk.model.diagnostics import check_beta_stability

    mod_with = _FakeMod(["dist_mill"], [3.0])
    mod_without = _FakeMod(["dist_mill"], [2.0])
    result = check_beta_stability(mod_with, mod_without, "dist_mill")

    assert result["shift_pct"] == pytest.approx(50.0)
    assert result["confounder_warning"] is True


def test_check_beta_stability_no_warning_small_shift(tmp_path):
    from palmoil_risk.model.diagnostics import check_beta_stability

    mod_with = _FakeMod(["altitude"], [1.05])
    mod_without = _FakeMod(["altitude"], [1.0])
    result = check_beta_stability(mod_with, mod_without, "altitude")

    assert result["shift_pct"] == pytest.approx(5.0)
    assert result["confounder_warning"] is False


def test_check_vif_collinear_columns():
    from palmoil_risk.model.diagnostics import check_vif

    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    df = pd.DataFrame({"a": x, "b": x + rng.standard_normal(200) * 0.01})
    vifs = check_vif(df, ["a", "b"])

    assert vifs["a"] > 5
    assert vifs["b"] > 5


def test_check_vif_independent_columns():
    from palmoil_risk.model.diagnostics import check_vif

    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "x1": rng.standard_normal(200),
        "x2": rng.standard_normal(200),
    })
    vifs = check_vif(df, ["x1", "x2"])

    assert vifs["x1"] < 5
    assert vifs["x2"] < 5


def test_compile_dic_table_creates_csv(tmp_path):
    from palmoil_risk.model.diagnostics import compile_dic_table

    ctx = _FakeCtx(tmp_path)
    results = {
        "A": {"model": _FakeMod([], [], dic=450.2)},
        "B": {"model": _FakeMod([], [], dic=430.7)},
    }
    df = compile_dic_table(results, ctx)

    assert len(df) == 2
    assert df.iloc[0]["DIC"] == pytest.approx(430.7)
    csv_path = tmp_path / "diagnostics" / "dic_table.csv"
    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert list(loaded["variant"]) == ["B", "A"]
