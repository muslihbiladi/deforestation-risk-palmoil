import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock


class _FakeConfig:
    run_gwr = True


class _FakeCtx:
    def __init__(self, tmp_path):
        self.output_dir = Path(tmp_path) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = _FakeConfig()


def _make_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "y": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "x": rng.uniform(100, 200, n),
        "y_coord": rng.uniform(-10, 0, n),
    })


def test_run_gwr_skipped_when_disabled(tmp_path):
    from palmoil_risk.model.gwr import run_gwr

    ctx = _FakeCtx(tmp_path)
    ctx.config.run_gwr = False

    result = run_gwr(ctx, _make_data(), "y", ["x1", "x2"],
                     coords_cols=("x", "y_coord"))
    assert result is None


def test_run_gwr_skipped_when_mgwr_missing(tmp_path):
    from palmoil_risk.model.gwr import run_gwr

    ctx = _FakeCtx(tmp_path)

    with patch.dict("sys.modules", {"mgwr": None, "mgwr.gwr": None, "mgwr.sel_bw": None}):
        import importlib
        import palmoil_risk.model.gwr as gwr_mod
        importlib.reload(gwr_mod)
        result = gwr_mod.run_gwr(ctx, _make_data(), "y", ["x1", "x2"],
                                 coords_cols=("x", "y_coord"))
    assert result is None


def test_run_gwr_skipped_when_too_few_rows(tmp_path):
    from palmoil_risk.model.gwr import run_gwr

    ctx = _FakeCtx(tmp_path)
    small_df = _make_data(n=10)

    mock_gwr_results = MagicMock()
    mock_gwr_results.params = np.zeros((10, 3))
    mock_gwr_results.aicc = 100.0
    mock_gwr_results.R2 = 0.5

    mock_gwr = MagicMock()
    mock_gwr.return_value.fit.return_value = mock_gwr_results
    mock_sel = MagicMock()
    mock_sel.return_value.search.return_value = 50000.0

    with patch.dict("sys.modules", {
        "mgwr": MagicMock(),
        "mgwr.gwr": MagicMock(GWR=mock_gwr),
        "mgwr.sel_bw": MagicMock(Sel_BW=mock_sel),
    }):
        result = run_gwr(ctx, small_df, "y", ["x1", "x2"],
                         coords_cols=("x", "y_coord"))
    assert result is None


def test_run_gwr_writes_outputs(tmp_path):
    from palmoil_risk.model.gwr import run_gwr

    ctx = _FakeCtx(tmp_path)
    df = _make_data(n=100)
    n = len(df)

    mock_gwr_results = MagicMock()
    mock_gwr_results.params = np.zeros((n, 3))
    mock_gwr_results.aicc = 250.5
    mock_gwr_results.R2 = 0.42

    mock_gwr_cls = MagicMock()
    mock_gwr_cls.return_value.fit.return_value = mock_gwr_results
    mock_sel_cls = MagicMock()
    mock_sel_cls.return_value.search.return_value = 30000.0

    with patch.dict("sys.modules", {
        "mgwr": MagicMock(),
        "mgwr.gwr": MagicMock(GWR=mock_gwr_cls),
        "mgwr.sel_bw": MagicMock(Sel_BW=mock_sel_cls),
    }):
        result = run_gwr(ctx, df, "y", ["x1", "x2"],
                         coords_cols=("x", "y_coord"))

    assert result is not None
    assert result["bandwidth"] == pytest.approx(30000.0)
    assert result["aicc"] == pytest.approx(250.5)
    assert result["r2"] == pytest.approx(0.42)
    assert result["n"] == n
    assert (ctx.output_dir / "gwr" / "gwr_coefficients.csv").exists()
    assert (ctx.output_dir / "gwr" / "gwr_summary.json").exists()
