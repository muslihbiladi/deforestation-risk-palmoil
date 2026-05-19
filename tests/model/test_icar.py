import pytest
from pathlib import Path
from palmoil_risk.model.icar import build_formula, _add_interactions
import pandas as pd
import numpy as np


class _FakeConfig:
    lq_direction = "mp"
    peatland_type = "binary"
    mcmc = 10
    burnin = 5
    thin = 1


class _FakeCtx:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(data_dir) / "output"
        self.config = _FakeConfig()


def test_variant_a_baseline_only(tmp_path):
    ctx = _FakeCtx(tmp_path)
    f = build_formula("A", ctx)
    assert f.startswith("I(1 - fcc23) + trial ~")
    assert "altitude" in f
    assert "slope" in f
    assert "hgu" not in f
    assert "lq" not in f
    assert f.endswith("+ protected + cell")


def test_variant_b_adds_policy_only_when_files_exist(tmp_path):
    ctx = _FakeCtx(tmp_path)
    f_no_files = build_formula("B", ctx)
    assert "hgu" not in f_no_files
    (tmp_path / "hgu.tif").touch()
    f_with_hgu = build_formula("B", ctx)
    assert "hgu" in f_with_hgu


def test_variant_b_peatland_continuous(tmp_path):
    ctx = _FakeCtx(tmp_path)
    ctx.config.peatland_type = "continuous"
    (tmp_path / "peatland.tif").touch()
    f = build_formula("B", ctx)
    assert "scale(peatland_depth)" in f
    assert "peatland_depth" in f


def test_variant_c_raw_palm(tmp_path):
    ctx = _FakeCtx(tmp_path)
    (tmp_path / "dist_mill.tif").touch()
    (tmp_path / "dist_plantation_edge.tif").touch()
    (tmp_path / "mill_kde.tif").touch()
    (tmp_path / "plantation_surface.tif").touch()
    f = build_formula("C", ctx)
    assert "dist_mill" in f
    assert "mill_kde" in f
    assert "plantation_surface" in f
    assert "hgu" not in f


def test_variant_g_adds_interaction_terms(tmp_path):
    ctx = _FakeCtx(tmp_path)
    for fname in ["hgu.tif", "peatland.tif", "lq_mp.tif", "lq_sq.tif",
                  "dist_mill.tif", "dist_plantation_edge.tif"]:
        (tmp_path / fname).touch()
    f = build_formula("G", ctx)
    assert "scale(lq_hgu)" in f
    assert "scale(lq_peatland)" in f
    assert "scale(lq_sq)" in f


def test_add_interactions_creates_columns():
    df = pd.DataFrame({
        "lq_mp": [1.0, 2.0, 3.0],
        "hgu": [1, 0, 1],
        "peatland": [0, 1, 1],
    })
    result = _add_interactions(df.copy(), "lq_mp")
    assert "lq_hgu" in result.columns
    assert "lq_peatland" in result.columns
    np.testing.assert_array_almost_equal(result["lq_hgu"], [1.0, 0.0, 3.0])
    np.testing.assert_array_almost_equal(result["lq_peatland"], [0.0, 2.0, 3.0])


def test_unknown_variant_raises(tmp_path):
    ctx = _FakeCtx(tmp_path)
    with pytest.raises(ValueError, match="Unknown variant"):
        build_formula("Z", ctx)
