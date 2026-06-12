import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from palmdef_risk.model.icar import build_formula


_ALL_COLS = [
    "altitude", "slope",
    "log_dist_defor", "log_dist_edge", "log_dist_road", "log_dist_town", "log_dist_river",
    "gravity_resid", "plantation_resid", "hgu_b1", "hgu_b2",
]


def _sample_df(constant_col: str | None = None) -> pd.DataFrame:
    """Minimal DataFrame with two rows; optionally make one column constant."""
    data = {c: [1.0, 2.0] for c in _ALL_COLS}
    if constant_col and constant_col in data:
        data[constant_col] = [5.0, 5.0]
    return pd.DataFrame(data)


def test_formula_a_baseline_covariates():
    f = build_formula("A", _sample_df())
    assert "scale(altitude)" in f
    assert "scale(slope)" in f
    assert "log_dist_defor" in f
    assert "log_dist_edge" in f
    assert "log_dist_road" in f
    assert "log_dist_town" in f
    assert "log_dist_river" in f
    assert "protected" in f
    assert "cell" in f


def test_formula_a_no_gravity():
    f = build_formula("A", _sample_df())
    assert "gravity_resid" not in f


def test_formula_b_adds_gravity():
    f = build_formula("B", _sample_df())
    assert "scale(gravity_resid)" in f
    assert "hgu_b1" not in f


def test_formula_c_adds_plantation_not_gravity():
    f = build_formula("C", _sample_df())
    assert "scale(plantation_resid)" in f
    assert "gravity_resid" not in f
    assert "hgu_b1" not in f


def test_formula_d_adds_gravity_and_plantation():
    f = build_formula("D", _sample_df())
    assert "scale(gravity_resid)" in f
    assert "scale(plantation_resid)" in f
    assert "hgu_b1" not in f


def test_formula_e_adds_hgu_spline_and_both_access():
    f = build_formula("E", _sample_df())
    assert "scale(gravity_resid)" in f
    assert "scale(plantation_resid)" in f
    assert "hgu_b1" in f
    assert "hgu_b2" in f


def test_formula_no_dist_mill_in_any_variant():
    for v in ["A", "B", "C", "D", "E"]:
        assert "dist_mill" not in build_formula(v, _sample_df())


def test_formula_no_lq_terms():
    for v in ["A", "B", "C", "D", "E"]:
        f = build_formula(v, _sample_df())
        assert "lq" not in f.lower()
        assert "kde" not in f.lower()


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_formula("Z", _sample_df())


def test_variant_extra_cols():
    from palmdef_risk.model.icar import variant_extra_cols
    assert variant_extra_cols("A") == []
    assert variant_extra_cols("B") == ["gravity_resid"]
    assert variant_extra_cols("C") == ["plantation_resid"]
    assert variant_extra_cols("D") == ["gravity_resid", "plantation_resid"]
    assert variant_extra_cols("E") == ["gravity_resid", "plantation_resid", "hgu_b1", "hgu_b2"]


def test_response_lhs():
    f = build_formula("A", _sample_df())
    assert f.startswith("I(1 - fcc23) + trial ~")


def test_constant_column_excluded():
    """Constant columns must be excluded (patsy scale() would divide by zero)."""
    f = build_formula("A", _sample_df(constant_col="altitude"))
    assert "scale(altitude)" not in f
    assert "scale(slope)" in f
