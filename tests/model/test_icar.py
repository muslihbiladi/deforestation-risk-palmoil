import pytest
from unittest.mock import MagicMock
from palmdef_risk.model.icar import build_formula


def _ctx():
    ctx = MagicMock()
    ctx.config.peatland_type = "binary"
    ctx.config.Vbeta = 1000
    return ctx


def test_formula_a_baseline_covariates():
    f = build_formula("A", _ctx())
    assert "scale(altitude)" in f
    assert "scale(slope)" in f
    assert "dist_defor" in f
    assert "dist_edge" in f
    assert "dist_road" in f
    assert "dist_town" in f
    assert "dist_river" in f
    assert "protected" in f
    assert "cell" in f


def test_formula_a_no_gravity():
    f = build_formula("A", _ctx())
    assert "gravity_resid" not in f


def test_formula_b_adds_gravity():
    f = build_formula("B", _ctx())
    assert "scale(gravity_resid)" in f
    assert "hgu_b1" not in f


def test_formula_c_adds_hgu_spline():
    f = build_formula("C", _ctx())
    assert "hgu_b1" in f
    assert "hgu_b2" in f
    assert "scale(gravity_resid)" in f


def test_formula_no_dist_mill_in_any_variant():
    for v in ["A", "B", "C"]:
        assert "dist_mill" not in build_formula(v, _ctx())


def test_formula_no_lq_terms():
    for v in ["A", "B", "C"]:
        f = build_formula(v, _ctx())
        assert "lq" not in f.lower()
        assert "kde" not in f.lower()


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_formula("D", _ctx())


def test_response_lhs():
    f = build_formula("A", _ctx())
    assert f.startswith("I(1 - fcc23) + trial ~")
