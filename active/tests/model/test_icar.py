import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from palmdef_risk.model.icar import build_formula


def _fake_cellneigh(**kwargs):
    return np.array([2, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32)


def test_fit_all_fits_every_variant_in_parallel(tmp_path, minimal_config_yaml, monkeypatch):
    """fit_all dispatches one worker per variant via run_parallel."""
    from palmdef_risk.io.run import create_run
    from palmdef_risk.model import icar

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")  # variants A, B
    monkeypatch.setattr("forestatrisk.cellneigh", _fake_cellneigh, raising=False)
    monkeypatch.setattr("palmdef_risk.parallel.adaptive_workers", lambda *a, **k: 1)

    import palmdef_risk.parallel as parallel_mod
    spy = MagicMock(side_effect=parallel_mod.run_parallel)
    monkeypatch.setattr(icar, "run_parallel", spy, raising=False)

    calls = []

    def fake_fit_model(variant, c, nneigh=None, adj=None):
        assert nneigh is not None and adj is not None  # shared adjacency threaded through
        calls.append(variant)
        p = c.output_dir / "models" / f"model_{variant}" / f"mod_{variant}.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(icar, "fit_model", fake_fit_model)
    paths = icar.fit_all(ctx)

    spy.assert_called_once()                 # dispatched through run_parallel
    assert sorted(calls) == ["A", "B"]
    assert len(paths) == 2
    assert all(isinstance(p, Path) for p in paths)


def test_fit_all_skips_variants_with_existing_pkl(tmp_path, minimal_config_yaml, monkeypatch):
    """A pre-existing mod_<v>.pkl must skip the fit (resumability)."""
    from palmdef_risk.io.run import create_run
    from palmdef_risk.model import icar

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")  # A, B
    monkeypatch.setattr("forestatrisk.cellneigh", _fake_cellneigh, raising=False)
    monkeypatch.setattr("palmdef_risk.parallel.adaptive_workers", lambda *a, **k: 1)

    pre = ctx.output_dir / "models" / "model_A" / "mod_A.pkl"
    pre.parent.mkdir(parents=True, exist_ok=True)
    pre.write_bytes(b"existing")

    calls = []

    def fake_fit_model(variant, c, nneigh=None, adj=None):
        calls.append(variant)
        p = c.output_dir / "models" / f"model_{variant}" / f"mod_{variant}.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(icar, "fit_model", fake_fit_model)
    paths = icar.fit_all(ctx)

    assert calls == ["B"]            # A skipped
    assert len(paths) == 2           # both pkl paths returned


def test_fit_all_propagates_worker_failure(tmp_path, minimal_config_yaml, monkeypatch):
    """A variant's fit failure must surface (fail-fast, as the sequential loop did)."""
    from palmdef_risk.io.run import create_run
    from palmdef_risk.model import icar

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")  # A, B
    monkeypatch.setattr("forestatrisk.cellneigh", _fake_cellneigh, raising=False)
    monkeypatch.setattr("palmdef_risk.parallel.adaptive_workers", lambda *a, **k: 1)

    def fake_fit_model(variant, c, nneigh=None, adj=None):
        if variant == "B":
            raise RuntimeError("mcmc boom")
        p = c.output_dir / "models" / f"model_{variant}" / f"mod_{variant}.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(icar, "fit_model", fake_fit_model)
    with pytest.raises(RuntimeError, match="mcmc boom"):
        icar.fit_all(ctx)


def test_fit_worker_is_picklable():
    """ProcessPoolExecutor pickles the worker by reference — it must be a
    module-level function and its task tuple must be picklable."""
    import pickle
    from palmdef_risk.model.icar import _fit_one_variant
    assert pickle.loads(pickle.dumps(_fit_one_variant)) is _fit_one_variant
    pickle.dumps(("A", "runs/x", "nneigh.npy", "adj.npy"))


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
