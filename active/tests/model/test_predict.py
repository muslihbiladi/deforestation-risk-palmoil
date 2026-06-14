import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import pickle
from osgeo import gdal


_PREDICT_FORMULA = "I(1 - fcc23) + trial ~ scale(altitude) + protected + cell"


def _setup_run_with_models(tmp_path, minimal_config_yaml, variants=("A", "B")):
    """A materialized run with sample.csv + a fitted-state pkl per variant."""
    from palmdef_risk.io.run import create_run
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    _make_sample_csv(ctx.output_dir)
    for v in variants:
        md = ctx.output_dir / "models" / f"model_{v}"
        md.mkdir(parents=True, exist_ok=True)
        state = {"formula": _PREDICT_FORMULA, "betas": np.zeros(3),
                 "rho": np.zeros(10), "variant": v}
        with open(md / f"mod_{v}.pkl", "wb") as f:
            pickle.dump(state, f)
    return ctx


def test_predict_all_predicts_every_variant_in_parallel(tmp_path, minimal_config_yaml,
                                                        monkeypatch):
    """predict_all dispatches one worker per variant via run_parallel."""
    from palmdef_risk.model import predict
    import palmdef_risk.parallel as parallel_mod

    ctx = _setup_run_with_models(tmp_path, minimal_config_yaml)  # A, B
    monkeypatch.setattr("palmdef_risk.parallel.adaptive_workers", lambda *a, **k: 1)
    spy = MagicMock(side_effect=parallel_mod.run_parallel)
    monkeypatch.setattr(predict, "run_parallel", spy, raising=False)

    calls = []

    def fake_predict_risk(c, model_path, variant):
        calls.append(variant)
        rp = c.output_dir / "predictions" / f"risk_{variant}.tif"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_bytes(b"x")
        return rp

    monkeypatch.setattr(predict, "predict_risk", fake_predict_risk)
    monkeypatch.setattr(predict, "project_future", lambda *a, **k: None)
    monkeypatch.setattr(predict, "predict_forecast", lambda *a, **k: None)

    paths = predict.predict_all(ctx)

    spy.assert_called_once()
    assert sorted(calls) == ["A", "B"]
    assert any("risk_A.tif" in str(p) for p in paths)
    assert any("risk_B.tif" in str(p) for p in paths)


def test_predict_all_skips_existing_risk(tmp_path, minimal_config_yaml, monkeypatch):
    """A pre-existing risk_<v>.tif must not be re-predicted."""
    from palmdef_risk.model import predict

    ctx = _setup_run_with_models(tmp_path, minimal_config_yaml)  # A, B
    monkeypatch.setattr("palmdef_risk.parallel.adaptive_workers", lambda *a, **k: 1)
    pre = ctx.output_dir / "predictions" / "risk_A.tif"
    pre.parent.mkdir(parents=True, exist_ok=True)
    pre.write_bytes(b"existing")

    calls = []

    def fake_predict_risk(c, model_path, variant):
        calls.append(variant)
        rp = c.output_dir / "predictions" / f"risk_{variant}.tif"
        rp.write_bytes(b"x")
        return rp

    monkeypatch.setattr(predict, "predict_risk", fake_predict_risk)
    monkeypatch.setattr(predict, "project_future", lambda *a, **k: None)
    monkeypatch.setattr(predict, "predict_forecast", lambda *a, **k: None)

    predict.predict_all(ctx)
    assert calls == ["B"]   # A skipped


def test_predict_all_isolates_one_variant_failure(tmp_path, minimal_config_yaml,
                                                  monkeypatch):
    """One variant's prediction failure is logged but the others proceed (no raise)."""
    from palmdef_risk.model import predict

    ctx = _setup_run_with_models(tmp_path, minimal_config_yaml)  # A, B
    monkeypatch.setattr("palmdef_risk.parallel.adaptive_workers", lambda *a, **k: 1)

    def fake_predict_risk(c, model_path, variant):
        if variant == "A":
            raise RuntimeError("predict boom A")
        rp = c.output_dir / "predictions" / f"risk_{variant}.tif"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_bytes(b"x")
        return rp

    monkeypatch.setattr(predict, "predict_risk", fake_predict_risk)
    monkeypatch.setattr(predict, "project_future", lambda *a, **k: None)
    monkeypatch.setattr(predict, "predict_forecast", lambda *a, **k: None)

    paths = predict.predict_all(ctx)   # must NOT raise
    assert any("risk_B.tif" in str(p) for p in paths)
    assert not any("risk_A.tif" in str(p) for p in paths)


def test_predict_all_computes_forest_areas_once(tmp_path, minimal_config_yaml, monkeypatch):
    """forest_t2/t3 areas are variant-invariant: countpix must run twice total
    (one per forest), not twice per variant."""
    from palmdef_risk.model import predict

    ctx = _setup_run_with_models(tmp_path, minimal_config_yaml)  # A, B
    ctx.config.project_future = True
    (ctx.data_dir / "forest_t2.tif").write_bytes(b"x")
    (ctx.data_dir / "forest_t3.tif").write_bytes(b"x")
    monkeypatch.setattr("palmdef_risk.parallel.adaptive_workers", lambda *a, **k: 1)

    countpix_calls = []

    def fake_countpix(input_raster, value):
        countpix_calls.append(input_raster)
        return {"area": 1000.0}

    def fake_predict_risk(c, model_path, variant):
        rp = c.output_dir / "predictions" / f"risk_{variant}.tif"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_bytes(b"x")
        return rp

    monkeypatch.setattr("forestatrisk.countpix", fake_countpix, raising=False)
    monkeypatch.setattr("forestatrisk.deforest",
                        lambda **k: {"threshold": 1, "error_perc": 0.0}, raising=False)
    monkeypatch.setattr(predict, "predict_risk", fake_predict_risk)
    monkeypatch.setattr(predict, "predict_forecast", lambda *a, **k: None)

    predict.predict_all(ctx)

    # 2 variants would have meant 4 countpix calls in the per-variant version.
    assert len(countpix_calls) == 2


def test_predict_worker_is_picklable():
    """ProcessPoolExecutor pickles the worker by reference — it must be a
    module-level function and its task tuple must be picklable."""
    import pickle as _pickle
    from palmdef_risk.model.predict import _predict_one_variant
    assert _pickle.loads(_pickle.dumps(_predict_one_variant)) is _predict_one_variant
    _pickle.dumps(("A", "runs/x", 1000.0, 900.0))


class _FakeConfig:
    project_future = False
    projection_year = 2030
    forest_years = [2010, 2020, 2023]
    risk_classes = 5
    csize = 10


class _FakeCtx:
    def __init__(self, tmp_path):
        self.data_dir = Path(tmp_path) / "data"
        self.output_dir = Path(tmp_path) / "output"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = _FakeConfig()


def _make_sample_csv(output_dir, n=60):
    """Minimal sample.csv with the columns prepare_sample + the test formula need."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "fcc23": rng.integers(0, 2, n),
        "altitude": rng.normal(100, 10, n),
        "slope": rng.normal(5, 1, n),
        "dist_defor": rng.uniform(1, 1000, n),
        "dist_edge": rng.uniform(1, 1000, n),
        "dist_road": rng.uniform(1, 1000, n),
        "dist_town": rng.uniform(1, 1000, n),
        "dist_river": rng.uniform(1, 1000, n),
        "protected": rng.integers(0, 2, n),
        "cell": rng.integers(0, 10, n),
    })
    df.to_csv(output_dir / "sample.csv", index=False)


def test_predict_risk_calls_far_and_returns_path(tmp_path):
    from palmdef_risk.model.predict import predict_risk

    ctx = _FakeCtx(tmp_path)
    _make_sample_csv(ctx.output_dir)
    # predict_risk validates that each covariate raster exists on disk (existence
    # check only — empty files satisfy it).
    (ctx.data_dir / "altitude.tif").write_bytes(b"")
    (ctx.data_dir / "protected.tif").write_bytes(b"")

    formula = "I(1 - fcc23) + trial ~ scale(altitude) + protected + cell"
    state = {"formula": formula, "betas": np.zeros(3), "rho": np.zeros(10)}
    pkl_path = tmp_path / "mod_A.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(state, fh)

    with patch("forestatrisk.icarModelPred") as m_pred, \
         patch("forestatrisk.interpolate_rho") as m_interp, \
         patch("forestatrisk.predict_raster_binomial_iCAR") as m_raster:
        result = predict_risk(ctx, pkl_path, "A")

    m_pred.assert_called_once()
    m_interp.assert_called_once()
    m_raster.assert_called_once()
    assert result == ctx.output_dir / "predictions" / "risk_A.tif"
    assert (ctx.output_dir / "predictions").is_dir()


def test_predict_risk_skips_interpolate_rho_when_rho_exists(tmp_path):
    """Resumability: a pre-existing rho.tif must not be recomputed."""
    from palmdef_risk.model.predict import predict_risk

    ctx = _FakeCtx(tmp_path)
    _make_sample_csv(ctx.output_dir)
    (ctx.data_dir / "altitude.tif").write_bytes(b"")
    (ctx.data_dir / "protected.tif").write_bytes(b"")

    formula = "I(1 - fcc23) + trial ~ scale(altitude) + protected + cell"
    state = {"formula": formula, "betas": np.zeros(3), "rho": np.zeros(10)}
    pkl_path = tmp_path / "mod_A.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(state, fh)
    # model_dir == pkl_path.parent == tmp_path; pre-create rho.tif.
    (tmp_path / "rho.tif").write_bytes(b"")

    with patch("forestatrisk.icarModelPred"), \
         patch("forestatrisk.interpolate_rho") as m_interp, \
         patch("forestatrisk.predict_raster_binomial_iCAR"):
        predict_risk(ctx, pkl_path, "A")

    m_interp.assert_not_called()


def test_project_future_skipped_when_disabled(tmp_path):
    from palmdef_risk.model.predict import project_future

    ctx = _FakeCtx(tmp_path)
    ctx.config.project_future = False
    risk_path = tmp_path / "risk_A.tif"

    result = project_future(ctx, risk_path, "A")
    assert result is None


def test_project_future_skipped_when_years_nonpositive(tmp_path):
    from palmdef_risk.model.predict import project_future

    ctx = _FakeCtx(tmp_path)
    ctx.config.project_future = True
    ctx.config.projection_year = 2020  # same as forest_years[-1]=2023 → n_years<0
    ctx.config.forest_years = [2010, 2020, 2023]
    risk_path = tmp_path / "risk_A.tif"

    result = project_future(ctx, risk_path, "A")
    assert result is None


def test_project_future_calls_far_deforest(tmp_path):
    from palmdef_risk.model.predict import project_future

    ctx = _FakeCtx(tmp_path)
    ctx.config.project_future = True
    ctx.config.projection_year = 2030
    ctx.config.forest_years = [2010, 2020, 2023]  # span t2→t3 = 3 yr, projection 7 yr
    (ctx.data_dir / "forest_t2.tif").write_bytes(b"")
    (ctx.data_dir / "forest_t3.tif").write_bytes(b"")
    risk_path = tmp_path / "risk_A.tif"

    # Historical loss 30,000 ha over 3 yr → 10,000 ha/yr → 70,000 ha for 7-yr projection.
    countpix_side_effect = [{"area": 100_000.0}, {"area": 70_000.0}]
    with patch("forestatrisk.countpix", side_effect=countpix_side_effect), \
         patch("forestatrisk.deforest", return_value={"threshold": 32000, "error_perc": 0.1}) as mock_def:
        result = project_future(ctx, risk_path, "A")

    mock_def.assert_called_once()
    call_kwargs = mock_def.call_args.kwargs
    assert call_kwargs["input_raster"] == str(risk_path)
    assert call_kwargs["hectares"] == pytest.approx(70_000.0)
    assert call_kwargs["output_file"] == str(
        ctx.output_dir / "predictions" / "forest_future_A.tif"
    )
    assert result == ctx.output_dir / "predictions" / "forest_future_A.tif"


def test_classify_risk_correct_zones():
    from palmdef_risk.model.predict import classify_risk

    arr = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    thresholds = [0.2, 0.4, 0.6, 0.8]
    zones = classify_risk(arr, thresholds)
    assert list(zones) == [1, 2, 3, 4, 5]


def test_classify_risk_all_below_first_threshold():
    from palmdef_risk.model.predict import classify_risk

    arr = np.array([0.01, 0.05, 0.1])
    zones = classify_risk(arr, [0.2, 0.4, 0.6, 0.8])
    assert list(zones) == [1, 1, 1]


def test_risk_raster_is_uint16_nodata_zero(tmp_path, write_raster):
    """predict_risk must write UInt16 with NoData=0 (0=NoData, 1-65535=prob)."""
    from palmdef_risk.model.predict import _write_risk_raster
    import numpy as np
    from osgeo import gdal
    arr = np.random.uniform(0, 1, (10, 10)).astype(np.float32)
    ref = write_raster(tmp_path / "ref.tif", np.ones((10, 10), dtype=np.uint8),
                       gt=[500000, 30, 0, 9000300, 0, -30], epsg=32750)
    out = tmp_path / "risk.tif"
    _write_risk_raster(arr, str(ref), str(out))
    ds = gdal.Open(str(out))
    band = ds.GetRasterBand(1)
    assert band.DataType == gdal.GDT_UInt16
    assert band.GetNoDataValue() == 0
    result = band.ReadAsArray()
    assert result[result > 0].min() >= 1
    ds = None


def test_build_forecast_vardir_copies_statics(tmp_path, write_raster,
                                              minimal_config_yaml):
    import numpy as np
    from osgeo import gdal
    from palmdef_risk.io.run import create_run
    from palmdef_risk.model.predict import build_forecast_vardir

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    d = ctx.data_dir
    d.mkdir(parents=True, exist_ok=True)
    gt = [500000, 30, 0, 9000300, 0, -30]
    arr = np.ones((10, 10), dtype=np.float32)
    for name in ["altitude.tif", "slope.tif", "dist_road.tif", "dist_river.tif",
                 "protected.tif", "hgu_signed_dist.tif"]:
        write_raster(d / name, arr, gt, 32750, dtype=gdal.GDT_Float32, nodata=-9999.0)

    fcast = build_forecast_vardir(ctx)
    for name in ["altitude.tif", "slope.tif", "dist_road.tif", "dist_river.tif",
                 "protected.tif", "hgu_signed_dist.tif"]:
        assert (fcast / name).exists(), f"static not copied: {name}"


def _write_tiled(path, arr, dtype, nodata, block=16):
    """Small-block TILED GeoTIFF so derived-raster windowing hits partial blocks."""
    from osgeo import gdal, osr
    drv = gdal.GetDriverByName("GTiff")
    ny, nx = arr.shape
    ds = drv.Create(
        str(path), nx, ny, 1, dtype,
        options=["TILED=YES", f"BLOCKXSIZE={block}", f"BLOCKYSIZE={block}"],
    )
    ds.SetGeoTransform([500000, 30, 0, 9000000, 0, -30])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32750)
    ds.SetProjection(srs.ExportToWkt())
    b = ds.GetRasterBand(1)
    b.WriteArray(arr)
    b.SetNoDataValue(nodata)
    ds.FlushCache()
    ds = None
    return path


def _read_arr(path):
    from osgeo import gdal
    ds = gdal.Open(str(path))
    arr = ds.GetRasterBand(1).ReadAsArray()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    return arr, nd


def test_create_log_dist_rasters_windowed_matches_full(tmp_path):
    """Windowed log_dist raster is bit-identical to the full-array log(x+1) path."""
    from palmdef_risk.model.predict import _create_log_dist_rasters

    src = (np.arange(70 * 70).reshape(70, 70).astype(np.float32) + 1.0) * 3.0
    src[0, :] = -9999.0          # nodata edge
    src[33:38, 10:25] = -9999.0  # nodata block spanning tile edges
    _write_tiled(tmp_path / "dist_road.tif", src, gdal.GDT_Float32, -9999.0)

    formula = "I(1 - fcc23) + trial ~ scale(log_dist_road) + cell"
    _create_log_dist_rasters(tmp_path, formula)

    got, nd = _read_arr(tmp_path / "log_dist_road.tif")

    expected = np.full((70, 70), -9999.0, dtype=np.float32)
    valid = src != -9999.0
    expected[valid] = np.log(src[valid] + 1)
    assert np.array_equal(got, expected)
    assert nd == -9999.0


def test_create_hgu_spline_rasters_windowed_matches_full(tmp_path):
    """Windowed cr() spline basis rasters are bit-identical to whole-raster eval."""
    from patsy import dmatrix, build_design_matrices
    from palmdef_risk.model.predict import _create_hgu_spline_rasters

    # sample.csv drives the memorized cr() knots.
    rng = np.random.default_rng(7)
    hgu_train = rng.uniform(-8000, 8000, 400)
    pd.DataFrame({"hgu_signed_dist": hgu_train}).to_csv(tmp_path / "sample.csv", index=False)

    src = np.linspace(-8000, 8000, 70 * 70).reshape(70, 70).astype(np.float32)
    src[1, :] = -9999.0
    src[40:44, 5:30] = -9999.0
    _write_tiled(tmp_path / "hgu_signed_dist.tif", src, gdal.GDT_Float32, -9999.0)

    formula = "I(1 - fcc23) + trial ~ hgu_b1 + hgu_b2 + cell"
    _create_hgu_spline_rasters(tmp_path, formula, tmp_path / "sample.csv")

    # Baseline: the pre-refactor whole-raster evaluation.
    design_info = dmatrix(
        "cr(x, knots=(-5000, 0, 5000)) - 1", {"x": hgu_train}, return_type="matrix"
    ).design_info
    n_basis = len(design_info.column_names)
    arr = src.astype(np.float64)
    valid = arr != -9999.0
    basis = np.asarray(build_design_matrices([design_info], {"x": arr[valid].ravel()})[0])

    for i, name in enumerate(("hgu_b1", "hgu_b2")):
        col_idx = min(i, n_basis - 1)
        expected = np.full((70, 70), -9999.0, dtype=np.float32)
        expected[valid] = basis[:, col_idx].astype(np.float32)
        got, nd = _read_arr(tmp_path / f"{name}.tif")
        assert np.array_equal(got, expected), f"{name} mismatch"
        assert nd == -9999.0


def test_predict_forecast_skips_when_covariates_missing(tmp_path, write_raster,
                                                        minimal_config_yaml):
    import pickle
    import numpy as np
    import pandas as pd
    from osgeo import gdal
    from palmdef_risk.io.run import create_run
    from palmdef_risk.model.predict import predict_forecast

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    # Minimal sample.csv with every column prepare_sample + the test formula need.
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 20
    pd.DataFrame({
        "fcc23": rng.integers(0, 2, n),
        "altitude": rng.uniform(0, 100, n),
        "slope": rng.uniform(0, 30, n),
        "protected": rng.integers(0, 2, n),
        "cell": rng.integers(0, 4, n),
        "dist_defor": rng.uniform(1, 5000, n),
        "dist_edge": rng.uniform(1, 5000, n),
        "dist_road": rng.uniform(1, 5000, n),
        "dist_town": rng.uniform(1, 5000, n),
        "dist_river": rng.uniform(1, 5000, n),
        "X": rng.uniform(500000, 501000, n),
        "Y": rng.uniform(9000000, 9001000, n),
    }).to_csv(ctx.output_dir / "sample.csv", index=False)
    model_dir = ctx.output_dir / "models" / "model_A"
    model_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "betas": np.zeros(1), "rho": np.zeros(4),
        "formula": "I(1 - fcc23) + trial ~ scale(altitude) + protected + cell",
        "variant": "A",
    }
    with open(model_dir / "mod_A.pkl", "wb") as f:
        pickle.dump(state, f)
    write_raster(model_dir / "rho.tif", np.ones((4, 4), dtype=np.float32),
                 [500000, 30, 0, 9000120, 0, -30], 32750,
                 dtype=gdal.GDT_Float32, nodata=-9999.0)
    (ctx.data_dir / "forecast").mkdir(parents=True, exist_ok=True)
    # forecast var_dir lacks altitude.tif/protected.tif → guard returns None
    result = predict_forecast(ctx, model_dir / "mod_A.pkl", "A")
    assert result is None
