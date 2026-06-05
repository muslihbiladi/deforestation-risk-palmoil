import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import pickle
import tempfile


class _FakeConfig:
    project_future = False
    projection_year = 2030
    forest_years = [2010, 2020, 2023]
    risk_classes = 5


class _FakeCtx:
    def __init__(self, tmp_path):
        self.data_dir = Path(tmp_path) / "data"
        self.output_dir = Path(tmp_path) / "output"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = _FakeConfig()


class _StubModel:
    betas_names = []
    betas = []
    DIC = 400.0


def _make_model_pkl(tmp_path):
    mod = _StubModel()
    pkl_path = tmp_path / "mod_A.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(mod, fh)
    return pkl_path, mod


def test_predict_risk_calls_far_and_returns_path(tmp_path):
    from palmdef_risk.model.predict import predict_risk

    ctx = _FakeCtx(tmp_path)
    pkl_path, mod = _make_model_pkl(tmp_path)

    with patch("forestatrisk.predict.predict_raster") as mock_pred:
        result = predict_risk(ctx, pkl_path, "A")

    mock_pred.assert_called_once()
    assert result == ctx.output_dir / "predictions" / "risk_A.tif"
    assert (ctx.output_dir / "predictions").is_dir()


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
