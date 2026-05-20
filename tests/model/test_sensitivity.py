import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_sensitivity_json_has_entry_per_sigma(tmp_path, minimal_config_yaml):
    from palmdef_risk.io.run import create_run
    from palmdef_risk.model.sensitivity import run_gravity_sensitivity

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")

    def _fake_compute(ctx_arg, sigma_km):
        return tmp_path / "gravity_raw.tif"

    def _fake_ortho(ctx_arg):
        return tmp_path / "gravity_resid.tif"

    def _fake_fit(variant, ctx_arg):
        return tmp_path / "mod.pkl"

    def _fake_load(pkl):
        m = MagicMock()
        m.betas = [0.1, 0.2, 0.3]
        m.deviance = [100.0]
        return m

    with patch("palmdef_risk.model.sensitivity.compute_gravity_raw", _fake_compute):
        with patch("palmdef_risk.model.sensitivity.orthogonalize_gravity_ctx", _fake_ortho):
            with patch("palmdef_risk.model.sensitivity.fit_model", _fake_fit):
                with patch("palmdef_risk.model.sensitivity._load_model", _fake_load):
                    out = ctx.output_dir / "diagnostics" / "gravity_sensitivity.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    result = run_gravity_sensitivity(ctx)

    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == len(ctx.config.sensitivity_sigmas)
    for entry in data:
        assert "sigma_km" in entry
        assert "accessibility_coef" in entry
