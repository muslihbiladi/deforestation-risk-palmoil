import json
import numpy as np
import pandas as pd
from unittest.mock import patch


def test_sensitivity_json_has_entry_per_sigma(tmp_path, minimal_config_yaml):
    from palmdef_risk.io.run import create_run
    from palmdef_risk.model.sensitivity import run_gravity_sensitivity

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")

    # Sensitivity now reads sample.csv up front to re-sample gravity_resid per sigma.
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "X": [0.0, 1.0, 2.0],
        "Y": [0.0, 1.0, 2.0],
        "gravity_resid": [0.0, 0.0, 0.0],
    }).to_csv(ctx.output_dir / "sample.csv", index=False)

    sigmas = ctx.config.sensitivity_sigmas
    # Return a different beta per call so the test sees variation across sigmas.
    call_counter = {"n": 0}

    def _fake_ortho(ctx_arg, force=False):
        return 0.0

    def _fake_resample(grav_resid_path, df_xy):
        return np.zeros(len(df_xy), dtype=np.float64)

    def _fake_build_and_fit(variant, ctx_arg, data):
        call_counter["n"] += 1
        return {
            "betas": np.array([0.1, 0.2, 0.3 + 0.01 * call_counter["n"]]),
            "formula": "I(1 - fcc23) + trial ~ scale(altitude) + gravity_resid + cell",
            "deviance": np.array([100.0 + call_counter["n"]]),
        }

    with patch("palmdef_risk.model.sensitivity._rasterize_mills_density",
               return_value=tmp_path / "density.tif") as m_rast, \
         patch("palmdef_risk.model.sensitivity._apply_gaussian_filter") as m_filter, \
         patch("palmdef_risk.model.sensitivity.orthogonalize_gravity_ctx", _fake_ortho), \
         patch("palmdef_risk.model.sensitivity._resample_gravity_resid", _fake_resample), \
         patch("palmdef_risk.model.sensitivity._build_and_fit", _fake_build_and_fit):
        out_json = run_gravity_sensitivity(ctx)

    # Mill density bitmap is sigma-invariant: rasterize ONCE, vary only the
    # Gaussian kernel per sigma.
    assert m_rast.call_count == 1
    assert m_filter.call_count == len(sigmas)

    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert len(data) == len(sigmas)
    for entry in data:
        assert "sigma_km" in entry
        assert "accessibility_coef" in entry
        assert "mean_deviance" in entry
    # Bug regression: with per-call beta variation, coefficients must differ across sigmas.
    coefs = [e["accessibility_coef"] for e in data]
    assert len(set(coefs)) == len(sigmas), f"Expected distinct coefs per sigma, got {coefs}"
