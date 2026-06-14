import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch


class _FakeConfig:
    csize = 10
    model_variants = ["A", "B"]
    risk_thresholds = None


class _FakeCtx:
    def __init__(self, tmp_path):
        self.output_dir = Path(tmp_path) / "output"
        self.data_dir = Path(tmp_path) / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = _FakeConfig()


def _make_sample_csv(output_dir, n=80):
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "fcc23": rng.integers(0, 2, n),
        "altitude": rng.normal(100, 10, n),
        "slope": rng.normal(5, 1, n),
        "dist_defor": rng.uniform(1, 1000, n),
        "dist_edge": rng.uniform(1, 1000, n),
        "dist_road": rng.uniform(1, 1000, n),
        "dist_town": rng.uniform(1, 1000, n),
        "dist_river": rng.uniform(1, 1000, n),
        "gravity_resid": rng.normal(0, 1, n),
        "protected": rng.integers(0, 2, n),
        "cell": rng.integers(0, 10, n),
    }).to_csv(output_dir / "sample.csv", index=False)


def test_predict_in_sample_caches_per_variant(tmp_path):
    """The CSV read + design-matrix build must run once per variant, even though
    three exporters each call _predict_in_sample (3 variants x 3 = 9x/run)."""
    from palmdef_risk.model import reports

    ctx = _FakeCtx(tmp_path)
    _make_sample_csv(ctx.output_dir)
    state_a = {
        "formula": "I(1 - fcc23) + trial ~ scale(altitude) + protected + cell",
        "betas": np.zeros(3), "rho": np.zeros(20),
    }
    state_b = {
        "formula": (
            "I(1 - fcc23) + trial ~ scale(altitude) + scale(gravity_resid) "
            "+ protected + cell"
        ),
        "betas": np.zeros(4), "rho": np.zeros(20),
    }

    with patch("palmdef_risk.model.reports.pd.read_csv", wraps=pd.read_csv) as m_read:
        reports._predict_in_sample(ctx, state_a, "A")
        reports._predict_in_sample(ctx, state_a, "A")  # same variant → cached
        reports._predict_in_sample(ctx, state_a, "A")  # still cached
        reports._predict_in_sample(ctx, state_b, "B")  # new variant → fresh build

    # Once for A, once for B — not 4 times.
    assert m_read.call_count == 2


def test_predict_in_sample_returns_consistent_result(tmp_path):
    """Cached result must equal the freshly computed one."""
    from palmdef_risk.model import reports

    ctx = _FakeCtx(tmp_path)
    _make_sample_csv(ctx.output_dir)
    state = {
        "formula": "I(1 - fcc23) + trial ~ scale(altitude) + protected + cell",
        "betas": np.zeros(3), "rho": np.zeros(20),
    }
    p1, y1, names1 = reports._predict_in_sample(ctx, state, "A")
    p2, y2, names2 = reports._predict_in_sample(ctx, state, "A")
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(y1, y2)
    assert names1 == names2
