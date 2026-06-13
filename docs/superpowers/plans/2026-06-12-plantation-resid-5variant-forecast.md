# Plantation Residual + 5 Variants + t3 Forecast — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an orthogonalized `plantation_resid` covariate, expand model variants A/B/C → A/B/C/D/E, and wire true t3 forecast re-prediction (mill gravity, GHSL town, plantation) so risk can be projected from the t3-state landscape.

**Architecture:** A new `process/plantation.py` mirrors `process/gravity.py`'s OLS-residual orthogonalization. Variant definitions become a single source of truth in `model/icar.py` (`variant_extra_cols`), eliminating triplicated `variant in (...)` logic in icar/diagnostics/reports. Forecast prediction assembles a clean `data/forecast/` var_dir (t3 dynamics + copied static t2 rasters) and runs `predict_raster_binomial_iCAR` against `forest_t3.tif`, reusing the time-agnostic model object.

**Tech Stack:** Python, GDAL/osgeo, NumPy, pandas, patsy, forestatrisk, pytest. Tests run offline against synthetic UTM-50S (EPSG:32750) fixtures in `active/tests/conftest.py`.

**Spec:** `docs/superpowers/specs/2026-06-12-plantation-resid-5variant-forecast-design.md`

**Conventions:**
- Run all commands with **CWD = repo root** (pytest `testpaths` = `active/tests`).
- Use `python -m pytest`, not bare `pytest` (Windows PATH shadowing — see CLAUDE.md).
- Float rasters: NoData = `-9999.0`, dtype Float32. Never process in EPSG:4326.
- Commit after each task. Each commit must be green.

---

## File Map

| File | Change |
|---|---|
| `active/palmdef_risk/process/plantation.py` | **Create** — `orthogonalize_plantation`, `compute_plantation_resid`, `compute_plantation_resid_forecast` |
| `active/palmdef_risk/model/icar.py` | 5-variant table, `variant_extra_cols`, use it in `_build_and_fit`, error text |
| `active/palmdef_risk/model/diagnostics.py` | use `variant_extra_cols` in `compute_residuals_all` |
| `active/palmdef_risk/model/reports.py` | use `variant_extra_cols` in `_predict_in_sample` |
| `active/palmdef_risk/model/predict.py` | `build_forecast_vardir`, `predict_forecast`, wire into `predict_all` |
| `active/palmdef_risk/process/distances.py` | rename forecast outputs to `forecast/dist_*.tif`; forecast `dist_plantation_edge` |
| `active/palmdef_risk/io/config.py` | `VALID_VARIANTS`, default variants, validate text |
| `active/configs/{template,central-kalimantan,east-kotawaringin}.yaml` | `variants: [A, B, C, D, E]` |
| `active/notebooks/02_process.ipynb` | call plantation resid (model + forecast) |
| `active/notebooks/03_model.ipynb` | VIF list adds `plantation_resid` (guarded) |
| `active/tests/process/test_plantation.py` | **Create** |
| `active/tests/process/test_distances.py` | update forecast expectations |
| `active/tests/model/test_icar.py` | D/E formulas, `variant_extra_cols`, fix `test_formula_c_*` |
| `active/tests/model/test_predict.py` | `build_forecast_vardir`, `predict_forecast` |
| `active/tests/io/test_config.py` | default variants, validate rejects unknown |
| `.claude/CLAUDE.md`, `docs/WORKFLOW.md`, `README.md`, memory | doc/invariant updates |

---

## Phase 1 — `plantation_resid` covariate + 5 variants (model period)

### Task 1: `process/plantation.py` — orthogonalized residual

**Files:**
- Create: `active/palmdef_risk/process/plantation.py`
- Test: `active/tests/process/test_plantation.py`

- [ ] **Step 1: Write the failing test**

Create `active/tests/process/test_plantation.py`:

```python
# tests/process/test_plantation.py
import numpy as np
from osgeo import gdal


def _f32(write_raster, tmp_path, name, arr, gt):
    return write_raster(tmp_path / name, arr.astype(np.float32), gt, 32750,
                        dtype=gdal.GDT_Float32, nodata=-9999.0)


def test_orthogonalize_plantation_writes_residual(tmp_path, write_raster):
    from palmdef_risk.process.plantation import orthogonalize_plantation
    rng = np.random.default_rng(0)
    gt = [500000, 100, 0, 9005000, 0, -100]
    plant = rng.uniform(1, 5000, (20, 20))
    edge = rng.uniform(1, 5000, (20, 20))
    defor = rng.uniform(1, 5000, (20, 20))
    road = rng.uniform(1, 5000, (20, 20))
    out = tmp_path / "plantation_resid.tif"
    r2 = orthogonalize_plantation(
        _f32(write_raster, tmp_path, "dist_plantation_edge.tif", plant, gt),
        _f32(write_raster, tmp_path, "dist_edge.tif", edge, gt),
        _f32(write_raster, tmp_path, "dist_defor.tif", defor, gt),
        _f32(write_raster, tmp_path, "dist_road.tif", road, gt),
        out,
    )
    assert out.exists()
    assert 0.0 <= r2 <= 1.0
    ds = gdal.Open(str(out))
    resid = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    valid = resid[resid != -9999.0]
    # OLS residual is mean-zero by construction
    assert abs(valid.mean()) < 1e-3


def test_orthogonalize_plantation_respects_nodata(tmp_path, write_raster):
    from palmdef_risk.process.plantation import orthogonalize_plantation
    gt = [500000, 100, 0, 9001000, 0, -100]
    plant = np.full((10, 10), 100.0)
    plant[0, 0] = -9999.0  # nodata pixel must stay nodata in output
    edge = np.full((10, 10), 50.0)
    defor = np.full((10, 10), 75.0)
    road = np.full((10, 10), 25.0)
    out = tmp_path / "plantation_resid.tif"
    orthogonalize_plantation(
        _f32(write_raster, tmp_path, "dist_plantation_edge.tif", plant, gt),
        _f32(write_raster, tmp_path, "dist_edge.tif", edge, gt),
        _f32(write_raster, tmp_path, "dist_defor.tif", defor, gt),
        _f32(write_raster, tmp_path, "dist_road.tif", road, gt),
        out,
    )
    ds = gdal.Open(str(out))
    resid = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    assert resid[0, 0] == -9999.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest active/tests/process/test_plantation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'palmdef_risk.process.plantation'`

- [ ] **Step 3: Write minimal implementation**

Create `active/palmdef_risk/process/plantation.py`:

```python
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from osgeo import gdal

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)


def _load_flat(p):
    ds = gdal.Open(str(p))
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    shape = arr.shape
    ds = None
    return arr, nd, gt, proj, shape


def orthogonalize_plantation(
    dist_plant_path: Path | str,
    dist_edge_path: Path | str,
    dist_defor_path: Path | str,
    dist_road_path: Path | str,
    out_path: Path | str,
) -> float:
    """OLS: log(dist_plant+1) ~ log(dist_edge+1)+log(dist_defor+1)+log(dist_road+1).

    Writes the residual to out_path (Float32, NoData -9999). Returns R².
    The residual is the `plantation_resid` covariate (already in log space — it must
    NOT be re-logged downstream).
    """
    p_arr, p_nd, gt, proj, shape = _load_flat(dist_plant_path)
    e_arr, e_nd, *_ = _load_flat(dist_edge_path)
    f_arr, f_nd, *_ = _load_flat(dist_defor_path)
    r_arr, r_nd, *_ = _load_flat(dist_road_path)

    mask = (
        (p_arr != p_nd) & (e_arr != e_nd) & (f_arr != f_nd) & (r_arr != r_nd)
    )
    y = np.log(p_arr[mask] + 1.0)
    Xe = np.log(e_arr[mask] + 1.0)
    Xf = np.log(f_arr[mask] + 1.0)
    Xr = np.log(r_arr[mask] + 1.0)

    X = np.column_stack([np.ones(len(y)), Xe, Xf, Xr])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual_flat = y - X @ beta

    ss_res = np.sum(residual_flat ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r2 > 0.85:
        logger.warning(
            "Plantation R²=%.3f > 0.85: plantation proximity largely collinear "
            "with edge/defor/road.", r2,
        )

    ny, nx = shape
    resid_arr = np.full(shape, -9999.0, dtype=np.float32)
    resid_arr[mask] = residual_flat.astype(np.float32)

    out_ds = gdal.GetDriverByName("GTiff").Create(
        str(out_path), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(resid_arr)
    out_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    out_ds = None
    logger.info("Plantation orthogonalized R²=%.3f; residual → %s", r2, out_path)
    return r2


def compute_plantation_resid(ctx: "RunContext", force: bool = False) -> float:
    """Compute model-period plantation_resid → data/plantation_resid.tif.

    Returns R²; 0.0 (skipped) when dist_plantation_edge.tif is absent.
    """
    d = ctx.data_dir
    dist_plant = d / "dist_plantation_edge.tif"
    out_resid = d / "plantation_resid.tif"
    if not dist_plant.exists():
        logger.warning("dist_plantation_edge.tif absent — skipping plantation_resid")
        return 0.0
    if out_resid.exists() and not force:
        logger.info("plantation_resid.tif exists — skipping")
        return 0.0
    return orthogonalize_plantation(
        dist_plant, d / "dist_edge.tif", d / "dist_defor.tif",
        d / "dist_road.tif", out_resid,
    )


def compute_plantation_resid_forecast(ctx: "RunContext", force: bool = False) -> float:
    """Compute forecast (t3) plantation_resid → data/forecast/plantation_resid.tif.

    Orthogonalizes against t3 dist_edge/dist_defor (data/forecast/) and the static
    t2 dist_road (data/dist_road.tif). Returns R²; 0.0 when t3 plantation absent.
    """
    d = ctx.data_dir
    fcast = d / "forecast"
    fcast.mkdir(parents=True, exist_ok=True)
    dist_plant = fcast / "dist_plantation_edge.tif"
    out_resid = fcast / "plantation_resid.tif"
    if not dist_plant.exists():
        logger.warning("forecast/dist_plantation_edge.tif absent — skipping forecast plantation_resid")
        return 0.0
    if out_resid.exists() and not force:
        logger.info("forecast/plantation_resid.tif exists — skipping")
        return 0.0
    return orthogonalize_plantation(
        dist_plant, fcast / "dist_edge.tif", fcast / "dist_defor.tif",
        d / "dist_road.tif", out_resid,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest active/tests/process/test_plantation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add active/palmdef_risk/process/plantation.py active/tests/process/test_plantation.py
git commit -m "feat(process): plantation_resid orthogonalized covariate"
```

---

### Task 2: `model/icar.py` — 5-variant table + `variant_extra_cols`

**Files:**
- Modify: `active/palmdef_risk/model/icar.py:17-49` (variant table, build_formula error)
- Test: `active/tests/model/test_icar.py`

- [ ] **Step 1: Update the tests (these are the new spec)**

In `active/tests/model/test_icar.py`, replace `test_formula_c_adds_hgu_spline`,
`test_formula_no_dist_mill_in_any_variant`, `test_formula_no_lq_terms`, and
`test_unknown_variant_raises` with the block below, and add the new tests:

```python
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
```

Also add `plantation_resid` to the `_ALL_COLS` list near the top so `_sample_df`
provides the column:

```python
_ALL_COLS = [
    "altitude", "slope",
    "log_dist_defor", "log_dist_edge", "log_dist_road", "log_dist_town", "log_dist_river",
    "gravity_resid", "plantation_resid", "hgu_b1", "hgu_b2",
]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest active/tests/model/test_icar.py -v`
Expected: FAIL — `build_formula("C")` still emits gravity/hgu; `variant_extra_cols` missing.

- [ ] **Step 3: Implement the 5-variant table + helper**

In `active/palmdef_risk/model/icar.py`, replace lines 17-22 (the
`_VARIANT_SCALED_COLS` block) with:

```python
_BASE_SCALED_COLS = ["altitude", "slope"] + [f"log_{c}" for c in _LOG_DIST_COLS]

# Covariates each variant adds beyond the biophysical base. Single source of truth
# for both the formula RHS and the NaN-drop subset used in fit/residuals/predict.
_VARIANT_EXTRA_COLS: dict[str, list[str]] = {
    "A": [],
    "B": ["gravity_resid"],
    "C": ["plantation_resid"],
    "D": ["gravity_resid", "plantation_resid"],
    "E": ["gravity_resid", "plantation_resid", "hgu_b1", "hgu_b2"],
}

# Full scaled-covariate list per variant (order determines column order in X).
_VARIANT_SCALED_COLS: dict[str, list[str]] = {
    v: _BASE_SCALED_COLS + extra for v, extra in _VARIANT_EXTRA_COLS.items()
}


def variant_extra_cols(variant: str) -> list[str]:
    """Covariates a variant adds beyond the biophysical base.

    Single source of truth for the NaN-drop subset consumed by _build_and_fit,
    diagnostics.compute_residuals_all, and reports._predict_in_sample.
    """
    if variant not in _VARIANT_EXTRA_COLS:
        raise ValueError(
            f"Unknown variant: {variant!r}. Valid variants: A, B, C, D, E"
        )
    return list(_VARIANT_EXTRA_COLS[variant])
```

Then in `build_formula` (now ~line 34), update the error message:

```python
    all_scaled = _VARIANT_SCALED_COLS.get(variant)
    if all_scaled is None:
        raise ValueError(f"Unknown variant: {variant!r}. Valid variants: A, B, C, D, E")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest active/tests/model/test_icar.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add active/palmdef_risk/model/icar.py active/tests/model/test_icar.py
git commit -m "feat(model): 5-variant table A-E with variant_extra_cols single source of truth"
```

---

### Task 3: `model/icar.py` — use `variant_extra_cols` in `_build_and_fit`

**Files:**
- Modify: `active/palmdef_risk/model/icar.py:103-106`

- [ ] **Step 1: Replace the hardcoded extra_cols**

In `_build_and_fit`, replace:

```python
    extra_cols = (
        (["gravity_resid"] if variant in ("B", "C") else [])
        + (["hgu_b1", "hgu_b2"] if variant == "C" else [])
    )
```

with:

```python
    extra_cols = variant_extra_cols(variant)
```

(`variant_extra_cols` is defined in the same module — no import needed.)

- [ ] **Step 2: Run tests to verify nothing broke**

Run: `python -m pytest active/tests/model/ -v`
Expected: PASS (existing icar/predict/diagnostics tests still green)

- [ ] **Step 3: Commit**

```bash
git add active/palmdef_risk/model/icar.py
git commit -m "refactor(model): _build_and_fit uses variant_extra_cols"
```

---

### Task 4: `diagnostics.py` + `reports.py` — use `variant_extra_cols`

**Files:**
- Modify: `active/palmdef_risk/model/diagnostics.py:102, 125-129`
- Modify: `active/palmdef_risk/model/reports.py:32, 39-43`

- [ ] **Step 1: Update `diagnostics.compute_residuals_all`**

In `active/palmdef_risk/model/diagnostics.py`, change the import line (currently
`from palmdef_risk.model.icar import prepare_sample, _LOG_DIST_COLS`) to:

```python
    from palmdef_risk.model.icar import prepare_sample, _LOG_DIST_COLS, variant_extra_cols
```

Replace:

```python
        extra_cols = (
            (["gravity_resid"] if variant in ("B", "C") else [])
            + (["hgu_b1", "hgu_b2"] if variant == "C" else [])
        )
```

with:

```python
        extra_cols = variant_extra_cols(variant)
```

- [ ] **Step 2: Update `reports._predict_in_sample`**

In `active/palmdef_risk/model/reports.py`, change the import line (currently
`from palmdef_risk.model.icar import prepare_sample, _LOG_DIST_COLS`) to:

```python
    from palmdef_risk.model.icar import prepare_sample, _LOG_DIST_COLS, variant_extra_cols
```

Replace:

```python
    extra_cols = (
        (["gravity_resid"] if variant in ("B", "C") else [])
        + (["hgu_b1", "hgu_b2"] if variant == "C" else [])
    )
```

with:

```python
    extra_cols = variant_extra_cols(variant)
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest active/tests/model/test_diagnostics.py active/tests/model/test_predict.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add active/palmdef_risk/model/diagnostics.py active/palmdef_risk/model/reports.py
git commit -m "refactor(model): diagnostics/reports use variant_extra_cols"
```

---

### Task 5: `io/config.py` — VALID_VARIANTS, default, validate text

**Files:**
- Modify: `active/palmdef_risk/io/config.py:9, 116, 155`
- Test: `active/tests/io/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `active/tests/io/test_config.py`:

```python
def test_default_variants_are_a_to_e(minimal_config_yaml):
    import yaml
    from palmdef_risk.io.config import RunConfig
    raw = yaml.safe_load(minimal_config_yaml.read_text())
    del raw["model"]["variants"]  # force the default
    minimal_config_yaml.write_text(yaml.dump(raw))
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert cfg.model_variants == ["A", "B", "C", "D", "E"]


def test_validate_accepts_d_and_e(minimal_config_yaml):
    import yaml
    from palmdef_risk.io.config import RunConfig
    raw = yaml.safe_load(minimal_config_yaml.read_text())
    raw["model"]["variants"] = ["A", "D", "E"]
    minimal_config_yaml.write_text(yaml.dump(raw))
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert "model.variants" not in " ".join(cfg.validate())


def test_validate_rejects_unknown_variant(minimal_config_yaml):
    import yaml
    from palmdef_risk.io.config import RunConfig
    raw = yaml.safe_load(minimal_config_yaml.read_text())
    raw["model"]["variants"] = ["A", "Z"]
    minimal_config_yaml.write_text(yaml.dump(raw))
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert any("unknown variant" in e.lower() for e in cfg.validate())
```

Confirm `from_yaml` is the actual constructor name — if the codebase uses a
different entry point (e.g. `RunConfig.load`), match it. Check with
`grep -n "def from_yaml\|def load\|classmethod" active/palmdef_risk/io/config.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest active/tests/io/test_config.py -k "variants or variant" -v`
Expected: FAIL — default is `["A","B","C"]`, validate rejects D/E.

- [ ] **Step 3: Implement**

In `active/palmdef_risk/io/config.py`:

Line 9 — `VALID_VARIANTS = {"A", "B", "C"}` → `VALID_VARIANTS = {"A", "B", "C", "D", "E"}`

Line ~116 — `model_variants=list(mod.get("variants", ["A", "B", "C"])),` →
`model_variants=list(mod.get("variants", ["A", "B", "C", "D", "E"])),`

Line ~155 — `errors.append(f"model.variants: unknown variant '{v}' (valid: A, B, C)")` →
`errors.append(f"model.variants: unknown variant '{v}' (valid: A, B, C, D, E)")`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest active/tests/io/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add active/palmdef_risk/io/config.py active/tests/io/test_config.py
git commit -m "feat(config): variants A-E, default to all five"
```

---

### Task 6: Config YAMLs — `variants: [A, B, C, D, E]`

**Files:**
- Modify: `active/configs/template.yaml:71`
- Modify: `active/configs/central-kalimantan.yaml:70`
- Modify: `active/configs/east-kotawaringin.yaml:70`

- [ ] **Step 1: Edit all three**

In each file replace the variants line with:

```yaml
  variants: [A, B, C, D, E]   # A=biophysical, B=+gravity, C=+plantation, D=+both, E=+both+HGU spline
```

- [ ] **Step 2: Validate configs parse**

Run: `cd active && python run.py --config configs/central-kalimantan.yaml --dry-run && cd ..`
Expected: dry-run prints `variants : ['A', 'B', 'C', 'D', 'E']`, no validation errors.

- [ ] **Step 3: Commit**

```bash
git add active/configs/template.yaml active/configs/central-kalimantan.yaml active/configs/east-kotawaringin.yaml
git commit -m "config: default run configs to variants A-E"
```

---

### Task 7: Notebooks — plantation resid call + VIF entry

**Files:**
- Modify: `active/notebooks/02_process.ipynb` (after the gravity cell)
- Modify: `active/notebooks/03_model.ipynb` (VIF cell)

- [ ] **Step 1: 02_process — add a plantation residual cell**

After the gravity cell (the one importing `compute_gravity_accessibility`), add a new
code cell:

```python
from palmdef_risk.process.plantation import (
    compute_plantation_resid, compute_plantation_resid_forecast,
)
r2_plant = compute_plantation_resid(ctx)
print(f"plantation_resid R² (model period) = {r2_plant:.3f}")
r2_plant_fc = compute_plantation_resid_forecast(ctx)
print(f"plantation_resid R² (forecast)     = {r2_plant_fc:.3f}")
```

(The forecast call is harmless now and feeds Phase 2; it skips silently when t3
plantation is absent.)

- [ ] **Step 2: 03_model — add plantation_resid to VIF (guarded)**

Replace the VIF cell body with:

```python
from palmdef_risk.model.diagnostics import compute_vif
import pandas as pd
_sample_cols = set(pd.read_csv(ctx.output_dir / "sample.csv", nrows=1).columns)
covariates = ["altitude", "slope", "dist_defor", "dist_edge", "dist_road",
              "dist_town", "dist_river", "gravity_resid"]
if "plantation_resid" in _sample_cols:
    covariates.append("plantation_resid")
compute_vif([c for c in covariates if c in _sample_cols],
            ctx.output_dir / "sample.csv",
            ctx.output_dir / "diagnostics" / "vif.json")
```

- [ ] **Step 3: Sanity check (notebooks aren't unit-tested; verify JSON validity)**

Run: `python -c "import json,sys; json.load(open(r'active/notebooks/02_process.ipynb')); json.load(open(r'active/notebooks/03_model.ipynb')); print('notebooks parse OK')"`
Expected: `notebooks parse OK`

- [ ] **Step 4: Commit**

```bash
git add active/notebooks/02_process.ipynb active/notebooks/03_model.ipynb
git commit -m "feat(notebooks): compute plantation_resid + add to VIF"
```

---

## Phase 2 — t3 forecast re-prediction

### Task 8: `distances.py` — rename forecast outputs to model names

**Files:**
- Modify: `active/palmdef_risk/process/distances.py:174-203`
- Test: `active/tests/process/test_distances.py:37-44`

- [ ] **Step 1: Update the distances test to expect new forecast names**

In `active/tests/process/test_distances.py`, update the expected list and add forecast
assertions:

```python
    expected = [
        "dist_edge.tif", "dist_defor.tif",
        "dist_road.tif", "dist_river.tif", "dist_town.tif",
        "dist_plantation_edge.tif",
    ]
    for name in expected:
        assert (d / name).exists(), f"Missing: {name}"
    # Forecast distances now live under data/forecast/ with model names
    for name in ["dist_edge.tif", "dist_defor.tif", "dist_plantation_edge.tif"]:
        assert (d / "forecast" / name).exists(), f"Missing forecast: {name}"
```

The `_make_ctx` helper already writes `plantation.tif`; add a `plantation_t3` raster so
the forecast plantation distance is exercised. In `_make_ctx`, after the
`write_raster(d / "plantation.tif", ...)` line add:

```python
    write_raster(d / "plantation_t3.tif", arr, gt, 32750, nodata=255)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest active/tests/process/test_distances.py -v`
Expected: FAIL — `data/forecast/dist_edge.tif` not produced (still `dist_edge_forecast.tif`).

- [ ] **Step 3: Implement the rename + forecast plantation distance**

In `active/palmdef_risk/process/distances.py`:

In the modelling-distances block, after the `dist_plantation_edge` block (~line 178),
the model-period plantation source is `plantation.tif`. Keep it. Then in the
forecast block (~line 182-189), replace:

```python
    for name, src, tgt in [
        ("dist_edge_forecast",  "forest_t3.tif", 0),
        ("dist_defor_forecast", "fcc23.tif",     0),
    ]:
        src_path = d / src
        out_path = d / f"{name}.tif"
        if src_path.exists() and _needs_recompute(out_path, ref_shape):
            tasks.append(("raster", src_path, out_path, tgt))
```

with:

```python
    for name, src, tgt in [
        ("dist_edge",  "forest_t3.tif", 0),
        ("dist_defor", "fcc23.tif",     0),
    ]:
        src_path = d / src
        out_path = fcast / f"{name}.tif"
        if src_path.exists() and _needs_recompute(out_path, ref_shape):
            tasks.append(("raster", src_path, out_path, tgt))

    # Forecast plantation edge distance (t3 plantation raster), model name under forecast/
    plant_t3 = d / "plantation_t3.tif"
    dist_plant_fc = fcast / "dist_plantation_edge.tif"
    if plant_t3.exists() and _needs_recompute(dist_plant_fc, ref_shape):
        tasks.append(("raster", plant_t3, dist_plant_fc, 1))
```

Update the module docstring's "Forecast distances (data/forecast/)" list to read
`dist_edge, dist_defor, dist_town, dist_plantation_edge`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest active/tests/process/test_distances.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add active/palmdef_risk/process/distances.py active/tests/process/test_distances.py
git commit -m "feat(process): forecast distances use model names under data/forecast/"
```

---

### Task 9: `process/plantation.py` — forecast residual test

**Files:**
- Test: `active/tests/process/test_plantation.py`

(The `compute_plantation_resid_forecast` implementation already landed in Task 1; this
task adds its dedicated test.)

- [ ] **Step 1: Add the forecast test**

Append to `active/tests/process/test_plantation.py`:

```python
def test_compute_plantation_resid_forecast(tmp_path, write_raster, write_vector,
                                           minimal_config_yaml):
    import numpy as np
    from osgeo import gdal
    from palmdef_risk.io.run import create_run
    from palmdef_risk.process.plantation import compute_plantation_resid_forecast

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    d = ctx.data_dir
    fcast = d / "forecast"
    fcast.mkdir(parents=True, exist_ok=True)
    gt = [500000, 100, 0, 9001000, 0, -100]
    rng = np.random.default_rng(1)
    for path in [fcast / "dist_plantation_edge.tif", fcast / "dist_edge.tif",
                 fcast / "dist_defor.tif", d / "dist_road.tif"]:
        write_raster(path, rng.uniform(1, 5000, (10, 10)).astype(np.float32),
                     gt, 32750, dtype=gdal.GDT_Float32, nodata=-9999.0)

    r2 = compute_plantation_resid_forecast(ctx)
    assert (fcast / "plantation_resid.tif").exists()
    assert 0.0 <= r2 <= 1.0


def test_compute_plantation_resid_skips_when_absent(tmp_path, write_vector,
                                                    minimal_config_yaml):
    from palmdef_risk.io.run import create_run
    from palmdef_risk.process.plantation import compute_plantation_resid
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    # No dist_plantation_edge.tif in data_dir → must skip, return 0.0
    assert compute_plantation_resid(ctx) == 0.0
    assert not (ctx.data_dir / "plantation_resid.tif").exists()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest active/tests/process/test_plantation.py -v`
Expected: PASS (4 tests total)

- [ ] **Step 3: Commit**

```bash
git add active/tests/process/test_plantation.py
git commit -m "test(process): forecast plantation_resid + skip-when-absent"
```

---

### Task 10: `predict.py` — `build_forecast_vardir`

**Files:**
- Modify: `active/palmdef_risk/model/predict.py` (add function near top of module body)
- Test: `active/tests/model/test_predict.py`

- [ ] **Step 1: Write the failing test**

Add to `active/tests/model/test_predict.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest active/tests/model/test_predict.py::test_build_forecast_vardir_copies_statics -v`
Expected: FAIL — `build_forecast_vardir` not defined.

- [ ] **Step 3: Implement**

In `active/palmdef_risk/model/predict.py`, add after the imports / before
`_create_log_dist_rasters`:

```python
# Static covariates reused from the t2 grid for forecast prediction (no t3 source).
_FORECAST_STATIC_RASTERS = (
    "altitude.tif", "slope.tif", "dist_road.tif", "dist_river.tif",
    "protected.tif", "hgu_signed_dist.tif",
)


def build_forecast_vardir(ctx: RunContext) -> Path:
    """Assemble a clean data/forecast/ holding the full model-named covariate set.

    Copies static t2 rasters in; the t3 dynamics (dist_edge, dist_defor, dist_town,
    gravity_resid, plantation_resid) are produced upstream in Stage 2 and already
    live under data/forecast/. Returns the forecast directory path.
    """
    import shutil
    d = ctx.data_dir
    fcast = d / "forecast"
    fcast.mkdir(parents=True, exist_ok=True)
    for name in _FORECAST_STATIC_RASTERS:
        src = d / name
        if src.exists():
            shutil.copy2(src, fcast / name)
        else:
            logger.warning(
                "Static covariate %s missing — forecast var_dir may be incomplete", name
            )
    return fcast
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest active/tests/model/test_predict.py::test_build_forecast_vardir_copies_statics -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add active/palmdef_risk/model/predict.py active/tests/model/test_predict.py
git commit -m "feat(predict): build_forecast_vardir assembles t3 var_dir"
```

---

### Task 11: `predict.py` — `predict_forecast` + wire into `predict_all`

**Files:**
- Modify: `active/palmdef_risk/model/predict.py` (add `predict_forecast`; edit `predict_all`)
- Test: `active/tests/model/test_predict.py`

- [ ] **Step 1: Write the failing test**

`predict_forecast` calls forestatrisk, which is heavy; test the **guard path** (missing
forecast covariates → returns None without raising), which needs no forestatrisk run.
Add to `active/tests/model/test_predict.py`:

```python
def test_predict_forecast_skips_when_covariates_missing(tmp_path, write_raster,
                                                        minimal_config_yaml):
    import pickle
    import numpy as np
    from osgeo import gdal
    from palmdef_risk.io.run import create_run
    from palmdef_risk.model.predict import predict_forecast

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    # Minimal sample.csv so DesignInfo can rebuild
    _make_sample_csv(ctx.output_dir)
    model_dir = ctx.output_dir / "models" / "model_A"
    model_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "betas": np.zeros(1), "rho": np.zeros(4),
        "formula": "I(1 - fcc23) + trial ~ scale(altitude) + protected + cell",
        "variant": "A",
    }
    with open(model_dir / "mod_A.pkl", "wb") as f:
        pickle.dump(state, f)
    # rho.tif present but forecast var_dir empty → guard returns None
    write_raster(model_dir / "rho.tif", np.ones((4, 4), dtype=np.float32),
                 [500000, 30, 0, 9000120, 0, -30], 32750,
                 dtype=gdal.GDT_Float32, nodata=-9999.0)
    (ctx.data_dir / "forecast").mkdir(parents=True, exist_ok=True)
    result = predict_forecast(ctx, model_dir / "mod_A.pkl", "A")
    assert result is None
```

Confirm `_make_sample_csv` exists in `test_predict.py` (it does, per the file) and
includes an `altitude` column; if the helper's columns don't cover the test formula,
extend the formula string to match what the helper provides.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest active/tests/model/test_predict.py::test_predict_forecast_skips_when_covariates_missing -v`
Expected: FAIL — `predict_forecast` not defined.

- [ ] **Step 3: Implement `predict_forecast`**

In `active/palmdef_risk/model/predict.py`, add after `predict_risk`:

```python
def predict_forecast(ctx: RunContext, model_path: Path, variant: str) -> Optional[Path]:
    """Predict t3 forecast risk for a fitted variant from data/forecast/ covariates.

    Reuses the model's interpolated rho.tif (spatial effect is location-based and
    time-agnostic). Returns risk_<variant>_forecast.tif, or None when the forecast
    var_dir lacks a required covariate raster.
    """
    import forestatrisk as far
    from patsy import dmatrices
    from palmdef_risk.model.icar import prepare_sample

    with open(model_path, "rb") as fh:
        state = pickle.load(fh)

    fcast = ctx.data_dir / "forecast"
    rho_path = model_path.parent / "rho.tif"
    if not rho_path.exists():
        logger.warning(
            "rho.tif missing for variant %s — run predict_risk before predict_forecast",
            variant,
        )
        return None

    # Rebuild DesignInfo from sample.csv (never pickle DesignInfo).
    sample_path = ctx.output_dir / "sample.csv"
    data = pd.read_csv(sample_path)
    data = prepare_sample(data)
    scaled_cols = re.findall(r"scale\((\w+)\)", state["formula"])
    if scaled_cols:
        data = data.dropna(subset=scaled_cols)
    y, x = dmatrices(state["formula"], data, return_type="matrix")

    pred_mod = far.icarModelPred(
        formula=state["formula"],
        _y_design_info=y.design_info,
        _x_design_info=x.design_info,
        betas=state["betas"],
        rho=state["rho"],
    )

    # Build derived rasters INTO the forecast dir (from t3 dist_* + copied statics).
    _create_log_dist_rasters(fcast, state["formula"])
    _create_hgu_spline_rasters(fcast, state["formula"], sample_path)

    # Guard: every scaled covariate + protected must exist in the forecast var_dir.
    bare_covs = {"protected"}
    needed = {v for v in set(scaled_cols) | bare_covs if v != "cell"}
    missing = [v for v in needed if not (fcast / f"{v}.tif").exists()]
    if missing:
        logger.warning(
            "Forecast prediction for variant %s skipped — missing forecast rasters %s",
            variant, sorted(missing),
        )
        return None

    out_dir = ctx.output_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_path = out_dir / f"risk_{variant}_forecast.tif"
    forest_t3 = ctx.data_dir / "forest_t3.tif"
    if not forest_t3.exists():
        logger.warning("forest_t3.tif missing — cannot predict forecast for %s", variant)
        return None

    far.predict_raster_binomial_iCAR(
        pred_mod,
        var_dir=str(fcast),
        input_cell_raster=str(rho_path),
        input_forest_raster=str(forest_t3),
        output_file=str(risk_path),
    )
    logger.info("Forecast risk raster written: %s", risk_path)
    return risk_path
```

- [ ] **Step 4: Wire into `predict_all`**

In `predict_all`, add `build_forecast_vardir(ctx)` before the variant loop, and after
the existing `project_future` block (inside the per-variant `try`), add a forecast call.
The loop body becomes:

```python
    build_forecast_vardir(ctx)
    for variant in tqdm(variants, desc="Predicting risk", unit="variant"):
        model_path = ctx.output_dir / "models" / f"model_{variant}" / f"mod_{variant}.pkl"
        if not model_path.exists():
            logger.warning("Model pkl not found, skipping variant %s: %s", variant, model_path)
            continue
        risk_path = ctx.output_dir / "predictions" / f"risk_{variant}.tif"
        if risk_path.exists():
            logger.info("risk_%s.tif exists — skipping prediction", variant)
            results.append(risk_path)
        else:
            try:
                risk_path = predict_risk(ctx, model_path, variant)
                results.append(risk_path)
            except Exception:
                import traceback
                logger.error("Prediction failed for variant %s:\n%s", variant, traceback.format_exc())
                continue
        try:
            future_path = project_future(ctx, risk_path, variant)
            if future_path is not None:
                results.append(future_path)
        except Exception:
            import traceback
            logger.error("Future projection failed for variant %s:\n%s", variant, traceback.format_exc())
        # t3 forecast risk (decision #8: does NOT yet feed project_future)
        fc_path = ctx.output_dir / "predictions" / f"risk_{variant}_forecast.tif"
        if fc_path.exists():
            logger.info("risk_%s_forecast.tif exists — skipping forecast", variant)
            results.append(fc_path)
        else:
            try:
                fc = predict_forecast(ctx, model_path, variant)
                if fc is not None:
                    results.append(fc)
            except Exception:
                import traceback
                logger.error("Forecast prediction failed for variant %s:\n%s", variant, traceback.format_exc())
    return results
```

(This preserves the existing skip-if-done guards and the t2 `project_future` behaviour
unchanged, per decision #8.)

- [ ] **Step 5: Run tests**

Run: `python -m pytest active/tests/model/test_predict.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add active/palmdef_risk/model/predict.py active/tests/model/test_predict.py
git commit -m "feat(predict): predict_forecast t3 risk wired into predict_all"
```

---

### Task 12: Notebook 02 — assemble forecast var_dir

**Files:**
- Modify: `active/notebooks/02_process.ipynb`

`build_forecast_vardir` is also called inside `predict_all`, but calling it at the end of
Stage 2 makes the forecast dir inspectable before modelling.

- [ ] **Step 1: Add a cell at the end of 02_process**

```python
from palmdef_risk.model.predict import build_forecast_vardir
fcast_dir = build_forecast_vardir(ctx)
print(f"forecast var_dir assembled: {fcast_dir}")
```

- [ ] **Step 2: Notebook parses**

Run: `python -c "import json; json.load(open(r'active/notebooks/02_process.ipynb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add active/notebooks/02_process.ipynb
git commit -m "feat(notebook): assemble forecast var_dir in 02_process"
```

---

## Phase 3 — Docs, invariants, full-suite verification

### Task 13: Update docs & invariants

**Files:**
- Modify: `.claude/CLAUDE.md`
- Modify: `docs/WORKFLOW.md`
- Modify: `README.md`
- Modify: memory (`MEMORY.md` + `feedback_naming_conventions.md` or a new project memory)

- [ ] **Step 1: CLAUDE.md — iCAR model rules**

Replace the line `- No model variants D–G exist in this codebase.` with:

```markdown
- Models: A (biophysical), B (+gravity_resid), C (+plantation_resid),
  D (+gravity_resid +plantation_resid), E (+both +HGU spline).
- No variants beyond A–E exist.
```

Replace `- `dist_plantation_edge` is computed but NOT entered into any model formula.`
with:

```markdown
- `dist_plantation_edge` is NOT entered directly; plantation proximity is the
  orthogonalized residual `plantation_resid` (variants C/D/E). Already log-space —
  never re-log it.
```

In the "Conventions that bite if ignored" list, change
`**Only model variants A, B, C exist.** No D–G.` to
`**Only model variants A–E exist.**`.

- [ ] **Step 2: WORKFLOW.md — variant table**

Find the model-variants section and replace the A/B/C description with the A–E table
from the spec (§3). Add `risk_<v>_forecast.tif` and `data/forecast/plantation_resid.tif`
to the Stage-2/Stage-3 output descriptions.

- [ ] **Step 3: README.md — variants + outputs**

Update any `variants: [A, B, C]` reference and the output-layout section to mention the
five variants and the `risk_<v>_forecast.tif` forecast rasters.

- [ ] **Step 4: Memory**

Append to `MEMORY.md`:

```markdown
- [Model variants A–E](project_variants_scheme.md) — 5-variant relabel + plantation_resid covariate
```

Create `…/memory/project_variants_scheme.md`:

```markdown
---
name: project_variants_scheme
description: 5-variant model scheme (A–E) and plantation_resid covariate naming
metadata:
  type: project
---

Model variants relabeled from A/B/C to **A–E** (spec
`docs/superpowers/specs/2026-06-12-plantation-resid-5variant-forecast-design.md`):
A=biophysical, B=+gravity_resid, C=+plantation_resid, D=+both, E=+both+HGU spline.
Old C (gravity+HGU) no longer exists. `plantation_resid` is the orthogonalized
log-residual of dist_plantation_edge against dist_edge/defor/road — raster + sample
column both named `plantation_resid`; it is already log-space (never re-log).
t3 forecast re-prediction is wired (`risk_<v>_forecast.tif`); forecast **validation**
is a separate deferred spec. See [[feedback_naming_conventions]].
```

- [ ] **Step 5: Commit**

```bash
git add -f .claude/CLAUDE.md docs/WORKFLOW.md README.md
git add "C:/Users/musli/.claude/projects/g--My-Drive-JOB-WRI-GDRIVE-RS-Deforestation-Deforestation-Risk-Technical-Parts-deforestation-risk-palmoil-v2-0/memory/MEMORY.md" "C:/Users/musli/.claude/projects/g--My-Drive-JOB-WRI-GDRIVE-RS-Deforestation-Deforestation-Risk-Technical-Parts-deforestation-risk-palmoil-v2-0/memory/project_variants_scheme.md"
git commit -m "docs: variants A-E, plantation_resid, forecast prediction"
```

(The memory files live outside the repo; if `git add` rejects them, just write them with
the Write tool and skip git — they are not version-controlled with the project.)

---

### Task 14: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `python -m pytest`
Expected: all green. Investigate any failure before proceeding — do not commit through red.

- [ ] **Step 2: Targeted re-run of the touched areas**

Run: `python -m pytest active/tests/process/test_plantation.py active/tests/process/test_distances.py active/tests/model/test_icar.py active/tests/model/test_predict.py active/tests/model/test_diagnostics.py active/tests/io/test_config.py -v`
Expected: PASS

- [ ] **Step 3: Config dry-run smoke test**

Run: `cd active && python run.py --config configs/central-kalimantan.yaml --dry-run && cd ..`
Expected: variants `['A','B','C','D','E']`, no validation errors.

- [ ] **Step 4: Final commit (if any verification fixups were needed)**

```bash
git add -A
git commit -m "test: full suite green for plantation_resid + 5 variants + forecast"
```

---

## Self-Review Notes (carried from spec)

- **Decision #8 (open):** `project_future`/`deforest()` still ranks on the t2
  `risk_<v>.tif`. Forecast risk `risk_<v>_forecast.tif` is produced but not yet feeding
  allocation. Flip is a one-line change in `predict_all`'s `project_future(ctx, risk_path, ...)`
  call (pass the forecast path) when the user decides.
- **Forecast validation** is out of scope (Spec B).
- **HGU spline at t3:** `hgu_signed_dist` is a static covariate copied into the forecast
  dir, so `_create_hgu_spline_rasters` rebuilds `hgu_b1/b2` there from the training
  sample's knots — identical basis to the t2 model. Correct by construction.
```