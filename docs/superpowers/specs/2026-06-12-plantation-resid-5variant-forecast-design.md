# Plantation Residual + 5 Variants + t3 Forecast Prediction — Design Spec

**Date:** 2026-06-12
**Status:** Approved (brainstorming)
**Topic:** (1) Add `plantation_resid` as an orthogonalized model covariate; (2) expand
the model from 3 variants (A/B/C) to 5 (A–E) on a full-factorial accessibility design;
(3) wire true t3 forecast re-prediction so risk can be projected from the t3-state
landscape (mills, towns, plantations, forest edge) instead of only the t2 state.
**Source note:** `notes/note_3.md`
**Sibling spec (separate):** Spec B — forecast **validation** methodology (deferred; not
designed here).

---

## 1. Problem

Three coupled gaps in the current model stage:

1. **Plantation proximity is computed but unused.** `dist_plantation_edge.tif` is
   produced in Stage 2 but never enters any model formula. Boundary encroachment
   (clearing at the margins of existing concessions) is a real deforestation driver
   that the model currently ignores.

2. **Only 3 variants, with a conflated C.** Today A = biophysical, B = +gravity,
   C = +gravity +HGU spline. There is no variant that isolates plantation proximity,
   and no full-factorial comparison of the two accessibility axes (mill gravity vs
   plantation edge).

3. **Forecast covariates are built but dead.** Stage 2 computes t3-state rasters
   (`compute_gravity_forecast` → `data/forecast/gravity_resid.tif`; GHSL-t3 →
   `data/forecast/dist_town.tif`; `dist_edge_forecast.tif`, `dist_defor_forecast.tif`),
   but no model code reads them (`grep forecast active/palmdef_risk/model/*.py` → 0
   matches). `predict_risk` reads only `ctx.data_dir` (t2 covariates); `project_future`
   merely thresholds the t2 risk map by the historical hectare rate. So future
   projection ranks pixels by a **stale t2-era landscape**.

forestatrisk fully supports t3 prediction — `predict_raster_binomial_iCAR` globs every
`*.tif` in `var_dir` and selects columns by formula-term name via
`build_design_matrices`. The model object (betas, design_info) is time-agnostic. The
only blockers are (a) the forecast rasters are mis-named / scattered, and (b) the
derived rasters (`log_dist_*`, `hgu_b1/b2`, and the new `plantation_resid`) need t3
versions. This spec removes both blockers.

---

## 2. Decisions (locked in brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Variant scheme | **Adopt note_3's 5-variant relabel** (A–E). Old C (gravity+HGU) ceases to exist. |
| 2 | Plantation accessibility shape | **Orthogonalized log-distance residual**, not a gravity surface, not raw distance (per note_3 rationale). |
| 3 | Raster + sample-column name | **`plantation_resid`** for both (file `plantation_resid.tif` → sample col `plantation_resid`). |
| 4 | Default variants | **`[A, B, C, D, E]`** in template + both run configs. |
| 5 | Forecast scope | **Wire t3 re-prediction** for mill gravity, GHSL town, and plantation. |
| 6 | t3 covariate split | **Time-varying t3**: `dist_edge`, `dist_defor`, `dist_town`, `gravity_resid`, `plantation_resid`. **Static reuse-t2**: `altitude`, `slope`, `dist_road`, `dist_river`, `protected`, `hgu`. |
| 7 | Forecast var_dir assembly | **Copy** static t2 rasters into a clean `data/forecast/` (cross-platform; one var_dir for the predict glob). |
| 8 | `deforest()` input (t2 vs t3 risk) | **OPEN — deferred.** Build forecast-risk production with a clean hook; leave `project_future` on t2 risk for now. User decides later. |
| 9 | Forecast validation | **Out of scope here** → Spec B. |

### Why orthogonalize plantation (from note_3)

`dist_plantation_edge` correlates with `dist_edge` (plantations adjoin forest),
`dist_defor` (plantations expand into cleared land), and `dist_road` (concessions
cluster along haul roads). Raw inclusion → collinearity → unstable MCMC. The residual

```
plantation_resid = log(dist_plantation_edge + 1)
    − OLS( log(dist_plantation_edge+1) ~ log(dist_edge+1) + log(dist_defor+1) + log(dist_road+1) )
```

is conditioned for the `beta_start=-99` logistic-MLE init. By Frisch–Waugh–Lovell the
fitted coefficient equals that of the raw term when placed alongside the same
regressors, so the orthogonalization is for **numerical conditioning**, not for
changing interpretation. `dist_town` is intentionally **not** an orthogonalization
regressor (weak plantation/town spatial link) and stays an independent covariate.

---

## 3. The 5-variant design

Base covariates (all variants): `altitude`, `slope`, `log_dist_defor`, `log_dist_edge`,
`log_dist_road`, `log_dist_town`, `log_dist_river`, `protected`, `cell` (iCAR random
effect).

| Variant | Extra beyond base | Meaning |
|---|---|---|
| A | — | biophysical baseline |
| B | `gravity_resid` | + mill accessibility |
| C | `plantation_resid` | + plantation proximity |
| D | `gravity_resid`, `plantation_resid` | + both accessibility axes |
| E | `gravity_resid`, `plantation_resid`, `hgu_b1`, `hgu_b2` | richest: + HGU concession spline |

**Data conditionality (unchanged behaviour):** `build_formula` already excludes
constant/missing columns. So `plantation_resid` silently drops from C/D/E when
`plantation.tif` is absent; `gravity_resid` drops from B/D/E when no mills; HGU drops
from E when no concessions. A degraded variant still fits on whatever remains.

---

## 4. Architecture

### 4.1 New module `process/plantation.py`

Mirrors `process/gravity.py`'s orthogonalization shape.

```
compute_plantation_resid(ctx, force=False) -> float        # → data/plantation_resid.tif, returns R²
compute_plantation_resid_forecast(ctx, force=False) -> float  # → data/forecast/plantation_resid.tif
```

- Reads `dist_plantation_edge.tif`, `dist_edge.tif`, `dist_defor.tif`, `dist_road.tif`.
- `log(x+1)` each; mask = intersection of all non-NoData; OLS; residual written
  Float32, NoData −9999. Returns R²; warn if R² > 0.85 (collinearity), same threshold
  as gravity.
- Skips silently (returns 0.0) if `dist_plantation_edge.tif` is absent.
- **Forecast variant** orthogonalizes against the **t3** `dist_edge`/`dist_defor`
  (from `data/forecast/`) and the **static** `dist_road` (`data/dist_road.tif`),
  consistent with decision #6. Reads `data/forecast/dist_plantation_edge.tif` (t3
  plantation; see §4.4) and writes `data/forecast/plantation_resid.tif`.

`dist_plantation_edge` itself stays computed in `process/distances.py` (already is).

### 4.2 Variant single source of truth in `model/icar.py`

- `_VARIANT_SCALED_COLS` becomes the 5-variant table of §3.
- **New helper** `variant_extra_cols(variant) -> list[str]`: returns the variant's
  covariates **beyond the biophysical base** (i.e. `_VARIANT_SCALED_COLS[variant]`
  minus the base list). This is the single definition of the NaN-drop `extra_cols`.
- `_build_and_fit` ([icar.py]), `compute_residuals_all` ([diagnostics.py]), and
  `_predict_in_sample` ([reports.py]) **replace their hardcoded** `variant in ("B","C")`
  / `== "C"` blocks with `variant_extra_cols(variant)`. Removes triplication and
  prevents drift as variants grow.
- `build_formula` error text updated from "Valid variants: A, B, C" → "A, B, C, D, E".

### 4.3 Sampling & in-sample prediction — minimal change

- `far.sample` globs all of `var_dir` → `plantation_resid.tif` auto-sampled as column
  `plantation_resid`. No sampling-code change.
- `plantation_resid` enters formulas as `scale(plantation_resid)`, **not** via
  `_LOG_DIST_COLS` (value is already a log-space residual; must not be re-logged).
- In-sample / hindcast prediction reads `plantation_resid.tif` directly from disk like
  `gravity_resid` — no derived-raster builder needed. The existing missing-raster guard
  in `predict.py` covers it.

### 4.4 t3 forecast re-prediction

**Naming normalization (process/distances.py):** forecast distance outputs move to
model names under `data/forecast/`:

| Current | New |
|---|---|
| `data/dist_edge_forecast.tif` | `data/forecast/dist_edge.tif` |
| `data/dist_defor_forecast.tif` | `data/forecast/dist_defor.tif` |
| `data/forecast/dist_town.tif` | unchanged (already correct) |
| `data/forecast/gravity_resid.tif` | unchanged (already correct) |

(No model code reads the old `*_forecast.tif` names — safe rename. Tests updated.)

`dist_plantation_edge` for t3: distances.py computes `data/forecast/dist_plantation_edge.tif`
from the t3 plantation raster (`plantation_t3`) when present, mirroring how the model-period
`dist_plantation_edge.tif` is built from `plantation.tif`.

**`build_forecast_vardir(ctx)`** (new; in `model/predict.py` or a small `process` helper):
assembles a clean `data/forecast/` holding the **complete model-named covariate set**:
- **Copy** static t2 rasters in: `altitude`, `slope`, `dist_road`, `dist_river`,
  `protected`, `hgu_signed_dist` (decision #7).
- t3 dynamics already present: `dist_edge`, `dist_defor`, `dist_town`, `gravity_resid`,
  `plantation_resid`.
- Ensures the dir contains **only** covariate rasters (no stray intermediates), and that
  every raster has a NoData value set (predict_raster `sys.exit`s otherwise).

**`predict_forecast(ctx, model_path, variant)`** (new; mirrors `predict_risk`):
- `var_dir = ctx.data_dir / "forecast"`, `input_forest_raster = forest_t3.tif`.
- Builds derived `log_dist_*` and `hgu_b1/b2` rasters **into the forecast dir** (see
  §4.5), from the forecast `dist_*` and the copied `hgu_signed_dist`.
- **Reuses the model's interpolated `rho.tif`** (spatial random effect is location-based
  and time-agnostic — no refit, no re-interpolation).
- Writes `predictions/risk_<variant>_forecast.tif`.

**`predict_all`** per variant, in order:
1. `predict_risk` → `risk_<v>.tif` (t2 hindcast / validation output — kept).
2. `predict_forecast` → `risk_<v>_forecast.tif` (t3 projection).
3. `project_future` → `forest_future_<v>.tif`. **For now unchanged: still ranks on the
   t2 `risk_<v>.tif`** (decision #8 open). A one-line switch will later point it at the
   forecast risk if the user chooses.

Forecast applies to **all 5 variants**; the per-variant formula selects its columns, so
extra t3 rasters in the dir are harmless.

### 4.5 Generalize derived-raster builders (`model/predict.py`)

`_create_log_dist_rasters(data_dir, formula)` and
`_create_hgu_spline_rasters(data_dir, formula, sample_path)` are generalized to take an
explicit **target directory** argument (defaulting to `ctx.data_dir` to preserve current
callers). `predict_forecast` passes `ctx.data_dir / "forecast"` so the t3 `log_dist_*`
and `hgu_b1/b2` rasters are built from the t3 `dist_*` and the copied static
`hgu_signed_dist`. The HGU spline basis is still rebuilt from the **training** `sample.csv`
knots (so the prediction basis matches the fitted betas), only the input `x` raster
differs (t3 `hgu_signed_dist` = copied static = same as t2; HGU is in the static set).

---

## 5. Config & validation (`io/config.py`)

- `VALID_VARIANTS = {"A", "B", "C", "D", "E"}`.
- Default `model_variants` → `["A", "B", "C", "D", "E"]`.
- `validate()` error text → "valid: A, B, C, D, E".
- `template.yaml`, `central-kalimantan.yaml`, `east-kotawaringin.yaml`:
  `variants: [A, B, C, D, E]` with corrected comment
  (`A=biophysical, B=+gravity, C=+plantation, D=+both, E=+both+HGU`).

---

## 6. Notebook wiring

- **`02_process.ipynb`**: after the gravity cell, add
  `compute_plantation_resid(ctx)` and `compute_plantation_resid_forecast(ctx)`; add a
  `build_forecast_vardir(ctx)` call so the forecast dir is assembled in Stage 2.
- **`03_model.ipynb`**: VIF covariate list gains `plantation_resid` (guarded — only if
  the column is present in `sample.csv`).

---

## 7. Docs & invariants

- **`.claude/CLAUDE.md`**: replace "Only model variants A, B, C exist. No D–G" with the
  A–E table; update "`dist_plantation_edge` is computed but NOT entered into any model
  formula" (now `plantation_resid` enters C/D/E); note forecast re-prediction exists.
- **`docs/WORKFLOW.md`**, **`README.md`**: update the variant tables and the Stage-3
  output list (`risk_<v>_forecast.tif`).
- **Memory** (`feedback_naming_conventions` / scope): record the 5-variant scheme and
  `plantation_resid` naming.

---

## 8. Testing

| File | Coverage |
|---|---|
| `tests/process/test_plantation.py` (new) | OLS residual math, NoData masking, R² return, warn>0.85, silent-skip when `dist_plantation_edge.tif` absent; forecast variant uses t3 regressors. |
| `tests/model/test_icar.py` | `build_formula` for D and E; `variant_extra_cols(v)` for all five; D/E NaN-drop subset. |
| `tests/model/test_predict.py` (new case) | `build_forecast_vardir` assembles the full model-named set; `predict_forecast` writes `risk_<v>_forecast.tif`; derived rasters land in the forecast dir. |
| `tests/process/test_distances.py` | renamed forecast outputs (`forecast/dist_edge.tif`, `forecast/dist_defor.tif`); `forecast/dist_plantation_edge.tif` when `plantation_t3` present. |
| `tests/model/test_diagnostics.py` | `extra_cols` via `variant_extra_cols` for D/E. |
| `tests/conftest.py` | synthetic `dist_plantation_edge.tif` (UTM 50S) + t3 fixtures (`forest_t3`, forecast dist rasters). |

All tests run offline against synthetic UTM-50S fixtures, per existing conftest
conventions. Full suite green before commit.

---

## 9. Out of scope (explicit)

- **Forecast validation / accuracy assessment** — Spec B. Options surfaced but not
  chosen: ① in-sample calibration only; ② temporal back-test (train fcc12 → validate
  fcc23, partly blocked by missing t1 covariates: no GHSL-2001, no pre-2001 defor
  front); ③ allocation hindcast on 2012–2024 vs observed `forest_t3` via
  `forestatrisk.validate` (fully feasible). Decision deferred.
- **Rewiring `deforest()` to the t3 surface** — hook left in place; flip deferred
  (decision #8).
- **t3 sources for road/river/protected/HGU** — none exist in the pipeline; these stay
  static-reuse-t2.

---

## 10. Open questions carried forward

1. **Decision #8** — should `project_future`/`deforest()` allocate on the t3 forecast
   risk (`risk_<v>_forecast.tif`) or keep ranking on the t2 `risk_<v>.tif`? Building the
   forecast machinery argues for t3; left as a one-line switch pending the user's call.
