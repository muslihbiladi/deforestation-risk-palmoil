# palmdef_risk — Production Refactor Design Spec
**Date:** 2026-05-19  
**Authoritative design reference:** `WORKFLOW.md`  
**Approach:** Incremental module replacement (Approach 1)

---

## Overview

Refactor `palmoil_risk` into `palmdef_risk` — a production-ready deforestation risk pipeline for palm-oil landscapes. The refactor:

- Renames the package from `palmoil_risk` → `palmdef_risk`
- Replaces the LQ/KDE/SLX/GWR methodology with gravity-based mill accessibility + orthogonalization
- Replaces model variants A–G (LQ-focused) with three nested models A/B/C (biophysical → gravity → HGU spline)
- Adds parallelization, a shared area cache, UTM pre-processing enforcement, and new diagnostics (VIF, Moran's I, gravity bandwidth sensitivity)
- Simplifies each of the 3 notebooks
- Updates `.claude/` with project CLAUDE.md and two pipeline skills

**Deferred (not in this refactor):**
- Leave-one-island-out spatial block cross-validation
- Temporal sub-period robustness refits

---

## Section 1 — Package Structure

### Rename
`palmoil_risk` → `palmdef_risk` everywhere: package directory, `pyproject.toml`, all imports, egg-info.

### Deleted modules
| Module | Reason |
|---|---|
| `process/lq.py` | LQ/KDE methodology replaced by gravity |
| `process/correlation.py` | SLX removed entirely |
| `model/gwr.py` | GWR removed entirely |

### Module map

```
palmdef_risk/
├── io/
│   ├── config.py        ← updated: new fields, drop LQ/GWR/SLX
│   ├── run.py           ← unchanged
│   └── helpers.py       ← kept
├── data/
│   ├── forest.py        ← polished: UTM output enforced
│   ├── variables.py     ← polished: UTM output enforced
│   ├── mill.py          ← rewritten: t2/t3 cumulative filter
│   ├── user_inputs.py   ← kept
│   └── utm.py           ← NEW
├── process/
│   ├── align.py         ← kept + HGU signed-distance replaces binary
│   ├── distances.py     ← NEW (extracted from align.py, parallelized)
│   └── gravity.py       ← NEW
├── model/
│   ├── icar.py          ← rewritten: A/B/C formulas only
│   ├── predict.py       ← minor updates
│   ├── diagnostics.py   ← rewritten: VIF, Moran's I, reliability diagrams
│   └── sensitivity.py   ← NEW: gravity bandwidth sensitivity
└── parallel.py          ← NEW
```

---

## Section 2 — Config Schema

### Dropped fields
`kde_bandwidth_km`, `lq_epsilon`, `lq_direction`, `run_gwr`, `gwr_bandwidth`, `output.scenarios`, `output.risk_classes`

### Full updated template

```yaml
run:
  project: myproject
  area: kalimantan
  task: baseline

aoi:
  source: data/aoi/study_area.gpkg   # vector file OR "xmin,ymin,xmax,ymax"
  buffer: 5000                        # metres

crs: null   # null = auto-detect UTM from AOI centroid; or pin e.g. "EPSG:32749"

cache_dir: cache/   # shared area cache, relative to repo root

forest:
  source: tmf          # "tmf" or "gfc"
  years: [2015, 2020, 2023]
  perc: 75             # gfc only

variables:
  use_ghsl_towns: false
  ghsl_years: null     # required when use_ghsl_towns: true
  osm_timeout: 180

user_inputs:
  peatland:
    path: data/user_inputs/peatland.gpkg
    type: binary        # "binary" or "continuous"
  hgu:
    path: data/user_inputs/hgu.gpkg
  plantation:
    t2: data/user_inputs/plantation_t2.tif
    t3: null
    industrial_value: 1
    smallholder_value: 2

mill:
  source: trase    # "trase" (Trase Open Data) or "user"
  path: null       # required when source: user; must be a point GPKG with earliest_year_of_existence field

process:
  gravity:
    sigma_km: 25.0
    radius_km: 80.0
  sensitivity:
    sigmas_km: [15, 25, 40]

model:
  variants: [A, B, C]   # subset of {A, B, C}; C requires hgu data
  nsamp: 10000
  csize: 10
  Vbeta: 1000           # forestatrisk default; reduce to ~10 under spatial confounding
  burnin: 1000
  mcmc: 1000
  thin: 1
  seed: 42

parallel:
  max_workers: null      # null = adaptive
  cpu_fraction: 0.9
  ram_per_dist_gb: 0.5
  ram_per_icar_gb: 1.0
  ram_per_predict_gb: 0.75

output:
  project_future: false
  projection_year: 2035
```

### `RunConfig` validation additions
- `model.variants` must be non-empty subset of `{A, B, C}`
- If `C` in variants and `hgu_path` missing → skip C with warning (not error)
- `sigma_km > 0`, `radius_km > sigma_km`
- `Vbeta > 0`; if `Vbeta > 100` → log warning about divergent chain risk
- `nsamp > 0`, `csize > 0`
- `len(sensitivity.sigmas_km) >= 1`
- `crs` may be null (auto-detected at `create_run()` time)

---

## Section 3 — Data Download Stage

### 3a. CRS auto-detection

`crs` is optional in YAML (default `null`). During `create_run()`:
- If null → `utm.py:primary_utm_zone(aoi_bbox)` computes and populates `config.crs`
- If set → used as-is

All downloaders receive `output_crs=ctx.config.crs`. Nothing leaves Stage 1 in EPSG:4326. All distance calculations downstream are in metres.

### 3b. `data/utm.py` (NEW)

```python
detect_utm_zones(bbox_4326) -> list[str]
# Returns EPSG codes for every UTM zone the bbox overlaps

primary_utm_zone(bbox_4326) -> str
# Returns EPSG code of zone containing bbox centroid
```

Multi-UTM forest handling (only forest requires per-zone downloads):
- Single zone → download + reproject directly to that CRS
- Multiple zones → download each zone into `raw/forest/zones/{epsg}/`, `gdal.Warp`-mosaic all into primary zone CRS, store flat in `raw/forest/`

SRTM, WDPA, OSM, GHSL: single download then `gdal.Warp` to UTM — no per-zone split.

### 3c. Shared area cache

**Cache directory:** `config.cache_dir` (default `cache/`), shared across all runs.

**Cache key logic (per dataset):**

| Dataset | Cache key | Validity check |
|---|---|---|
| Mill | `{t2_year}_{t3_year}` | Existence only (Indonesia-wide, Trase only) |
| Forest | `hash(aoi, buffer, source, years, perc)` | Spatial coverage: `cached_extent ⊇ new_aoi+buffer` |
| Variables | `hash(aoi, buffer, use_ghsl, ghsl_years, osm_timeout)` | Spatial coverage |

**Cache structure:**
```
cache/
├── mill/{t2}_{t3}/
│   ├── mill_t2.gpkg
│   ├── mill_t3.gpkg
│   └── metadata.json
├── forest/{hash}/
│   ├── forest_t1.tif, forest_t2.tif, forest_t3.tif
│   ├── fcc12.tif, fcc23.tif, fcc123.tif
│   └── metadata.json   # includes downloaded_extent bbox
└── variables/{hash}/
    ├── altitude.tif, slope.tif, protected.gpkg, road.gpkg, river.gpkg, town.gpkg
    └── metadata.json   # includes downloaded_extent bbox
```

**Notebook 01 flow:**
1. Cache-check cell prints per-dataset status:
   ```
   Mill      → cache hit  (trase, t2=2020, t3=2023)
   Forest    → cache hit  (cached extent covers your AOI)
   Variables → no cache   (will download fresh)
   ```
2. User-editable flag:
   ```python
   use_cache = {"forest": True, "variables": False, "mill": True}
   ```
3. Each download cell respects `use_cache`. Fresh downloads write to cache automatically.

### 3d. `forest.py` — polished

`download_forest(ctx, use_cache)` passes `output_crs=ctx.config.crs` and routes multi-UTM via `utm.py`. Core `get_fcc()` logic (tiled GEE download, retry, VRT mosaic, clip, FCC encoding) unchanged.

### 3e. `variables.py` — polished

`download_variables(ctx, use_cache)` passes `output_crs=ctx.config.crs`. Core logic unchanged.

### 3f. `mill.py` — rewritten

```python
download_mill(ctx, use_cache) -> {"mill_t2": Path, "mill_t3": Path}
```

For each period year `t` in `[forest_years[1], forest_years[2]]`:
- Keep mills where `earliest_year_of_existence <= t` **OR** `earliest_year_of_existence` is null
- Clip to AOI + buffer
- Reproject to `ctx.config.crs`
- Write to `raw/mill/mill_t2.gpkg` / `raw/mill/mill_t3.gpkg`

**Mill cache design:** The cache stores date-filtered but **AOI-unfiltered** (Indonesia-wide) mill files, keyed by `(t2_year, t3_year)`. AOI clipping happens at `download_mill()` runtime from the cached full-Indonesia file. This is what makes mill data reusable across different AOIs.

### 3g. `user_inputs.py` — kept unchanged

---

## Section 4 — Process Stage

### 4a. `align.py` — kept + HGU replacement

Reference raster: `forest_t2.tif` (already in UTM from Stage 1). All inputs already in UTM — no CRS conversion at alignment time, only grid snapping. Alignment methods unchanged except:

- **HGU**: binary rasterization removed; replaced with signed-distance surface (see §4d)
- **GHSL**: nearest-neighbour 100 m → 30 m (unchanged)
- **Plantation**: majority resampling 10 m → 30 m (unchanged)

### 4b. `distances.py` — NEW (extracted + parallelized)

`compute_all_distances(ctx)` dispatched via `parallel.py`.

| Output | `values` | Source |
|---|---|---|
| `dist_edge` | 0 | `forest_t2` |
| `dist_defor` | 0 | `fcc12` |
| `dist_road` | 1 | `road` |
| `dist_river` | 1 | `river` |
| `dist_town` | 1 | `town` |
| `dist_plantation_edge` | 1 | `plantation` |
| `dist_ghsl_built` | 1 | `ghsl_built_t2` |
| `dist_edge_forecast` | 0 | `forest_t3` |
| `dist_defor_forecast` | 0 | `fcc23` |

Note: `dist_mill` is **dropped** — mill proximity represented entirely by `gravity_resid.tif`.

All distances in **metres** (UTM inputs).

### 4c. `gravity.py` — NEW

**`compute_gravity_accessibility(ctx) → Path`**

Implements WORKFLOW.md §3.3:
1. Rasterize `mill_t2.gpkg` presence onto reference grid → mill density raster
2. Apply Gaussian filter: `σ_px = (sigma_km × 1000) / pixel_size_m`, truncated at `radius_km`
3. Output: `data/gravity_raw.tif`

Formula: `A_i = Σ_m exp(−d²(i,m) / 2σ²)` — implemented as distance transform + Gaussian filter, not per-pixel loop.

**`orthogonalize_gravity(ctx) → Path`**

1. Load pixel sample: `gravity_raw`, `dist_road`, `dist_town`
2. OLS: `A_i ~ dist_road + dist_town`
3. Residual → `data/gravity_resid.tif`
4. Print R²; if R² > 0.85 → log warning: "Accessibility largely collinear with infrastructure — Model B marginal signal may be weak"

### 4d. HGU signed-distance (in `align.py`)

Replaces binary HGU rasterization with WORKFLOW.md §3.4:
1. Rasterize HGU polygons → binary mask
2. GDAL proximity from exterior → `dist_outside`
3. GDAL proximity from interior → `dist_inside`
4. `hgu_signed_dist = dist_outside − dist_inside` → `data/hgu_signed_dist.tif`
   - Negative inside concessions, positive outside, zero at boundary
   - Units: metres. Knots for Model C spline at −5000 m, 0 m, +5000 m

---

## Section 5 — Model Stage

### 5a. Sampling

`build_sample_data(ctx)` — unchanged from WORKFLOW.md §4.1. Config: `nsamp`, `csize`, `seed`. Output: `output/sample.csv`.

### 5b. Model variants A/B/C

`icar.py:build_formula(variant, ctx)`:

| Model | Covariates |
|---|---|
| **A** | `scale(altitude)`, `scale(slope)`, `scale(log(dist_defor+1))`, `scale(log(dist_edge+1))`, `scale(log(dist_road+1))`, `scale(log(dist_town+1))`, `scale(log(dist_river+1))`, `protected` (binary, not log-transformed) |
| **B** | Model A + `scale(gravity_resid)` |
| **C** | Model B + HGU restricted cubic spline `b1`, `b2` (knots at −5000 m, 0 m, +5000 m; each basis column individually `scale()`'d) |

- `Vbeta` from config (default 1000; warning logged if > 100)
- `beta_start=-99` → initialise from logistic MLE
- Pickle: `betas`, `rho`, `formula`, `betas_mcmc`, `deviance` — no `patsy.DesignInfo`
- If C requested and `hgu_signed_dist.tif` missing → skip C with warning
- Fitting parallelized via `parallel.py` (`_fit_worker`, `ram_per_icar_gb`)
- `dist_plantation_edge` is computed and present in the aligned stack but **not entered into any model formula** (kept for potential future use or user inspection)

### 5c. Prediction

For each fitted variant:
1. Rebuild `x_design_info` from `sample.csv` + saved formula
2. `interpolate_rho` → upsample spatial random effect to 1 km raster
3. `predict_raster_binomial_iCAR` → `output/predictions/risk_{A,B,C}.tif` (UInt16)
4. Optional: `project_future` → `forest_future_{X}.tif`

Parallelized via `parallel.py` (`_predict_worker`, `ram_per_predict_gb`, per-variant temp dir).

### 5d. VIF diagnostic (`diagnostics.py`)

Computed **before** model fitting, for each variant:
1. Load `sample.csv`, extract covariate matrix for that variant's formula
2. `VIF_j = 1 / (1 − R²_j)` — regress column j on all others
3. Flag: `VIF > 5` (moderate), `VIF > 10` (high) in log output
4. Write `output/diagnostics/vif.json`

### 5e. Moran's I (`diagnostics.py`)

For each fitted variant A → B → C:
1. Deviance residuals on iCAR grid
2. Row-standardised weights matrix W (libpysal/esda)
3. Moran's I + p-value
4. Write `output/diagnostics/moran.json` — expected to decline A → B → C
5. Reliability diagrams: decile-binned predicted probabilities vs observed rate → `output/diagnostics/reliability_{A,B,C}.png`

### 5f. Gravity bandwidth sensitivity (`sensitivity.py`)

For each σ in `config.process.sensitivity.sigmas_km` (default `[15, 25, 40]`):
1. Recompute `gravity_raw.tif` at alternate σ (reuses orthogonalization OLS)
2. Refit Model B
3. Extract: accessibility coefficient, 95% CI, ΔWAIC vs Model A
4. Write `output/diagnostics/gravity_sensitivity.json`

---

## Section 6 — `parallel.py`

### Process-based (CPU-bound)

```python
adaptive_workers(ram_per_task_gb, cfg) -> int
# min(available_RAM / ram_per_task, floor(cpu_count × cpu_fraction), max_workers or ∞)

run_parallel(fn, tasks, ram_per_task_gb, cfg) -> list
# ProcessPoolExecutor; falls back to sequential if workers=1
# returns results in submission order
```

| Stage | Worker | RAM config key |
|---|---|---|
| Distances | `_dist_worker` | `ram_per_dist_gb` |
| iCAR fitting | `_fit_worker` | `ram_per_icar_gb` |
| Prediction | `_predict_worker` | `ram_per_predict_gb` |

Worker entry points are module-level (picklable). Data crosses boundary as run-dir path or JSON bytes, never live objects. Prediction workers use per-variant temp dirs.

### Thread-based (I/O-bound)

GEE tile downloads in `forest.py` and `variables.py` use `ThreadPoolExecutor`. Thread count: `min(n_tiles, cpu_count − 1, 10)`. Unchanged.

---

## Section 7 — Notebooks

All notebooks have skip-if-done guards per cell. A crashed run resumes at first incomplete step.

### `01_download.ipynb`

```
[Cell 0] Config + setup
         config_path = "configs/my_run.yaml"
         use_cache = {"forest": False, "variables": False, "mill": True}
         ctx = create_run(config_path)

[Cell 1] Cache check     ← prints per-dataset status; no action taken
[Cell 2] Forest          ← download_forest(ctx, use_cache)
[Cell 3] Variables       ← download_variables(ctx, use_cache)
[Cell 4] Mill            ← download_mill(ctx, use_cache)
[Cell 5] User inputs     ← ingest_user_inputs(ctx)
```

### `02_process.ipynb`

```
[Cell 0] Config + load run
         ctx = load_run("runs/my_run_dir")

[Cell 1] Align           ← align_all(ctx, inputs)
[Cell 2] Distances       ← compute_all_distances(ctx)   [parallel]
[Cell 3] Gravity         ← compute_gravity_accessibility(ctx)
                            orthogonalize_gravity(ctx)   [prints R²]
[Cell 4] HGU distance    ← compute_hgu_signed_distance(ctx)
```

### `03_model.ipynb`

```
[Cell 0] Config + load run
         ctx = load_run("runs/my_run_dir")

[Cell 1] Sample          ← build_sample_data(ctx)
[Cell 2] VIF check       ← compute_vif(ctx)             [prints warnings]
[Cell 3] Fit models      ← fit_all(ctx)                 [parallel A/B/C]
[Cell 4] Predict         ← predict_all(ctx)             [parallel]
[Cell 5] Moran's I       ← compute_morans_i(ctx)
[Cell 6] Sensitivity     ← run_gravity_sensitivity(ctx)
```

---

## Section 8 — `.claude` Folder Updates

### `CLAUDE.md` (project-level)

Documents:
- Package name: `palmdef_risk`
- `WORKFLOW.md` is the authoritative methodology reference
- **Non-negotiable data conventions:**
  - Forest/categorical rasters: `NoData = 255`, dtype `Byte`
  - Float rasters: `NoData = -9999.0`, dtype `Float32`
  - Risk output: `UInt16`, `0 = NoData`, `1–65535 = probability`
  - All rasters in **UTM (metres)** after Stage 1 — never process in EPSG:4326
- Run folder key paths: `ctx.raw_dir`, `ctx.data_dir`, `ctx.output_dir`
- iCAR prior: `Vbeta > 100` risks divergent chains under spatial confounding
- Mill filter: `earliest_year_of_existence <= t_year OR null` (conservative inclusion)
- Cache validity rules: coverage check for forest/variables; existence check for mill
- Gravity formula: Gaussian filter on mill density raster (not per-pixel loop)

### New skill: `pipeline-run`

Executes all three notebooks in sequence via `papermill`, config path as argument.

```
/pipeline-run configs/my_run.yaml
```

### New skill: `pipeline-check`

Inspects a run folder, reports which stages are complete / partial / missing based on canonical output files.

```
/pipeline-check runs/wri_kalteng_20250501_120000
```

---

## Key Invariants (must not change across the codebase)

0. WDPA protected areas output is named `protected.gpkg` / `protected.tif` throughout — **never `pa`** (causes patsy formula parsing errors)

1. Forest FCC encoding: `1` = remained forest, `0` = deforested, `255` = NoData
2. `fcc23.tif` is always the model training response
3. All rasters aligned to `forest_t2.tif` grid
4. Signed HGU distance: negative inside, positive outside, zero at boundary
5. Gravity orthogonalization spec is pre-registered (fixed before model fitting)
6. `patsy.DesignInfo` never pickled — always rebuilt from `sample.csv` at predict time
7. Mill t2/t3 uses cumulative filter (`earliest_year_of_existence <= t OR null`)
