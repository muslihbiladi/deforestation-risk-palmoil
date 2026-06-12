# Palm-Oil Deforestation Risk — System Workflow

End-to-end pipeline that models and maps deforestation risk in oil-palm
landscapes. Three notebook stages — **download → process → model** — each
driven by a single YAML config and an isolated, timestamped run folder.

---

## 1. Run model & directory layout

A run is described by one YAML config, parsed into a `RunConfig` dataclass and
validated (`palmdef_risk/io/config.py`). `create_run()` then materialises a
timestamped, self-contained run folder:

```
runs/{project}_{area}_{task}_{YYYYMMDD_HHMMSS}/
├── config.yaml                  # frozen copy of the config used
├── data/
│   ├── raw/
│   │   ├── forest/               # downloaded forest cover (+ zones/ if multi-UTM)
│   │   ├── variables/            # SRTM, WDPA, OSM, GHSL
│   │   ├── mill/                 # mill_t2.gpkg, mill_t3.gpkg
│   │   └── user_inputs/          # peatland, hgu, plantation
│   ├── intermediate/             # reprojected vectors, merged plantation
│   └── *.tif                     # ALIGNED 30 m rasters (the model-ready stack)
├── output/
│   ├── sample.csv                # training sample
│   ├── models/model_{A..F}/      # fitted iCAR pickles
│   └── predictions/              # risk maps, rho, future forest
└── logs/run.log
```

`RunContext` exposes the key paths: `ctx.raw_dir` (downloads),
`ctx.data_dir` (aligned stack), `ctx.output_dir` (model artefacts). Every
function in the pipeline takes a `ctx` and resolves paths from it, so a run is
fully reproducible and never writes outside its own folder.

All three notebooks carry **skip-if-done guards**: each cell first checks
whether its canonical output already exists on disk and, if so, reconstructs
its result dict instead of recomputing. A re-run after a crash resumes at the
first incomplete step.

---

## 2. Stage 1 — Download (`01_download.ipynb`)

Four independent data families are pulled. Each is a separate guarded cell, so
a failure in one does not force re-downloading the others.

### 2.1 Handling multiple download sources

| Family | Source options | Module | Output |
|--------|----------------|--------|--------|
| Forest cover | `tmf` (JRC) / `gfc` (Hansen) via GEE | `data/forest.py` | `forest_t1/2/3.tif`, `fcc12.tif`, `fcc23.tif`, `fcc123.tif` |
| Variables | SRTM, WDPA, OSM, GHSL — or `user` per layer | `data/variables.py` | `altitude/slope.tif`, `protected/road/river/town.gpkg` |
| Mill | `trase` / `user` | `data/mill.py` | `mill_t2.gpkg`, `mill_t3.gpkg` (+ `capacity_tonnes_ffb_hour`) |
| User inputs | local files in `data/user_inputs/` | `data/user_inputs.py` | `peatland`, `hgu`, `plantation_t2/t3` |

**Per-layer source override.** `protected`, `road`, `river`, `town`, and
`mill` each have a `source` key. Setting it to `user` makes the downloader
ingest (validate CRS + geometry type, then copy) a local file instead of
hitting the network. Mixed runs — e.g. WDPA + user roads — are normal.

**GEE tiled download** (forest, SRTM, GHSL share one mechanism):

1. Parse AOI → bounding box in EPSG:4326, apply buffer.
2. **Snap** the extent to the global pixel grid so adjacent tiles never
   misalign during mosaicking.
3. Build a tile grid; tile size is auto-computed to keep each
   `ee.data.computePixels` request under the ~48 MB limit
   (`safe_pixels = 50 MB / (n_bands × bytes_per_pixel)`).
4. Download tiles concurrently (see §6) — each tile retried up to 3× with
   exponential backoff; a failed tile returns `None` rather than aborting.
5. Mosaic tiles via a GDAL **VRT** → translate to a compressed GeoTIFF.
6. Clip to the AOI vector boundary (true cutline clip, not just bbox).
7. Tiles are deleted after mosaicking.

> The repeated `TIFFReadDirectory … ExtraSamples` warnings seen during a run
> come from GDAL reading these EE tiles at the mosaic step. They are harmless
> metadata mismatches and do not stop the pipeline.

**Multi-UTM handling** (`data/utm.py`). `detect_utm_zones()` derives every UTM
zone the AOI bbox overlaps. If the AOI spans one zone, forest is downloaded
directly in that zone's CRS. If it spans several, each zone is downloaded
separately into `forest/zones/{epsg}/`, then all zones are `gdal.Warp`-mosaicked
into the **primary zone's** CRS (the zone containing the AOI centroid). The
chosen CRS is stored as `config.detected_crs` and becomes the project CRS for
every downstream layer.

**Forest cover encoding.** `fccXY.tif` rasters use: `1` = remained forest,
`0` = deforested during the period, `255` = NoData (outside AOI or not forest
at the period start — i.e. outside the analysis domain). `fcc23.tif` is the
primary modelling target.

---

## 3. Stage 2 — Process (`02_process.ipynb`)

Turns heterogeneous raw downloads into a single, perfectly co-registered 30 m
raster stack, then derives distance and accessibility surfaces.

### 3.1 Aligning process (`align_all`, `process/align.py`)

Everything is aligned to **`forest_t2.tif`** — the reference raster. Its grid
(CRS, extent, pixel size, dimensions) defines `mask_props`, including an
`invalid_mask` of out-of-AOI pixels. Each input type is aligned by the method
appropriate to its data type:

| Input | Method | dtype |
|-------|--------|-------|
| Forest rasters | copied as-is (already in project CRS) | Byte |
| SRTM altitude / slope | reproject-to-match, **bilinear** resampling | Float32 |
| Vectors (protected, road, river, town) | reproject vector → **rasterize** (burn = 1) | Byte |
| Peatland | rasterize (binary) *or* reproject (continuous) | Byte / Float32 |
| HGU | reproject vector → rasterize | Byte |
| Plantation t2/t3 | `merge_plantation` (industrial **or** smallholder value → 1) → reproject **majority**(10 m → 30 m) | Byte |
| Mill t2/t3 | rasterize presence | Byte |
| GHSL built-up t2/t3 | reproject **near** (100 m → 30 m grid) | Byte |

Continuous data uses bilinear resampling; categorical/presence data uses
nearest-neighbour (downlsampling) or majority (upsampling) so class codes are not blended. Every aligned raster has the
`invalid_mask` applied, so all layers share an identical NoData footprint.
Outputs land in `ctx.data_dir` as `{name}.tif`.

### 3.2 Distance calculation logic (`compute_all_distances`)

Distances are Euclidean distance transforms computed by
`forestatrisk.data.compute.compute_distance()`. The critical parameter is
**`values`**, which selects what the distance is measured *to*:

- **`values=0`** — distance to non-feature / background pixels. Used for
  *edge* and *deforestation* surfaces, where the meaningful gradient is the
  approach toward already-cleared land:
  `dist_edge` (from `forest_t2`), `dist_defor` (from `fcc12`), and their
  `*_forecast` variants from t3 layers.
- **`values=1`** — distance to the nearest **feature** pixel. Used for all
  presence rasters: `dist_road`, `dist_river`, `dist_town`, `dist_mill`,
  `dist_plantation_edge`, `dist_ghsl_built`. With `values=0` these would
  measure distance *away from* non-features and be meaningless.

A self-repair guard (`_repair_zero_variance_raster`) detects a `dist_mill.tif`
accidentally produced with `values=0` — its non-NoData pixels have near-zero
variance — deletes it, and recomputes with `values=1`.

Forecast distances (`dist_*_forecast`) repeat the same logic on t3 inputs to
support future projection. All distances run in parallel (§6).

### 3.3 Gravity-weighted mill accessibility (orthogonalized)

A mill **accessibility** surface built as a gravity-decay score — a
reformulation of the Two-Step Floating Catchment Area (2SFCA) concept that
sidesteps the supply/demand ratio and its demand-semantics assumptions.

**Source.** The Trase Universal Mill List — 1,875 verified mills globally,
filtered to Indonesia (≈600–800 mills).

**Gaussian decay score.** For each forest pixel *i*:

```
A_i = Σ_m exp(-d²(i,m) / 2σ²)
```

summed over every mill *m* within 80 km, with σ = 25 km. In Python this is
implemented as a distance transform plus a Gaussian filter applied to a
mill-density raster, rather than a per-pixel loop over mills.

**Orthogonalization.** `A_i` is partly redundant with generic infrastructure
proximity. It is therefore regressed on the pixel sample against road and
town distance, and the residual becomes the model covariate:

```
A_resid = A_i − (α + β₁·dist_road_i + β₂·dist_town_i)
```

The orthogonalization specification is **pre-registered** — fixed before any
model is fitted — so Model B (§4.2) tests the *marginal* supply-chain signal,
not the part already carried by infrastructure distance.

**Honesty diagnostic.** The R² of the `A_i ~ dist_road + dist_town`
regression is reported. If R² > 0.85 — accessibility is largely collinear
with infrastructure — the Claim 2 language is softened accordingly.

### 3.4 HGU signed-distance raster

A signed-distance surface to HGU concession boundaries, feeding the HGU
spline term of Model E (§4.2).

**Source.** User-supplied HGU concession polygons.

**Processing:**

1. Rasterize the HGU polygons to a 30 m binary mask.
2. GDAL proximity (Euclidean distance) from the **exterior** → `dist_outside`.
3. GDAL proximity from the **interior** → `dist_inside`.
4. Signed distance = `dist_from_inside − dist_from_outside` — **negative inside**
   concessions (inside ≈ 0 minus outside positive = negative), **positive outside**
   (outside positive minus inside ≈ 0 = positive), **zero at the boundary**.

Because the surface is signed, a spline fitted on it can bend differently on
either side of the boundary (§4.2) — the inside and outside slopes are not
forced to mirror each other. No clip is applied: the spline basis columns
are computed in the original km scale (knots fixed at −5, 0, +5 km) and
then standardized individually with `scale()`, so extreme values simply
extrapolate linearly beyond the outer knots without distorting the knot
positions.

### 3.5 Plantation proximity residual (`plantation_resid`)

An orthogonalized plantation-proximity covariate that isolates plantation
boundary proximity from the generic landscape connectivity already captured
by `dist_edge`, `dist_defor`, and `dist_road`.

**Source.** User-supplied plantation boundary polygons (industrial and/or
smallholder, merged to a single binary presence raster at align time).

**Processing (`process/plantation.py`):**

1. Align the plantation presence raster and compute Euclidean distance →
   `dist_plantation_edge.tif`.
2. Regress `log(dist_plantation_edge)` on
   `log(dist_edge) + log(dist_defor) + log(dist_road)` using OLS (pixel sample).
3. The model residual becomes `plantation_resid`:

```
plantation_resid = log(dist_plantation_edge) − OLS(log(dist_plantation_edge) ~ log(dist_edge) + log(dist_defor) + log(dist_road))
```

The output raster is written to `data/plantation_resid.tif` and the matching
sample column is named `plantation_resid`. **The values are already in
log-space — never apply a log transform again.**

The orthogonalization is pre-registered. Models C, D, and E (§4.2) test the
*marginal* plantation-boundary signal after controlling for landscape proximity
gradients.

---


## 4. Stage 3 — Model (`03_model.ipynb`)

### 4.1 Sampling methodology (`build_sample_data`)

Pixels are sampled from the aligned stack with `forestatrisk.data.sample`:

- **`nsamp`** pixels (default 10 000), 
- **`adapt=True`** — adaptive sampling that balances the two outcome classes.
  Deforestation is rare, so naive random sampling would yield almost no
  deforested pixels; adaptive sampling oversamples the minority class so the
  model sees both.
- **`csize`** — the side length (km) of the coarse grid of spatial cells. Each
  sampled pixel is tagged with its `cell` id; these cells carry the iCAR
  spatial random effect.
- **`seed`** fixes the draw for reproducibility; `blk_rows=128` controls
  windowed raster reads.

The result, `output/sample.csv`, has one row per pixel: `x`, `y`, `fcc23`
(0 = deforested, 1 = still forest), `cell`, and every covariate value sampled
from the rasters.

### 4.2 iCAR model fitting — five nested models

A Bayesian spatial logistic regression (`forestatrisk.model_binomial_iCAR`) is
fitted for each of five pre-registered nested models, each adding covariates
to the previous. Each fit records WAIC, explained deviance, the posterior
mean + 95 % CI for every β coefficient, and Rhat for convergence.

| Model | Covariates | Assessment |
|---|---|---|
| **A — Biophysical baseline** | altitude, slope, dist_edge, dist_road, dist_town, dist_river, dist_defor, pa_status (all z-score standardized) | WAIC, explained deviance, β posteriors, Rhat |
| **B — + gravity accessibility** | Model A + the orthogonalized gravity accessibility residual (§3.3) | ΔWAIC vs. A; Δdeviance; accessibility coefficient — direction, magnitude, 95 % CI |
| **C — + plantation residual** | Model A + the orthogonalized plantation proximity residual (`plantation_resid`, §3.5) | ΔWAIC vs. A; Δdeviance; plantation coefficient |
| **D — + gravity + plantation** | Model A + gravity_resid + plantation_resid | ΔWAIC vs. B and C; joint coefficient posteriors |
| **E — + HGU spline** | Model D + the HGU restricted cubic spline | ΔWAIC vs. D; Δdeviance; spline marginal-effect plot; inside- vs. outside-slope comparison |

**HGU spline specification.** Model E enters the §3.4 signed distance through
a **restricted cubic spline with 3 knots** at −5 km, 0 km (the boundary), and
+5 km. The basis columns b₁, b₂ are computed in the original km scale so the
knot positions are always correct regardless of the value distribution; each
basis column is then standardized individually with `scale()` before entering
the model. Pixels far from any concession extrapolate linearly beyond the
+5 km outer knot — well-behaved by the restricted-spline constraint. The
asymmetric form lets the inside and outside slopes differ freely: the
**inside slope** characterizes legal conversion intensity; the **outside
slope** characterizes leakage / encroachment around concessions.

**HGU specification comparison (Block 2).** Three forms of the HGU term are
compared head-to-head:

| Form | HGU term |
|---|---|
| `C_linear` | signed distance as a single linear term |
| `C_binary` | inside/outside binary indicator |
| `C_spline` | restricted cubic spline (preferred) |

`C_spline` vs. `C_linear` is tested with a likelihood-ratio test, alongside a
deviance comparison.

`build_formula` assembles each formula **dynamically** from whichever rasters
actually exist, so missing optional layers degrade gracefully. `cellneigh`
builds the spatial neighbour structure at the `csize` grid. MCMC runs for
`burnin + mcmc` iterations with thinning `thin`; `beta_start=-99` initialises
from the logistic-regression MLE, and `Vbeta` sets the prior variance —
**use ≈10**, not the forestatrisk default of 1000, which causes divergent
chains under spatial confounding.

Each fit saves a numeric-only pickle (`betas`, `rho` posterior mean, `formula`,
full `betas_mcmc` chain for diagnostics, `deviance`). patsy's `DesignInfo` is
deliberately not pickled — it is rebuilt at prediction time from `sample.csv`.

### 4.3 Prediction (`run_all_predictions` → `predict_risk`)

For each fitted model:

1. Rebuild `x_design_info` from `sample.csv` + the saved formula so `scale()`
   mean/std match those used at fit time.
2. `interpolate_rho` — upsample the per-cell spatial random effect from the
   `csize` grid to a 1 km raster.
3. `predict_raster_binomial_iCAR` — combine covariates + rho into
   `risk_{model}.tif`. Output is UInt16: values 1–65535 encode probability,
   `0` = NoData. Only currently-forested pixels are predicted.
4. `predict_forecast` — re-predicts risk using t3-state covariates assembled
   into `data/forecast/` by `build_forecast_vardir`, producing
   `predictions/risk_{v}_forecast.tif` per variant. (Note: `project_future` /
   `deforest()` still ranks on the t2 risk map; switching to the forecast risk
   for ranking is deferred.)
5. Optional `project_future`: for pixels forested at t3, compute
   `P(survive) = (1 − p_annual)^n_years` and write a binary
   `forest_future_{model}.tif`.

### 4.4 Spatial validation

**Leave-one-island-out cross-validation.** In-sample AUC is optimistic under
spatial autocorrelation. Model E is therefore validated with spatial block
cross-validation: four folds, each holding out one island — Sumatra,
Kalimantan, Papua, Sulawesi. Each fold refits the full Model E on the
remaining three islands, predicts on the held-out island, and computes AUC
and Brier score. Budget ≈ 2–3 h per fold, ≈ 8–12 h total.

**Residual Moran's I.** Deviance residuals for Models A, B, C, D, and E are computed
on the 10×10 km iCAR grid, then Moran's I is computed with a row-standardized
weights matrix W (libpysal / esda). Moran's I is expected to **decline from
A → E** as the iCAR term absorbs the remaining autocorrelation.

**Reliability diagrams.** Predicted probabilities are binned in deciles and
the observed deforestation rate is computed per bin, plotted per island as a
4-panel figure. A well-calibrated model has its reliability curve sitting
near the 1:1 diagonal.

### 4.5 Robustness and sensitivity

**Temporal robustness.** Model E is refit on two configurable sub-periods —
2017–2019 (`lossyear ∈ {17,18,19}`) and 2019–2021 (`lossyear ∈ {19,20,21}`)
— and the HGU spline shape and accessibility coefficient are compared across
them. This addresses the temporal mismatch of the 2019-vintage HGU layer.

**Gravity bandwidth sensitivity (optional).** Model B is refit with σ = 15 km
and σ = 40 km; the accessibility coefficient and ΔWAIC are compared against
the σ = 25 km baseline to confirm the result is not an artefact of the decay
bandwidth.

### 4.6 Inference strategy — all-outcome publishable

The A → B → C → D → E comparison is designed so that **every outcome yields a
defensible, policy-relevant claim** — there is no result that leaves the
analysis without a finding.

| Comparison outcome | Defensible claim |
|---|---|
| B ≫ A (accessibility significant) | Supply-chain pull carries spatial signal in deforestation risk beyond generic infrastructure proximity |
| B ≈ A (accessibility null) | Mill accessibility adds no marginal predictive value once road/town proximity is controlled — economic pull is already captured by infrastructure distance |
| C ≫ A (plantation significant) | Plantation proximity creates a measurable deforestation gradient beyond biophysical factors |
| C ≈ A (plantation null) | Plantation boundaries add no marginal signal beyond biophysical geography at 30 m scale |
| D ≫ B,C (joint signal) | Mill accessibility and plantation proximity carry independent spatial signals |
| E ≫ D (HGU significant, asymmetric) | Concession boundaries create a spatially asymmetric deforestation gradient consistent with leakage / containment / encroachment |
| E ≫ D (HGU significant, symmetric) | Concession boundaries create a uniform distance gradient; legal conversion and encroachment are equally intensive |
| E ≈ D (HGU null) | Concession boundaries add no marginal signal beyond supply-chain and plantation geography — boundary enforcement is spatially redundant |
| All ≈ A | Biophysical geography dominates; supply-chain and institutional variables are spatially redundant at 30 m scale |

---

## 5. Cross-cutting — Parallelization (`parallel.py`)

Two parallelism models are used, matched to the workload:

**Process-based (`run_parallel` + `adaptive_workers`)** — for CPU-bound work.
`adaptive_workers` picks a safe worker count as
`min(available_RAM / ram_per_task, cpu_count × cpu_fraction)` (default
`cpu_fraction = 0.90`). `run_parallel` dispatches tasks to a
`ProcessPoolExecutor`, falls back to sequential execution when only one worker
is warranted, and returns results in submission order. Used by:

| Stage | RAM/task | Worker entry point |
|-------|----------|--------------------|
| Distance computation | `ram_dist_gb` ≈ 0.50 | `_dist_worker` |
| iCAR fitting | `ram_icar_gb` ≈ 1.0 | `_fit_worker` |
| Risk prediction | `ram_predict_gb` ≈ 0.75 | `_predict_worker` |

Worker functions are **module-level** so `ProcessPoolExecutor` can pickle them;
data crosses the process boundary as a run-dir path (re-loaded via `load_run`)
or as JSON bytes, never as live objects. Prediction workers each use a
**per-variant temp dir** so they do not race on forestatrisk's hardcoded
`rho_orig.tif`.

**Thread-based (`ThreadPoolExecutor`)** — for the I/O-bound GEE tile
downloads, which spend their time waiting on the network. Up to
`min(n_tiles, cpu_count − 1, 10)` threads.

`config.parallel.max_workers` pins the worker count for any stage; leaving it
`null` selects the adaptive count.

---

## 6. Expected outputs

After a full three-stage run, the run folder contains:

| Path | Content |
|------|---------|
| `data/raw/**` | Raw downloaded source data (forest, variables, mill, user inputs) |
| `data/*.tif` | Aligned 30 m raster stack — covariates, distances, SFCA surfaces |
| `data/forecast/plantation_resid.tif` | Plantation proximity residual raster for t3-state forecast covariates |
| `output/sample.csv` | Training sample (pixels × covariates + outcome + cell) |
| `output/models/model_{A..E}/mod_{X}.pkl` | Fitted iCAR models (betas, rho, MCMC chain, deviance) |
| `output/predictions/risk_{X}.tif` | Deforestation risk maps (t2-state) — UInt16, 1–65535 = probability, 0 = NoData |
| `output/predictions/risk_{X}_forecast.tif` | Deforestation risk maps (t3-state covariates) — same encoding |
| `output/predictions/rho_{X}.tif` | Interpolated spatial random-effect surface |
| `output/predictions/forest_future_{X}.tif` | Optional future-forest projection (binary) |
| `logs/run.log` | Full run log |

The `risk_{X}.tif` maps are the headline deliverable: per-pixel annual
probability of deforestation for each model variant, ready for classification
into risk zones or comparison across variants.
