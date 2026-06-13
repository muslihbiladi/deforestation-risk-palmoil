# palmdef_risk — Deforestation Risk Assessment (Palm Oil, Indonesia)

A Python pipeline for modelling spatial deforestation risk driven by palm-oil mill accessibility. Downloads forest cover change and spatial covariates from Google Earth Engine and OpenStreetMap, aligns them to a UTM grid, computes mill accessibility surfaces, and fits Bayesian iCAR spatial models with three progressively richer covariate sets.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Project structure](#3-project-structure)
4. [Before you start — what to prepare](#4-before-you-start--what-to-prepare)
5. [Step 1 — Create your config file](#5-step-1--create-your-config-file)
6. [Step 2 — Run the pipeline](#6-step-2--run-the-pipeline)
7. [Output structure](#7-output-structure)
8. [Aligned rasters reference](#8-aligned-rasters-reference)
9. [Model variants](#9-model-variants)
10. [Diagnostics reference](#10-diagnostics-reference)
11. [UTM zone reference (Indonesia)](#11-utm-zone-reference-indonesia)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.10+ | via palmdef-risk environment |
| GDAL / osgeo | 3.x | included in palmdef-risk |
| forestatrisk | latest | iCAR modelling core |
| Google Earth Engine | — | needs authenticated GCP project |
| osmnx | — | OSM road / river / town download |
| geopandas | — | vector I/O |
| scipy / numpy / pandas | — | analytics |
| psutil | optional | RAM-aware parallelism |
| tqdm | — | progress bars |
| libpysal / esda | optional | Moran's I spatial diagnostics |

---

## 2. Installation

```bash
# 1. Clone or download this repository
git clone <repo-url>
cd deforestation-risk-palmoil

# 2. Activate the conda environment
conda activate palmdef-risk

# 3. Install the package in development mode
pip install -e .

# 4. Authenticate Google Earth Engine
earthengine authenticate
```

> **Windows note:** If PostgreSQL / PostGIS is installed, its `proj.db` conflicts with GDAL.
> The pipeline fixes this automatically at startup by pointing PROJ to the conda environment.

---

## 3. Project structure

```
deforestation-risk-palmoil/
├── configs/
│   ├── template.yaml              ← copy this and fill in your study area
│   ├── central-kalimantan.yaml    ← example run config
│   └── east-kotawaringin.yaml     ← example run config
├── notebooks/
│   ├── 01_download.ipynb          ← Stage 1: download forest, variables, mills
│   ├── 02_process.ipynb           ← Stage 2: align, gravity, distances
│   └── 03_model.ipynb             ← Stage 3: iCAR models, diagnostics, risk maps
├── palmdef_risk/                  ← Python package
│   ├── data/                      ← downloaders (forest, variables, mill, user inputs)
│   ├── io/                        ← config, run context, helpers
│   ├── model/                     ← iCAR fitting, prediction, diagnostics, reports
│   ├── process/                   ← alignment, gravity, distances
│   └── parallel.py                ← adaptive parallel executor
├── tests/                         ← pytest test suite
├── notes/                         ← analytical notes and design decisions
├── docs/                          ← specs and implementation plans
└── runs/                          ← all run outputs land here (auto-created)
```

---

## 4. Before you start — what to prepare

### 4.1 Area of Interest (AOI)

You need either:

**Option A — a vector file** (recommended)
- Format: GeoPackage (`.gpkg`) or Shapefile (`.shp`)
- CRS: any — the pipeline reprojects internally
- Single polygon representing your study area boundary

**Option B — a bounding box string**
- Format: `"xmin,ymin,xmax,ymax"` in decimal degrees (WGS84)
- Example: `"108.5,-2.5,116.0,2.0"`

### 4.2 Required user-supplied inputs

| File | Format | Description |
|---|---|---|
| `peatland.gpkg` | Vector polygon | Peatland extent. Binary (presence) or continuous (depth in metres) — set `type:` in config. |
| `hgu.gpkg` | Vector polygon | Hak Guna Usaha or concession polygons. Used to compute signed distance covariate. |

### 4.3 Optional user-supplied inputs

| File | Format | Description |
|---|---|---|
| `plantation_t2.tif` | Raster | Plantation cover at reference year t2. Pixel values: `industrial_value` and `smallholder_value` (set in config). Needed only when `plantation.source: user`. |
| `plantation_t3.tif` | Raster | Plantation cover at forecast year t3. Optional; needed only when `plantation.source: user`. |
| `river.gpkg` / `.shp` | Vector lines | Custom river / waterway network. If omitted, OSM waterways are downloaded automatically. |

> **Plantation can be downloaded instead of supplied.** Set `plantation.source: download`
> to pull the [Descals *Global oil palm extent and planting year 1990–2021*](https://zenodo.org/records/13379129)
> dataset. The pipeline downloads it once into a shared cache (`cache/plantation_global/`),
> then for each run accumulates the cumulative plantation extent up to `forest.years[1]`
> (t2) and `forest.years[2]` (t3) — every pixel planted from 1990 through the cutoff year,
> classed industrial/smallholder. Years beyond 2021 are clamped to 2021 with a caution.

### 4.4 Google Earth Engine project

You need a GCP project with the Earth Engine API enabled. Set the project ID in your config under `gee_project`. The pipeline calls `ee.Initialize()` automatically.

---

## 5. Step 1 — Create your config file

```bash
cp configs/template.yaml configs/my_study_area.yaml
```

Open the copy and fill in the fields. Key sections:

```yaml
run:
  project: wri
  area: central-kalimantan
  task: baseline

aoi:
  source: data/user_inputs/my_aoi.gpkg
  buffer: 500                      # metres buffered around AOI before downloading

crs: null                          # null = auto-detect UTM from AOI centroid

gee_project: "ee-myproject"        # your GEE cloud project ID

forest:
  source: gfc                      # "tmf" (JRC) or "gfc" (Hansen)
  years: [2001, 2012, 2024]        # [t1, t2, t3]; t2 = reference year, t3 = forecast start
  perc: 30                         # tree-cover threshold in percent, 30 is unofficial number but widely used in tropical areas (gfc only)

variables:
  use_ghsl_towns: false            # true = GHSL built-up surface instead of OSM towns
  ghsl_years: null                 # required if use_ghsl_towns: true, e.g. [2012, 2024], it's available for every 5 years
  osm_timeout: 180

user_inputs:
  peatland:
    path: data/user_inputs/peatland.gpkg
    type: binary                   # "binary" or "continuous"
  hgu:
    path: data/user_inputs/hgu.gpkg
  plantation:
    source: user                   # "download" (Descals Global Oil Palm) or "user"
    t2: null                       # path to plantation raster at t2, or null (source: user only)
    t3: null                       # path to plantation raster at t3, or null (source: user only)
    industrial_value: 1
    smallholder_value: 2
  river:
    path: null                     # path to custom river lines, or null (use OSM)

mill:
  source: trase                    # "trase" (auto-download) or "user" (supply your own)
  path: null                       # required when source: "user"

process:
  gravity:
    sigma_km: 25.0                 # Gaussian kernel σ for mill accessibility (km)
    radius_km: 80.0                # truncation radius (must be > sigma_km)
  sensitivity:
    sigmas_km: [15.0, 25.0, 40.0] # σ values for bandwidth sensitivity sweep

model:
  variants: [A, B, C, D, E]
  nsamp: 10000                     # training sample size
  csize: 10                        # iCAR spatial cell size (km)
  Vbeta: 100                       # prior variance for fixed effects
  burnin: 1000
  mcmc: 1000
  thin: 1
  seed: 42

parallel:
  max_workers: null                # null = auto (RAM + CPU governed)
  cpu_fraction: 0.9
  ram_per_dist_gb: 0.5
  ram_per_icar_gb: 1.0
  ram_per_predict_gb: 0.75

output:
  project_future: false
  projection_year: 2025            # must exceed forest.years[-1]
```

---

## 6. Step 2 — Run the pipeline

Run the three notebooks interactively in Jupyter, in order:

```
01_download.ipynb  →  02_process.ipynb  →  03_model.ipynb
```

Each notebook resumes gracefully: already-present outputs are detected and skipped, so you can re-run cells without re-downloading or recomputing.

To resume an existing run folder rather than creating a new one, set `resume=True` in the notebook's setup cell.

---

## 7. Output structure

```
runs/
└── wri_central-kalimantan_baseline_20260521_112147/
    ├── config.yaml                        ← exact config used (reproducibility)
    ├── logs/run.log
    ├── data/
    │   ├── raw/
    │   │   ├── forest/                    ← downloaded FCC tiles (forest_t1/t2/t3, fcc12/23/123)
    │   │   ├── variables/                 ← SRTM, WDPA, OSM/GHSL downloads
    │   │   ├── mill/                      ← mill_t2.gpkg, mill_t3.gpkg
    │   │   └── user_inputs/               ← copies of peatland, HGU, plantation, river files
    │   ├── intermediate/                  ← temporary reprojected vectors (auto-cleaned)
    │   ├── forecast/                      ← t3-state covariate rasters for forecast prediction
    │   │   ├── dist_edge.tif
    │   │   ├── dist_defor.tif
    │   │   ├── dist_town.tif
    │   │   └── plantation_resid.tif
    │   └── *.tif                          ← all aligned rasters (see Section 8)
    └── output/
        ├── sample.csv                     ← training sample (nsamp rows)
        ├── models/
        │   ├── model_A/mod_A.pkl
        │   ├── model_B/mod_B.pkl
        │   ├── model_C/mod_C.pkl
        │   ├── model_D/mod_D.pkl
        │   └── model_E/mod_E.pkl
        ├── diagnostics/
        │   ├── vif.json                   ← Variance Inflation Factors
        │   ├── moran.json                 ← Moran's I on deviance residuals
        │   ├── gravity_sensitivity.json   ← accessibility coefficient across σ sweep
        │   └── A/  B/  C/  D/  E/        ← per-variant diagnostic outputs (see Section 10)
        └── predictions/
            ├── risk_A.tif                 ← annual deforestation probability (UInt16 scaled)
            ├── risk_B.tif  risk_C.tif  risk_D.tif  risk_E.tif
            ├── risk_A_forecast.tif        ← t3-state forecast risk (same encoding)
            ├── risk_B_forecast.tif  risk_C_forecast.tif  risk_D_forecast.tif  risk_E_forecast.tif
            └── forest_future_*.tif        ← projected binary forest (if project_future: true)
```

---

## 8. Aligned rasters reference

All rasters in `data/` are aligned to `forest_t2.tif` (reference grid, UTM, 30 m).

| File | dtype | NoData | Description |
|---|---|---|---|
| `forest_t1/t2/t3.tif` | Byte | 255 | Binary forest cover at each timestamp (1=forest, 0=non-forest) |
| `fcc12.tif` | Byte | 255 | Forest cover change t1→t2: 1=remained, 0=deforested |
| `fcc23.tif` | Byte | 255 | Forest cover change t2→t3: **model training response** |
| `fcc123.tif` | Byte | 255 | Three-period trajectory |
| `altitude.tif` | Float32 | −9999 | SRTM elevation (metres) |
| `slope.tif` | Float32 | −9999 | Slope (degrees) |
| `protected.tif` | Byte | 255 | WDPA protected areas (1=protected) |
| `road.tif` | Byte | 255 | OSM road presence (1=road) |
| `river.tif` | Byte | 255 | OSM / user-supplied river presence (1=river) |
| `town.tif` | Byte | 255 | OSM settlement presence (1=town) |
| `dist_edge.tif` | Float32 | −9999 | Distance to forest edge at t2 (metres) |
| `dist_defor.tif` | Float32 | −9999 | Distance to past deforestation fcc12 (metres) |
| `dist_road.tif` | Float32 | −9999 | Distance to nearest road (metres) |
| `dist_river.tif` | Float32 | −9999 | Distance to nearest river (metres) |
| `dist_town.tif` | Float32 | −9999 | Distance to nearest town / GHSL centroid (metres) |
| `gravity_raw.tif` | Float32 | −9999 | Raw Gaussian mill accessibility: Σ exp(−d²/2σ²) |
| `gravity_resid.tif` | Float32 | −9999 | Mill accessibility residualised vs road + town distances |
| `hgu_signed_dist.tif` | Float32 | −9999 | Signed distance to HGU boundary (negative inside, positive outside, metres) |
| `peatland.tif` | Byte / Float32 | 255 / −9999 | Peatland (binary=presence; continuous=depth in metres) |
| `plantation.tif` | Byte | 255 | Merged plantation presence raster (optional) |
| `dist_plantation_edge.tif` | Float32 | −9999 | Distance to plantation boundary (optional) |
| `mill.tif` | Byte | 255 | Mill location presence raster |

---

## 9. Model variants

Five variants fit progressively richer covariate sets. All share the same iCAR spatial structure: `logit(p_i) = X_i β + ρ_cell(i)` where ρ is an intrinsic CAR spatial random effect absorbing residual spatial autocorrelation.

| Variant | Covariates | Purpose |
|---|---|---|
| **A** | altitude, slope, protected, log_dist_edge, log_dist_defor, log_dist_road, log_dist_river, log_dist_town | Biophysical baseline — no mill or plantation signal |
| **B** | A + gravity_resid | Adds mill accessibility (orthogonalised vs road + town) |
| **C** | A + plantation_resid | Adds plantation proximity (orthogonalised log-residual vs dist_edge + dist_defor + dist_road) |
| **D** | A + gravity_resid + plantation_resid | Combines mill accessibility and plantation proximity |
| **E** | D + hgu_b1 + hgu_b2 | Adds HGU concession effect via natural spline (knots at −5000 m, 0 m, +5000 m) |

**Recommended run:** `variants: [A, B, C, D, E]`

Use DIC (Deviance Information Criterion) to compare variants — lower is better. Moran's I on deviance residuals should be near zero for all accepted variants.

> **Notes**
> - `dist_plantation_edge` is **not** entered directly into any model formula. Plantation proximity is the orthogonalized residual `plantation_resid` (variants C/D/E). Values are already in log-space — never re-log them.
> - `dist_mill` is **not** a model covariate — mill proximity is represented by `gravity_resid` only.
> - `Vbeta > 100` triggers a warning. Consider reducing the value if the MCMC chain shows signs of poor convergence.

---

## 10. Diagnostics reference

Per-variant outputs are written to `output/diagnostics/<A|B|C|D|E>/`.

| File | Description |
|---|---|
| `summary_icar.txt` | MCMC posterior summary: mean, SD, 2.5 / 97.5 % credible intervals for each β |
| `accuracy_summary.txt` | sklearn classification report (digits=4) at 0.5 probability threshold |
| `roc_calibration.png` | ROC curve + calibration plot |
| `mcmc_trace_*.png` | MCMC trace plots for each fixed-effect coefficient |
| `mcmc_autocorr_*.png` | MCMC autocorrelation plots |
| `mcmc_ess.txt` | Effective Sample Size for each coefficient |
| `risk_map.png` | Spatial map of annual deforestation probability |
| `rho_map.png` | Smooth interpolated spatial random effect ρ (1 km cubic spline) |
| `risk_histogram.png` | Histogram of predicted probability values across the AOI |

Run-level outputs in `output/diagnostics/`:

| File | Description |
|---|---|
| `vif.json` | Variance Inflation Factors for all covariates. VIF > 5 triggers a warning. |
| `moran.json` | Moran's I (k=8 KNN) on deviance residuals per variant. Values near 0 confirm spatial autocorrelation is absorbed. |
| `gravity_sensitivity.json` | Accessibility coefficient and mean deviance at each σ in `sensitivity.sigmas_km`. Use to assess robustness to kernel bandwidth choice. |
| `csize_icar.txt` | Cell size and number of iCAR spatial cells used. |
| `fcc_history_map.png` | Forest cover change history map (fcc123). |

---

## 11. UTM zone reference (Indonesia)

| Longitude band | Zone | EPSG (S hemisphere) | EPSG (N hemisphere) |
|---|---|---|---|
| 96–102 °E | 47 | EPSG:32747 | EPSG:32647 |
| 102–108 °E | 48 | EPSG:32748 | EPSG:32648 |
| 108–114 °E | 49 | EPSG:32749 | EPSG:32649 |
| 114–120 °E | 50 | EPSG:32750 | EPSG:32650 |
| 120–126 °E | 51 | EPSG:32751 | EPSG:32651 |
| 126–132 °E | 52 | EPSG:32752 | EPSG:32652 |
| 132–138 °E | 53 | EPSG:32753 | EPSG:32653 |
| 138–144 °E | 54 | EPSG:32754 | EPSG:32654 |

Set `crs: null` to auto-detect from AOI centroid.

**Examples:** Central Kalimantan → EPSG:32749. East Kalimantan (east of 114°E) → EPSG:32750. North Sumatra → EPSG:32647.

---

## 12. Troubleshooting

**`ee.Initialize()` fails**
Run `earthengine authenticate` and ensure the Earth Engine API is enabled for your GCP project at [console.cloud.google.com](https://console.cloud.google.com).

**OSM download times out**
Increase `variables.osm_timeout` (e.g. `730`). For large AOIs the Overpass query can take several minutes.

**PROJ / proj.db conflict (Windows + PostGIS installed)**
The pipeline sets `PROJ_LIB` / `PROJ_DATA` automatically at startup by detecting the conda environment's `proj.db`. If PROJ errors still appear, confirm the palmdef-risk environment is active.

**Variables re-download even though some files exist**
The downloader now checks each output file individually. If a partial set is present, only the missing outputs are downloaded. Delete the entire `raw/variables/` folder to force a full re-download.

**`reproject_vector` produces "No such file or directory" on intermediate GPKG**
This means a vector was already in the target CRS, so `reproject_vector` returned the input path without creating the intermediate file. This is now fixed — the return value is used correctly for rasterization.

**`Vbeta` warning appears**
The prior variance for fixed effects is > 100. Consider reducing the value if the MCMC chain shows signs of poor convergence (e.g. elevated Moran's I on residuals, unstable betas).

**DIC shows `None`**
MCMC chain was too short. Increase `burnin` and `mcmc` to at least 500 each.

**Run takes too long**
- Reduce `nsamp`, `burnin`, `mcmc` for a quick test
- Use `variants: [A]` first to verify end-to-end
- Reduce `max_workers` or lower `cpu_fraction` if RAM is the bottleneck
