# palmoil_risk — Deforestation Risk Assessment (Palm Oil, Indonesia)

A Python pipeline for modelling spatial deforestation risk at the landscape scale. Downloads forest cover change data and spatial covariates from Google Earth Engine and OpenStreetMap, aligns them to a common grid, computes Location Quotient (LQ) surfaces, runs spatial correlation tests (SLX), and fits Bayesian ICAR models with optional Geographically Weighted Regression.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Project structure](#3-project-structure)
4. [Before you start — what to prepare](#4-before-you-start--what-to-prepare)
5. [Step 1 — Create your config file](#5-step-1--create-your-config-file)
6. [Step 2 — Run the pipeline](#6-step-2--run-the-pipeline)
7. [Output structure](#7-output-structure)
8. [Expected outputs explained](#8-expected-outputs-explained)
9. [Model variants](#9-model-variants)
10. [UTM zone reference (Indonesia)](#10-utm-zone-reference-indonesia)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.10+ | via conda-far |
| GDAL / osgeo | 3.x | included in conda-far |
| forestatrisk | latest | ICAR modelling core |
| Google Earth Engine | — | needs authenticated GCP project |
| osmnx | — | OSM road/river/town download |
| geopandas | — | vector I/O |
| scipy / numpy / pandas | — | analytics |
| papermill | — | notebook execution |
| mgwr | optional | only needed if `run_gwr: true` |

---

## 2. Installation

```bash
# 1. Clone or download this repository
git clone <repo-url>
cd deforestation-risk-palmoil

# 2. Activate the conda environment (must be created separately)
conda activate conda-far

# 3. Install the package in development mode
pip install -e .

# 4. Authenticate Google Earth Engine
earthengine authenticate
```

---

## 3. Project structure

```
deforestation-risk-palmoil/
├── configs/
│   └── template.yaml          ← copy this and fill in your study area
├── notebooks/
│   ├── 01_download.ipynb      ← Stage 1: download all data
│   ├── 02_process.ipynb       ← Stage 2: align, LQ, SLX
│   └── 03_model.ipynb         ← Stage 3: ICAR models, risk maps
├── palmoil_risk/              ← Python package (do not edit)
├── run.py                     ← CLI runner
└── runs/                      ← all outputs land here (auto-created)
```

---

## 4. Before you start — what to prepare

### 4.1 Area of Interest (AOI)

You need either:

**Option A — a vector file** (recommended)
- Format: GeoPackage (`.gpkg`) or Shapefile (`.shp`)
- CRS: any projected or geographic CRS (the pipeline reprojects internally)
- Single polygon representing your study area boundary
- Suggested location: `data/aoi/my_study_area.gpkg`

**Option B — a bounding box string**
- Format: `"xmin,ymin,xmax,ymax"` in decimal degrees (WGS84)
- Example: `"108.5,-2.5,116.0,2.0"` for central Kalimantan

### 4.2 User-supplied spatial inputs

Prepare these files before running the pipeline. Store them anywhere accessible — you point to them in the config.

| File | Format | Description |
|---|---|---|
| `peatland.gpkg` | Vector polygon | Peatland extent. May include depth attribute for continuous mode. |
| `hgu.gpkg` | Vector polygon | Hydrological Geomorphic Units (HGU) classes. |
| `plantation_t2.tif` | Raster (GeoTIFF) | Plantation cover at your reference year (t2). Must have distinct pixel values for industrial vs smallholder classes, e.g. 1=industrial, 2=smallholder, 0=no plantation. |
| `plantation_t3.tif` | Raster (GeoTIFF) | Plantation cover at forecast year (t3). Optional — omit if not available. |

> **Tip:** A sensible folder to keep these is `data/user_inputs/` at the repo root. The pipeline copies them into the run folder automatically.

### 4.3 Google Earth Engine project

You need a GCP project with Earth Engine API enabled. Add your project ID to the config under `aoi` → the pipeline calls `ee.Initialize()` automatically inside the notebooks.

---

## 5. Step 1 — Create your config file

```bash
cp configs/template.yaml configs/kalimantan_baseline.yaml
```

Open `configs/kalimantan_baseline.yaml` and fill in the fields:

```yaml
run:
  project: wri_palmoil        # short slug — used in run folder name
  area: kalimantan_tengah     # study area label
  task: baseline              # task label (baseline, scenario_a, …)

aoi:
  source: data/aoi/kalteng.gpkg   # path to your AOI vector file
  buffer: 5000                    # buffer in metres added before downloading data

crs: "EPSG:32749"   # UTM zone for your area — see Section 10 below

forest:
  source: tmf                     # tmf (JRC) or gfc (Hansen)
  years: [2015, 2020, 2023]       # [t1, t2, t3] — three timestamps required
  perc: 75                        # tree cover % threshold (gfc only)

variables:
  use_ghsl_towns: false           # true = use GHSL built-up surface instead of OSM towns
  osm_timeout: 180

user_inputs:
  peatland:
    path: data/user_inputs/peatland.gpkg
    type: binary                  # binary or continuous (depth in metres)
  hgu:
    path: data/user_inputs/hgu.gpkg
  plantation:
    t2: data/user_inputs/plantation_t2.tif
    t3: null                      # omit or set to path of t3 plantation raster
    industrial_value: 1
    smallholder_value: 2

mill:
  source: trase                   # trase or gfw

process:
  kde_bandwidth_km: 35.0
  lq_direction: mp                # mp = mills/plantation; pm = plantation/mills

model:
  variants: [A, B, E, F]         # which model variants to fit (A–G)
  burnin: 1000
  mcmc: 1000
  thin: 1
  run_gwr: false

output:
  project_future: false
  projection_year: 2035
  risk_classes: 5
```

**Validate your config without running anything:**

```bash
python run.py --config configs/kalimantan_baseline.yaml --dry-run
```

---

## 6. Step 2 — Run the pipeline

### Run all three stages in sequence

```bash
conda activate conda-far
python run.py --config configs/kalimantan_baseline.yaml
```

The pipeline creates a timestamped run folder under `runs/` and executes the three notebooks in order. Progress is logged to both the console and `runs/<run_folder>/logs/run.log`.

### Run a single stage

```bash
# Stage 1 only
python run.py --config configs/kalimantan_baseline.yaml --notebook 01_download

# Stage 2 only (re-use an existing run)
python run.py --config configs/kalimantan_baseline.yaml --notebook 02_process --run-dir runs/wri_palmoil_kalimantan_tengah_baseline_20250512_143022
```

### Run notebooks interactively

If you prefer Jupyter for exploration:

```bash
conda activate conda-far
jupyter notebook notebooks/01_download.ipynb
```

Change the `config_path` parameter cell to point to your config before running.

---

## 7. Output structure

Every run creates a self-contained timestamped folder:

```
runs/
└── wri_palmoil_kalimantan_tengah_baseline_20250512_143022/
    ├── config.yaml                     ← copy of your config (reproducibility)
    ├── logs/
    │   └── run.log
    ├── data/
    │   ├── raw/
    │   │   ├── forest/                 ← downloaded FCC tiles
    │   │   ├── variables/              ← SRTM, WDPA, OSM downloads
    │   │   ├── mill/                   ← mill.gpkg
    │   │   └── user_inputs/            ← copies of your peatland/HGU/plantation files
    │   └── intermediate/
    │       └── kde/                    ← mill KDE surface
    │   (+ all aligned rasters, flat)   ← forest_t2.tif, dist_road.tif, lq_mp.tif …
    └── output/
        ├── sample.csv                  ← training sample drawn from data/
        ├── models/
        │   ├── model_A/mod_A.pkl
        │   ├── model_B/mod_B.pkl
        │   └── …
        ├── diagnostics/
        │   ├── moran.json
        │   └── dic_table.csv
        ├── predictions/
        │   ├── risk_A.tif
        │   ├── risk_B.tif
        │   └── forest_future_A.tif     ← only if project_future: true
        ├── correlation/
        │   ├── slx_results.json
        │   └── slx_report.txt
        └── gwr/                        ← only if run_gwr: true
            ├── gwr_coefficients.csv
            └── gwr_summary.json
```

---

## 8. Expected outputs explained

### Aligned rasters (in `data/`)

These are the core inputs to the model — all aligned to `forest_t2.tif` as the reference grid.

| File | Type | Description |
|---|---|---|
| `forest_t1/t2/t3.tif` | Byte | Binary forest cover at each timestamp (1=forest, 0=non-forest) |
| `fcc12.tif` | Byte | Forest cover change t1→t2: 1=stayed forest, 0=deforested, 255=not forest at t1 |
| `fcc23.tif` | Byte | Forest cover change t2→t3: 1=stayed forest, 0=deforested, 255=not forest at t2 — **the model training response** |
| `fcc123.tif` | Byte | Trajectory: 0=never forest, 1=deforested t1→t2, 2=deforested t2→t3, 3=still forest |
| `altitude.tif` | Float32 | SRTM elevation in metres |
| `slope.tif` | Float32 | Slope in degrees |
| `dist_edge.tif` | Float32 | Distance to forest edge at t2 (metres) |
| `dist_defor.tif` | Float32 | Distance to past deforestation (fcc12, metres) |
| `dist_road.tif` | Float32 | Distance to nearest road (metres) |
| `dist_river.tif` | Float32 | Distance to nearest river (metres) |
| `dist_town.tif` | Float32 | Distance to nearest town (metres) |
| `dist_mill.tif` | Float32 | Distance to nearest palm oil mill (metres) |
| `lq_mp.tif` | Float32 | Location Quotient: mill density relative to plantation density |
| `lq_pm.tif` | Float32 | Location Quotient: plantation density relative to mill density |
| `lq_sq.tif` | Byte | 5-zone LQ classification (1=very low to 5=very high) |
| `mill_kde.tif` | Float32 | Gaussian KDE surface of mill locations |
| `hgu.tif` | Byte | Rasterized HGU polygon layer |
| `peatland.tif` | Byte / Float32 | Rasterized peatland (binary=presence, continuous=depth in m) |
| `M.tif` | Float32 | Mill density surface (input to LQ and SLX) |
| `P.tif` | Float32 | Plantation density surface (input to LQ and SLX) |

### Prediction rasters (in `output/predictions/`)

| File | Description |
|---|---|
| `risk_A.tif` — `risk_G.tif` | Continuous annual deforestation probability (0–1) for each fitted variant |
| `forest_future_A.tif` etc. | Projected binary forest cover at `projection_year` — only produced when `project_future: true` |

### Diagnostics (in `output/diagnostics/`)

| File | Description |
|---|---|
| `dic_table.csv` | DIC (Deviance Information Criterion) for all fitted variants, sorted lowest-first. Lower DIC = better fit. Use this to select the best variant. |
| `moran.json` | Moran's I statistic on ICAR spatial residuals. Values near 0 indicate the ICAR term has absorbed spatial autocorrelation. |

### Causal direction test (in `output/correlation/`)

| File | Description |
|---|---|
| `slx_results.json` | Forward (P ~ M + WM) and reverse (M ~ P + WP) SLX model results with R², coefficients, and direction finding |
| `slx_report.txt` | Human-readable summary — read this first. Tells you whether to use `lq_direction: mp` or `pm` in your config. |

### GWR outputs (in `output/gwr/`, only when `run_gwr: true`)

| File | Description |
|---|---|
| `gwr_coefficients.csv` | Local regression coefficients at each sample point (one row per point, one column per covariate) |
| `gwr_summary.json` | Global summary: bandwidth, AICc, R², sample size |

---

## 9. Model variants

Variants build on each other progressively. Start with A and B, then add complexity if DIC improves.

| Variant | Covariates added | When to use |
|---|---|---|
| **A** | Baseline only (altitude, slope, dist_defor, dist_edge, dist_road, dist_town, dist_river) | Always run as reference |
| **B** | + HGU, peatland | When HGU and peatland files are available |
| **C** | + dist_mill, dist_plantation, mill_kde, plantation_surface | When palm-sector variables are the focus |
| **D** | B + C | Full covariate set without LQ |
| **E** | B + C + LQ | Recommended baseline with LQ |
| **F** | E + LQ² | When LQ has a non-linear relationship with risk |
| **G** | F + LQ×HGU, LQ×peatland interactions | When SLX report shows strong policy-sector interaction |

**Recommended starting set:** `variants: [A, B, E, F]`

---

## 10. UTM zone reference (Indonesia)

Pick the UTM zone that covers most of your study area. Use the southern hemisphere (S) EPSG code unless the centroid is north of the equator.

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

**Examples:** Kalimantan Tengah → EPSG:32749 (108–114 °E, south). Sumatra (northern tip) → EPSG:32647.

---

## 11. Troubleshooting

**`ee.Initialize()` fails**
Run `earthengine authenticate` and ensure your GCP project has the Earth Engine API enabled at [console.cloud.google.com](https://console.cloud.google.com).

**OSM download times out**
Increase `variables.osm_timeout` (e.g. `300`). For very large AOIs, the query can take several minutes.

**`proj.db` version conflict on Windows (PostGIS installed)**
This is handled automatically. If you see PROJ errors anyway, ensure `pytest-env` is installed: `pip install pytest-env`.

**`forestatrisk` can't find rasters**
All aligned rasters must be in the flat `data/` folder of the run (not in subdirectories). The pipeline handles this — only check if you're moving files manually.

**DIC table shows `None`**
The fitted model object did not expose a `.DIC` attribute. This usually means the MCMC chain was too short (`burnin` + `mcmc` < 500 total). Increase both to at least 500.

**Run takes too long**
- Reduce `mcmc` and `burnin` for a quick test (e.g. `burnin: 100, mcmc: 100`)
- Reduce the AOI size
- Use `variants: [A]` first to check the pipeline end-to-end
- Set `run_gwr: false` (GWR bandwidth selection is slow)
