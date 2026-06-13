# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code
in this repository. It is the single, canonical project instruction file (the former
root `CLAUDE.md` was merged into this one).

## Repository layout (post-reorganization)

The root is intentionally minimal. All pipeline code, configs, and run artifacts live
under `active/`.

```
<repo root>/
├── README.md                  # user-facing setup & reference (kept at root)
├── pyproject.toml             # packaging + pytest config (kept at root)
├── .gitignore
├── .claude/                   # THIS file + skills/ (and any agents/deps)
│   ├── CLAUDE.md              # ← you are here (project rules + architecture)
│   └── skills/               # pipeline-run, pipeline-check, model-chat
├── docs/                      # specs, plans, and WORKFLOW.md (authoritative methodology)
│   ├── WORKFLOW.md
│   └── superpowers/{specs,plans}/
├── notes/                     # analytical notes & design decisions
└── active/                    # ALL pipeline code + data + run outputs
    ├── palmdef_risk/         # the Python package
    ├── run.py                # CLI runner (papermill orchestration)
    ├── notebooks/            # 01_download, 02_process, 03_model
    ├── notebook_outputs/     # executed notebook copies
    ├── configs/              # template.yaml, schema.json, example run configs
    ├── tests/                # pytest suite
    ├── data/                 # aligned rasters + user inputs (gitignored content)
    ├── runs/                 # timestamped run folders (gitignored)
    └── cache/                # cross-run cache (gitignored)
```

**Working-directory convention:** the pipeline runs with **CWD = `active/`**, so the
relative paths in configs (`data/...`), `run.py` (`runs/`), and the cache (`cache/`)
all resolve under `active/`. Always `cd active` before running the pipeline.

## Authoritative references (read these first)

- **`docs/WORKFLOW.md`** — the authoritative end-to-end methodology. Every stage,
  formula, encoding, and design decision is spelled out there. When code and
  `WORKFLOW.md` disagree, `WORKFLOW.md` wins.
- **`README.md`** (root) — user-facing setup, config reference, and output layout.
- **`docs/superpowers/specs/2026-05-19-palmdef-risk-refactor-design.md`** — design spec.
- `docs/superpowers/plans/` — the two-part implementation plan the refactor followed.

## Package
- Package name: `palmdef_risk` (not `palmoil_risk`), located at `active/palmdef_risk/`.

## Critical naming rule
WDPA protected areas must always be called `protected`:
- file: `protected.gpkg` / `protected.tif`
- formula term: `protected`
NEVER use `pa` or `pa_status` — causes patsy formula parsing errors in iCAR fitting.

## Data conventions (must not change)
| Type | NoData | dtype |
|---|---|---|
| Forest/categorical (Byte) | 255 | Byte |
| Float rasters | -9999.0 | Float32 |
| Risk output | 0 | UInt16 (0=NoData, 1–65535=probability) |
- All rasters must be in UTM (metres) after Stage 1. Never process in EPSG:4326.
- FCC encoding: 1=remained forest, 0=deforested, 255=NoData. Never change.
- `fcc23.tif` is always the model training response.

## iCAR model rules
- Models: A (biophysical), B (+gravity_resid), C (+plantation_resid),
  D (+gravity_resid +plantation_resid), E (+both +HGU spline).
- No variants beyond A–E exist.
- `beta_start=-99` initialises MCMC from logistic MLE (required).
- Never pickle `patsy.DesignInfo` — always rebuild from `sample.csv` at predict time.
- `dist_mill` is NOT a covariate — mill proximity is represented by `gravity_resid` only.
- `dist_plantation_edge` is NOT entered directly; plantation proximity is the
  orthogonalized residual `plantation_resid` (variants C/D/E). Already log-space —
  never re-log it.

## Mill data rules
- Source: Trase only (`https://resources.trase.earth/data/facilities-data/IDN_PO_mills_clean.geo.json`)
- Filter: `earliest_year_of_existence <= t_year OR null` (conservative — nulls included)
- Cache stores AOI-unfiltered (Indonesia-wide) data; AOI clipping at runtime.

## Plantation data rules
- Two sources via `user_inputs.plantation.source`: `user` (default — supply
  `plantation.t2`/`t3` rasters) or `download` (Descals Global Oil Palm, Zenodo
  `10.5281/zenodo.13379129`).
- `download` writes `plantation_t2.tif` / `plantation_t3.tif` into the run's
  `variables/` folder (same as GHSL); `user` files live in `user_inputs/`. align
  discovers plantation from both.
- Descals classes match config: `[1]=industrial`, `[2]=smallholder`.
- **Accumulation**: plantation at year Y = all pixels with `1990 ≤ YoP ≤ Y` (cumulative
  extent), class taken from the OP-extent layer. Year cutoffs come from `forest.years`
  (t2=`years[1]`, t3=`years[2]`).
- Dataset covers **1990–2021**; a requested year > 2021 is clamped to 2021 with a
  caution notice.
- `download` requires `forest.years` to have all three entries `[t1, t2, t3]`.

## Cache validity rules
- Mill: existence check only (keyed by `{t2_year}_{t3_year}`)
- Forest / Variables: spatial coverage check — `cached_extent ⊇ new_aoi + buffer`
  (variables key now includes `plantation_source` so download/user don't collide)
- Plantation (Descals): global, AOI-independent cache at `cache/plantation_global/`
  (`extent.vrt` + `yop.vrt` + `metadata.json`). Existence check only; downloaded once,
  reused across all runs. Availability checked at Stage 1 only when `source=download`.

## Gravity implementation
- `A_i = Σ_m exp(-d²(i,m)/2σ²)` as `scipy.ndimage.gaussian_filter` on mill density raster.
- NOT a per-pixel loop.
- `gravity_resid = A_i - OLS(A_i ~ dist_road + dist_town)` — residual is the covariate.

## HGU signed distance
- Formula: `dist_from_inside_seeds - dist_from_outside_seeds`
- Negative inside concessions, positive outside, zero at boundary. Units: metres.
- Spline knots at −5000 m, 0 m, +5000 m in model E formula.

## Run context paths
- `ctx.raw_dir` = `<run>/data/raw/`
- `ctx.data_dir` = `<run>/data/`
- `ctx.output_dir` = `<run>/output/`

## Environment & commands

```powershell
# Activate the conda env (heavy geospatial deps: gdal, forestatrisk,
# earthengine-api, osmnx — NOT installable via pip).
conda activate palmdef-risk      # per README

# Install the package in dev mode. pyproject.toml stays at the repo ROOT;
# packages.find points at active/, so install from the root:
python -m pip install -e .

# Register Jupyter kernel (required once after env setup or rename).
python -m ipykernel install --user --name palmdef-risk --display-name "palmdef-risk"

# Run the whole pipeline (executes all 3 notebooks via papermill).
cd active
python run.py --config configs/my_run.yaml

# Run one stage only; reuse an existing run folder for later stages.
python run.py --config configs/my_run.yaml --notebook 01_download
python run.py --config configs/my_run.yaml --notebook 02_process --run-dir runs/<run_dir>
python run.py --config configs/my_run.yaml --notebook 03_model  --run-dir runs/<run_dir>

# Validate config + preview the run folder without executing anything.
python run.py --config configs/my_run.yaml --dry-run

# Tests (testpaths is pinned to active/tests in pyproject; run from root).
python -m pytest                                  # full suite
python -m pytest active/tests/model/test_icar.py  # one file
python -m pytest active/tests/model/test_icar.py::test_x  # one test
python -m pytest -k gravity                        # by keyword
```

> **Use `python -m pip` / `python -m pytest`, not bare `pip` / `pytest` (Windows).** On
> Windows, the system PATH can shadow the conda env's executables with a different Python
> installation's. `python -m` forces the active env's interpreter and avoids silent
> failures.

> **`forestatrisk` cannot be pip-installed on Windows** — its C extension (`binomial_iCAR.c`)
> requires headers unavailable to MSVC. Install via conda-forge:
> `conda install -c conda-forge forestatrisk -y`. If that fails, clone a working env:
> `conda create --name palmdef-risk --clone <source-env>`.

> **After moving the package** (any reorg), the editable install's path pointer goes
> stale — re-run `python -m pip install -e .` from the repo root **inside the activated
> conda env** before importing `palmdef_risk` or running the pipeline.

> **PROJ / proj.db conflict (Windows + PostGIS).** PostgreSQL ships a stale `proj.db`
> that shadows the conda one and makes `GetSpatialRef()` return `None`. Fixed portably in two
> places that must stay in sync: `active/tests/conftest.py` (top-of-file, before any osgeo
> import) and `active/palmdef_risk/io/run.py::_fix_proj_path()` — both derive the conda PROJ
> dir from the active interpreter (`sys.executable` / `sys.prefix`), no hardcoded path. The
> override MUST happen before `osgeo` is imported — never move it later. (`pyproject.toml`
> previously hardcoded one user's absolute path; removed — broke non-`musli` machines / CI.)

## Architecture — the big picture

A **config-driven, notebook-orchestrated geospatial ML pipeline**: one YAML config →
one immutable timestamped run folder → three sequential, fully-resumable stages.

### Run model (the spine)
- A run is one `RunConfig` dataclass (`io/config.py`), parsed and validated from YAML.
  `crs: null` triggers UTM auto-detection from the AOI centroid at `create_run()` time.
- `io/run.py::create_run()` materialises `runs/{project}_{area}_{task}_{timestamp}/`,
  freezes a copy of the config into it, and returns a `RunContext`.
- **`RunContext` is passed to nearly every function.** Functions resolve all paths from
  `ctx.raw_dir` / `ctx.data_dir` / `ctx.output_dir` — a run never writes outside itself.
- **Resumability is a core invariant.** Every notebook cell is skip-if-done guarded.

### The three stages (one notebook each, thin wrappers over `palmdef_risk/`)
1. **Download (`01_download.ipynb` → `data/`)** — forest cover (`data/forest.py`, GEE:
   TMF/GFC), variables (`data/variables.py`: SRTM, WDPA, OSM/GHSL), mills
   (`data/mill.py`: Trase), user inputs (`data/user_inputs.py`). GEE downloads tiled,
   retried, mosaicked via GDAL VRT; multi-UTM AOIs warped into the primary zone.
2. **Process (`02_process.ipynb` → aligned `*.tif`)** — co-registers everything to
   `forest_t2.tif` (`process/align.py`), computes distance transforms
   (`process/distances.py`; `values=0` for edge/deforestation, `values=1` for presence),
   and the Gaussian mill-accessibility surface + residual (`process/gravity.py`).
3. **Model (`03_model.ipynb` → `output/`)** — samples pixels into `sample.csv`, fits
   nested iCAR models A/B/C (`model/icar.py`), predicts risk rasters
   (`model/predict.py`), writes diagnostics (`model/{diagnostics,sensitivity,reports}.py`).
   Formulas are assembled dynamically (`build_formula`) so missing optional layers
   degrade gracefully.

### Cross-cutting: parallelism (`parallel.py`)
- Process-based (`run_parallel` + `adaptive_workers`) for CPU-bound stages; worker count
  = `min(RAM/ram_per_task, cpu_count × cpu_fraction)`. Workers are module-level so
  `ProcessPoolExecutor` can pickle them; data crosses as run-dir paths or JSON, never
  live objects. Thread-based pool for I/O-bound GEE downloads.

## Package layout (`active/palmdef_risk/`)

| Subpackage | Responsibility |
|---|---|
| `io/` | `config.py` (RunConfig + validation), `run.py` (RunContext, create/resume/load, PROJ fix), `helpers.py` |
| `data/` | Stage-1 downloaders: `forest.py`, `variables.py`, `mill.py`, `user_inputs.py`, `utm.py` |
| `process/` | Stage-2: `align.py`, `distances.py`, `gravity.py` |
| `model/` | Stage-3: `icar.py`, `predict.py`, `diagnostics.py`, `sensitivity.py`, `reports.py` |
| `cache.py` | Cross-run cache (see Cache validity rules above) |
| `parallel.py` | Adaptive process/thread parallel executors |

`active/tests/` mirrors this structure. `conftest.py` provides synthetic raster/vector
fixtures in UTM 50S (EPSG:32750) so the suite runs without network or real GEE/OSM.

## Project-local skills (`.claude/skills/`)
- `/pipeline-run <config>` — run the full pipeline (wraps `python active/run.py`).
- `/pipeline-check <run_dir>` — report per-stage completion of a run folder.

## Conventions that bite if ignored
- **Run from `active/`** — the pipeline assumes CWD = `active/` for relative paths.
- **Never process in EPSG:4326.** All rasters are UTM (metres) after Stage 1.
- **Protected areas are always `protected`** (never `pa`/`pa_status`) — breaks patsy.
- **`fcc23.tif` is always the training response**; FCC encoding is fixed.
- **Only model variants A–E exist.**
- **Never pickle `patsy.DesignInfo`** — rebuilt from `sample.csv` at predict time.
