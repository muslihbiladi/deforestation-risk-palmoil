# palmdef_risk — Project Rules for Claude Code

## Package
- Package name: `palmdef_risk` (not `palmoil_risk`)
- Methodology reference: `WORKFLOW.md` (authoritative)
- Spec: `docs/superpowers/specs/2026-05-19-palmdef-risk-refactor-design.md`

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
- Models: A (biophysical), B (+gravity_resid), C (+HGU spline)
- No model variants D–G exist in this codebase.
- `beta_start=-99` initialises MCMC from logistic MLE (required).
- Never pickle `patsy.DesignInfo` — always rebuild from `sample.csv` at predict time.
- `Vbeta > 100` risks divergent MCMC chain under spatial confounding — warn the user.
- `dist_mill` is NOT a covariate — mill proximity is represented by `gravity_resid` only.
- `dist_plantation_edge` is computed but NOT entered into any model formula.

## Mill data rules
- Source: Trase only (`https://trase.earth/open-data/datasets/indonesia-palm-oil-mills/download?format=geojson`)
- Filter: `earliest_year_of_existence <= t_year OR null` (conservative — nulls included)
- Cache stores AOI-unfiltered (Indonesia-wide) data; AOI clipping at runtime.

## Cache validity rules
- Mill: existence check only (keyed by `{t2_year}_{t3_year}`)
- Forest / Variables: spatial coverage check — `cached_extent ⊇ new_aoi + buffer`

## Gravity implementation
- `A_i = Σ_m exp(-d²(i,m)/2σ²)` as `scipy.ndimage.gaussian_filter` on mill density raster.
- NOT a per-pixel loop.
- `gravity_resid = A_i - OLS(A_i ~ dist_road + dist_town)` — residual is the covariate.

## HGU signed distance
- Formula: `dist_from_inside_seeds - dist_from_outside_seeds`
- Negative inside concessions, positive outside, zero at boundary. Units: metres.
- Spline knots at −5000 m, 0 m, +5000 m in model C formula.

## Run context paths
- `ctx.raw_dir` = `<run>/data/raw/`
- `ctx.data_dir` = `<run>/data/`
- `ctx.output_dir` = `<run>/output/`
