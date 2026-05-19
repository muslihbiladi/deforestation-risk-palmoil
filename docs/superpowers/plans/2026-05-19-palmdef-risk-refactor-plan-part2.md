# palmdef_risk Refactor — Implementation Plan Part 2 (Phases 4–6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the process stage (HGU signed-distance, parallel distance computation, gravity accessibility + orthogonalization), model stage (iCAR A/B/C formulas, VIF, Moran's I, gravity bandwidth sensitivity), simplified notebooks, and `.claude` folder updates.

**Prerequisite:** Part 1 must be complete and all its tests passing.

**Spec:** `docs/superpowers/specs/2026-05-19-palmdef-risk-refactor-design.md`  
**Part 1:** `docs/superpowers/plans/2026-05-19-palmdef-risk-refactor-plan-part1.md`

---

## File Map

| Action | Path |
|---|---|
| Delete | `palmdef_risk/process/lq.py`, `palmdef_risk/process/correlation.py`, `palmdef_risk/model/gwr.py` |
| Delete | `tests/process/test_lq.py`, `tests/model/test_gwr.py` |
| Modify | `palmdef_risk/process/align.py` (HGU signed-dist, protected.tif) |
| New | `palmdef_risk/process/distances.py` |
| New | `palmdef_risk/process/gravity.py` |
| Rewrite | `palmdef_risk/model/icar.py` |
| Rewrite | `palmdef_risk/model/diagnostics.py` |
| New | `palmdef_risk/model/sensitivity.py` |
| Minor | `palmdef_risk/model/predict.py` |
| Simplify | `notebooks/01_download.ipynb`, `02_process.ipynb`, `03_model.ipynb` |
| New | `.claude/CLAUDE.md` |
| New | `.claude/skills/pipeline-run.md` |
| New | `.claude/skills/pipeline-check.md` |

---

## Task 10 — `align.py`: HGU signed-distance + `protected.tif` rename

**Files:**
- Modify: `palmdef_risk/process/align.py`
- Modify: `tests/process/test_align.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/process/test_align.py  — add these tests
import numpy as np
import pytest
from osgeo import gdal, ogr, osr
from pathlib import Path


def _write_hgu_gpkg(path: Path, epsg: int = 32750) -> Path:
    """Single polygon HGU covering central 10x10 pixels of a 30x30 raster."""
    driver = ogr.GetDriverByName("GPKG")
    if path.exists():
        driver.DeleteDataSource(str(path))
    ds = driver.CreateDataSource(str(path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    layer = ds.CreateLayer("hgu", srs, ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    # polygon: x=501000-502000, y=9001000-9002000 (UTM 50S)
    for pt in [(501000, 9001000), (502000, 9001000), (502000, 9002000),
               (501000, 9002000), (501000, 9001000)]:
        ring.AddPoint(*pt)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(poly)
    layer.CreateFeature(feat)
    ds = None
    return path


def test_hgu_signed_distance_negative_inside(tmp_path, write_raster):
    """Pixels inside HGU polygon must have negative signed distance."""
    from palmdef_risk.process.align import compute_hgu_signed_distance
    # 30x30 raster, 100m pixels, origin (500000, 9003000)
    ref_arr = np.ones((30, 30), dtype=np.uint8)
    ref = write_raster(tmp_path / "ref.tif", ref_arr,
                       gt=[500000, 100, 0, 9003000, 0, -100], epsg=32750)
    hgu = _write_hgu_gpkg(tmp_path / "hgu.gpkg")
    out = tmp_path / "hgu_signed_dist.tif"
    compute_hgu_signed_distance(str(hgu), str(ref), str(out))
    ds = gdal.Open(str(out))
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    # pixel (15, 15) is inside polygon → negative
    assert arr[15, 15] < 0
    # pixel (0, 0) is outside polygon → positive
    assert arr[0, 0] > 0


def test_hgu_signed_distance_zero_at_boundary(tmp_path, write_raster):
    from palmdef_risk.process.align import compute_hgu_signed_distance
    ref_arr = np.ones((30, 30), dtype=np.uint8)
    ref = write_raster(tmp_path / "ref.tif", ref_arr,
                       gt=[500000, 100, 0, 9003000, 0, -100], epsg=32750)
    hgu = _write_hgu_gpkg(tmp_path / "hgu.gpkg")
    out = tmp_path / "hgu_signed_dist.tif"
    compute_hgu_signed_distance(str(hgu), str(ref), str(out))
    ds = gdal.Open(str(out))
    arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
    ds = None
    # Boundary pixels should have value near zero (within ±1 pixel = 100m)
    boundary_val = abs(arr[10, 10])   # pixel at edge of polygon
    assert boundary_val < 200.0


def test_protected_tif_written_not_pa(tmp_path, minimal_config_yaml):
    """align_all must write protected.tif, never pa.tif."""
    from palmdef_risk.io.run import create_run
    # Just verify the constant is right; full align_all is an integration test
    from palmdef_risk.process.align import _PROTECTED_FILENAME
    assert _PROTECTED_FILENAME == "protected"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/process/test_align.py::test_hgu_signed_distance_negative_inside -v
pytest tests/process/test_align.py::test_protected_tif_written_not_pa -v
```
Expected: `ImportError` — `compute_hgu_signed_distance` and `_PROTECTED_FILENAME` not found

- [ ] **Step 3: Add to `palmdef_risk/process/align.py`**

Add at module level (near top):
```python
_PROTECTED_FILENAME = "protected"   # never "pa" — causes patsy formula errors
```

Replace all occurrences of `"pa"` used as raster/vector filename with `_PROTECTED_FILENAME`:
```
grep -n '"pa"' palmdef_risk/process/align.py
grep -n "'pa'" palmdef_risk/process/align.py
grep -n "pa.tif" palmdef_risk/process/align.py
```

Add `compute_hgu_signed_distance` function:
```python
def compute_hgu_signed_distance(
    hgu_gpkg: str,
    ref_tif: str,
    out_tif: str,
) -> None:
    """Write signed distance to HGU boundary: negative inside, positive outside."""
    from palmdef_risk.io.helpers import get_mask_properties, rasterize_vector
    import numpy as np
    from osgeo import gdal

    # Step 1: Rasterize HGU → binary mask (1=inside, 0=outside)
    mask_props = get_mask_properties(ref_tif)
    hgu_mask_path = Path(out_tif).parent / "_hgu_mask_tmp.tif"
    rasterize_vector(hgu_gpkg, str(hgu_mask_path), burn_value=1, mask_props=mask_props)

    ds_mask = gdal.Open(str(hgu_mask_path))
    inside = ds_mask.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    gt = ds_mask.GetGeoTransform()
    proj = ds_mask.GetProjection()
    ny, nx = inside.shape
    ds_mask = None

    def _proximity(arr: np.ndarray) -> np.ndarray:
        """GDAL proximity to nearest non-zero pixel (metres, GEO units)."""
        drv = gdal.GetDriverByName("MEM")
        src_ds = drv.Create("", nx, ny, 1, gdal.GDT_Byte)
        src_ds.SetGeoTransform(gt)
        src_ds.SetProjection(proj)
        src_ds.GetRasterBand(1).WriteArray(arr)
        out_ds = drv.Create("", nx, ny, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        gdal.ComputeProximity(
            src_ds.GetRasterBand(1),
            out_ds.GetRasterBand(1),
            options=["DISTUNITS=GEO"],
        )
        result = out_ds.GetRasterBand(1).ReadAsArray()
        return result

    # dist_from_inside: 0 inside HGU, >0 outside
    dist_from_inside = _proximity(inside)
    # dist_from_outside: 0 outside HGU, >0 inside
    outside = (1 - inside).astype(np.uint8)
    dist_from_outside = _proximity(outside)

    # signed = dist_from_inside - dist_from_outside
    # inside: 0 - (depth) = negative; outside: (dist) - 0 = positive
    signed = (dist_from_inside.astype(np.float32)
              - dist_from_outside.astype(np.float32))

    # Write output as Float32 raster
    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(
        str(out_tif), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(signed)
    out_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    out_ds = None
    Path(str(hgu_mask_path)).unlink(missing_ok=True)
```

Also replace the existing binary HGU rasterization block inside `align_all()` with a call to `compute_hgu_signed_distance`:
```python
# Find: rasterize_vector(hgu_path, ...) → hgu.tif
# Replace with:
compute_hgu_signed_distance(hgu_gpkg=str(inputs["hgu"]),
                             ref_tif=str(forest_t2),
                             out_tif=str(ctx.data_dir / "hgu_signed_dist.tif"))
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/process/test_align.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/process/align.py tests/process/test_align.py
git commit -m "feat: HGU signed-distance in align.py, rename pa→protected"
```

---

## Task 11 — `process/distances.py` (NEW — parallel distance computation)

**Files:**
- Create: `palmdef_risk/process/distances.py`
- Create: `tests/process/test_distances.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/process/test_distances.py
import numpy as np
import pytest
from pathlib import Path
from osgeo import gdal


def _make_ctx(tmp_path, write_raster, write_vector, minimal_config_yaml):
    from palmdef_risk.io.run import create_run
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    # Write minimal input rasters into data_dir
    arr = np.ones((10, 10), dtype=np.uint8)
    arr[5, 5] = 0
    gt = [500000, 30, 0, 9000300, 0, -30]
    d = ctx.data_dir
    d.mkdir(parents=True, exist_ok=True)
    write_raster(d / "forest_t2.tif", arr, gt, 32750, nodata=255)
    write_raster(d / "fcc12.tif", arr, gt, 32750, nodata=255)
    write_raster(d / "forest_t3.tif", arr, gt, 32750, nodata=255)
    write_raster(d / "fcc23.tif", arr, gt, 32750, nodata=255)
    # Vectors
    write_vector(d / "road.gpkg", epsg=32750)
    write_vector(d / "river.gpkg", epsg=32750)
    write_vector(d / "town.gpkg", epsg=32750)
    # plantation raster
    write_raster(d / "plantation.tif", arr, gt, 32750, nodata=255)
    return ctx


def test_compute_all_distances_creates_expected_files(
    tmp_path, write_raster, write_vector, minimal_config_yaml
):
    from palmdef_risk.process.distances import compute_all_distances
    ctx = _make_ctx(tmp_path, write_raster, write_vector, minimal_config_yaml)
    compute_all_distances(ctx)
    d = ctx.data_dir
    expected = [
        "dist_edge.tif", "dist_defor.tif",
        "dist_road.tif", "dist_river.tif", "dist_town.tif",
        "dist_plantation_edge.tif",
        "dist_edge_forecast.tif", "dist_defor_forecast.tif",
    ]
    for name in expected:
        assert (d / name).exists(), f"Missing: {name}"


def test_dist_mill_not_created(
    tmp_path, write_raster, write_vector, minimal_config_yaml
):
    from palmdef_risk.process.distances import compute_all_distances
    ctx = _make_ctx(tmp_path, write_raster, write_vector, minimal_config_yaml)
    compute_all_distances(ctx)
    assert not (ctx.data_dir / "dist_mill.tif").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/process/test_distances.py -v
```
Expected: `ModuleNotFoundError: No module named 'palmdef_risk.process.distances'`

- [ ] **Step 3: Create `palmdef_risk/process/distances.py`**

```python
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from osgeo import gdal, ogr

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)


def _proximity_from_raster(src_path: Path, out_path: Path, target_value: int = 0) -> None:
    """Compute GDAL proximity (metres) to pixels where value == target_value."""
    ds = gdal.Open(str(src_path))
    arr = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ny, nx = arr.shape
    ds = None

    # Binary mask: 1 where target, 0 elsewhere
    mask = (arr == target_value).astype(np.uint8)

    drv = gdal.GetDriverByName("MEM")
    src_ds = drv.Create("", nx, ny, 1, gdal.GDT_Byte)
    src_ds.SetGeoTransform(gt)
    src_ds.SetProjection(proj)
    src_ds.GetRasterBand(1).WriteArray(mask)

    out_ds = gdal.GetDriverByName("GTiff").Create(
        str(out_path), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    gdal.ComputeProximity(src_ds.GetRasterBand(1), out_ds.GetRasterBand(1),
                          options=["DISTUNITS=GEO"])
    out_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    out_ds = None


def _proximity_from_vector(vec_path: Path, ref_path: Path, out_path: Path) -> None:
    """Rasterize vector, then compute proximity from burned pixels."""
    from palmdef_risk.io.helpers import get_mask_properties, rasterize_vector
    mask_props = get_mask_properties(str(ref_path))
    tmp = out_path.parent / f"_vec_tmp_{out_path.stem}.tif"
    rasterize_vector(str(vec_path), str(tmp), burn_value=1, mask_props=mask_props)
    _proximity_from_raster(tmp, out_path, target_value=1)
    tmp.unlink(missing_ok=True)


def compute_all_distances(ctx: "RunContext") -> None:
    """Compute all distance rasters (metres). dist_mill is NOT computed."""
    d = ctx.data_dir

    tasks = []

    # Raster-based distances
    for name, src, tgt in [
        ("dist_edge",           "forest_t2.tif",  0),
        ("dist_defor",          "fcc12.tif",       0),
        ("dist_edge_forecast",  "forest_t3.tif",  0),
        ("dist_defor_forecast", "fcc23.tif",       0),
    ]:
        src_path = d / src
        out_path = d / f"{name}.tif"
        if src_path.exists() and not out_path.exists():
            tasks.append(("raster", src_path, out_path, tgt))

    # Vector-based distances
    for name, vec in [
        ("dist_road",             "road.gpkg"),
        ("dist_river",            "river.gpkg"),
        ("dist_town",             "town.gpkg"),
        ("dist_plantation_edge",  "plantation.tif"),
    ]:
        src_path = d / vec
        out_path = d / f"{name}.tif"
        if src_path.exists() and not out_path.exists():
            if src_path.suffix == ".gpkg":
                tasks.append(("vector", src_path, out_path, d / "forest_t2.tif"))
            else:
                tasks.append(("raster", src_path, out_path, 1))

    from palmdef_risk.parallel import run_parallel
    run_parallel(_dist_worker, tasks,
                 ram_per_task_gb=ctx.config.ram_per_dist_gb, cfg=ctx.config)
    logger.info("All distance rasters computed")


def _dist_worker(task: tuple) -> None:
    kind, src, out, extra = task
    if kind == "raster":
        _proximity_from_raster(src, out, target_value=extra)
    else:
        _proximity_from_vector(src, extra, out)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/process/test_distances.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/process/distances.py tests/process/test_distances.py
git commit -m "feat: add distances.py — parallel distance computation, no dist_mill"
```

---

## Task 12 — `process/gravity.py` (NEW)

**Files:**
- Create: `palmdef_risk/process/gravity.py`
- Create: `tests/process/test_gravity.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/process/test_gravity.py
import numpy as np
import pytest
from osgeo import gdal


def test_gaussian_filter_higher_near_source(tmp_path, write_raster):
    from palmdef_risk.process.gravity import _apply_gaussian_filter
    # 50x50 raster, 100m pixels; single mill at centre
    arr = np.zeros((50, 50), dtype=np.uint8)
    arr[25, 25] = 1
    ref = write_raster(tmp_path / "mills.tif", arr,
                       gt=[500000, 100, 0, 9005000, 0, -100], epsg=32750)
    out = tmp_path / "gravity_raw.tif"
    _apply_gaussian_filter(ref, out, sigma_km=0.5, radius_km=2.0)
    ds = gdal.Open(str(out))
    result = ds.GetRasterBand(1).ReadAsArray().astype(float)
    ds = None
    assert result[25, 25] == result.max()
    assert result[0, 0] < result[25, 25] * 0.01


def test_orthogonalize_produces_residual_raster(tmp_path, write_raster):
    """orthogonalize_gravity must write gravity_resid.tif."""
    from palmdef_risk.process.gravity import orthogonalize_gravity
    rng = np.random.default_rng(42)
    gt = [500000, 100, 0, 9005000, 0, -100]
    gravity = rng.uniform(0, 1, (20, 20)).astype(np.float32)
    road = rng.uniform(0, 5000, (20, 20)).astype(np.float32)
    town = rng.uniform(0, 20000, (20, 20)).astype(np.float32)
    g_path = write_raster(tmp_path / "gravity_raw.tif", gravity, gt, 32750,
                          dtype=gdal.GDT_Float32, nodata=-9999.0)
    r_path = write_raster(tmp_path / "dist_road.tif", road, gt, 32750,
                          dtype=gdal.GDT_Float32, nodata=-9999.0)
    t_path = write_raster(tmp_path / "dist_town.tif", town, gt, 32750,
                          dtype=gdal.GDT_Float32, nodata=-9999.0)
    out = tmp_path / "gravity_resid.tif"
    orthogonalize_gravity(g_path, r_path, t_path, out)
    assert out.exists()
    ds = gdal.Open(str(out))
    resid = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    # Residual should have near-zero mean
    valid = resid[resid != -9999.0]
    assert abs(valid.mean()) < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/process/test_gravity.py -v
```
Expected: `ModuleNotFoundError: No module named 'palmdef_risk.process.gravity'`

- [ ] **Step 3: Create `palmdef_risk/process/gravity.py`**

```python
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from osgeo import gdal
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)


def _apply_gaussian_filter(
    mill_raster: Path | str,
    out_path: Path | str,
    sigma_km: float,
    radius_km: float,
) -> None:
    """
    Gaussian kernel accessibility: A_i = Σ_m exp(-d²(i,m) / 2σ²).
    Implemented as scipy gaussian_filter on mill density raster.
    """
    ds = gdal.Open(str(mill_raster))
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    pixel_size_m = abs(gt[1])
    ds = None

    sigma_px = (sigma_km * 1000.0) / pixel_size_m
    truncate = radius_km * 1000.0 / (sigma_km * 1000.0)

    result = gaussian_filter(arr.astype(float), sigma=sigma_px,
                             truncate=truncate).astype(np.float32)

    ny, nx = result.shape
    out_ds = gdal.GetDriverByName("GTiff").Create(
        str(out_path), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(result)
    out_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    out_ds = None


def orthogonalize_gravity(
    gravity_path: Path | str,
    dist_road_path: Path | str,
    dist_town_path: Path | str,
    out_path: Path | str,
) -> float:
    """
    OLS: A_i ~ dist_road + dist_town. Residual → gravity_resid.tif.
    Returns R² of the regression. Warns if R² > 0.85.
    """
    def _load_flat(p):
        ds = gdal.Open(str(p))
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        nd = ds.GetRasterBand(1).GetNoDataValue()
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        shape = arr.shape
        ds = None
        return arr, nd, gt, proj, shape

    g_arr, g_nd, gt, proj, shape = _load_flat(gravity_path)
    r_arr, r_nd, *_ = _load_flat(dist_road_path)
    t_arr, t_nd, *_ = _load_flat(dist_town_path)

    # Valid mask
    mask = (g_arr != g_nd) & (r_arr != r_nd) & (t_arr != t_nd)
    g = g_arr[mask]
    r = r_arr[mask]
    t = t_arr[mask]

    # OLS: [1, dist_road, dist_town] → g
    X = np.column_stack([np.ones(len(g)), r, t])
    beta, *_ = np.linalg.lstsq(X, g, rcond=None)
    g_hat = X @ beta
    residual_flat = g - g_hat

    ss_res = np.sum(residual_flat ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if r2 > 0.85:
        logger.warning(
            "Gravity R²=%.3f > 0.85: accessibility is largely collinear with "
            "infrastructure — Model B marginal signal may be weak.", r2
        )

    # Write residual raster
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
    logger.info("Gravity orthogonalized R²=%.3f; residual → %s", r2, out_path)
    return r2


def compute_gravity_accessibility(ctx: "RunContext") -> Path:
    """Rasterize mill_t2.gpkg, apply Gaussian filter → data/gravity_raw.tif."""
    from palmdef_risk.io.helpers import get_mask_properties, rasterize_vector
    d = ctx.data_dir
    ref = d / "forest_t2.tif"
    mill_gpkg = ctx.raw_dir / "mill" / "mill_t2.gpkg"

    mask_props = get_mask_properties(str(ref))
    mill_raster = d / "_mill_density_tmp.tif"
    rasterize_vector(str(mill_gpkg), str(mill_raster), burn_value=1,
                     mask_props=mask_props)

    out = d / "gravity_raw.tif"
    _apply_gaussian_filter(mill_raster, out,
                           sigma_km=ctx.config.sigma_km,
                           radius_km=ctx.config.radius_km)
    mill_raster.unlink(missing_ok=True)
    return out


def orthogonalize_gravity_ctx(ctx: "RunContext") -> Path:
    """Run orthogonalize_gravity using run context paths."""
    d = ctx.data_dir
    out = d / "gravity_resid.tif"
    orthogonalize_gravity(d / "gravity_raw.tif", d / "dist_road.tif",
                          d / "dist_town.tif", out)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/process/test_gravity.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/process/gravity.py tests/process/test_gravity.py
git commit -m "feat: add gravity.py — Gaussian filter accessibility + OLS orthogonalization"
```

---

## Task 13 — `model/icar.py` rewrite (A/B/C formulas)

**Files:**
- Rewrite: `palmdef_risk/model/icar.py`
- Rewrite: `tests/model/test_icar.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/model/test_icar.py
import pytest
from unittest.mock import MagicMock
from palmdef_risk.model.icar import build_formula


def _ctx():
    ctx = MagicMock()
    ctx.config.peatland_type = "binary"
    ctx.config.Vbeta = 1000
    return ctx


def test_formula_a_baseline_covariates():
    f = build_formula("A", _ctx())
    assert "scale(altitude)" in f
    assert "scale(slope)" in f
    assert "dist_defor" in f
    assert "dist_edge" in f
    assert "dist_road" in f
    assert "dist_town" in f
    assert "dist_river" in f
    assert "protected" in f
    assert "cell" in f


def test_formula_a_no_gravity():
    f = build_formula("A", _ctx())
    assert "gravity_resid" not in f


def test_formula_b_adds_gravity():
    f = build_formula("B", _ctx())
    assert "scale(gravity_resid)" in f
    assert "hgu_b1" not in f


def test_formula_c_adds_hgu_spline():
    f = build_formula("C", _ctx())
    assert "hgu_b1" in f
    assert "hgu_b2" in f
    assert "scale(gravity_resid)" in f


def test_formula_no_dist_mill_in_any_variant():
    for v in ["A", "B", "C"]:
        assert "dist_mill" not in build_formula(v, _ctx())


def test_formula_no_lq_terms():
    for v in ["A", "B", "C"]:
        f = build_formula(v, _ctx())
        assert "lq" not in f.lower()
        assert "kde" not in f.lower()


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_formula("D", _ctx())


def test_response_lhs():
    f = build_formula("A", _ctx())
    assert f.startswith("I(1 - fcc23) + trial ~")
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/model/test_icar.py -v
```
Expected: multiple failures — old A–G logic, `dist_mill` present

- [ ] **Step 3: Rewrite `palmdef_risk/model/icar.py`**

```python
from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)

_BASELINE_RHS = (
    "scale(altitude)"
    " + scale(slope)"
    " + scale(log(dist_defor + 1))"
    " + scale(log(dist_edge + 1))"
    " + scale(log(dist_road + 1))"
    " + scale(log(dist_town + 1))"
    " + scale(log(dist_river + 1))"
    " + protected"
)

_HGU_SPLINE = "scale(hgu_b1) + scale(hgu_b2)"


def build_formula(variant: str, ctx: "RunContext") -> str:
    """Return the forestatrisk suitab_formula string for model variant A, B, or C."""
    if variant == "A":
        rhs = _BASELINE_RHS
    elif variant == "B":
        rhs = _BASELINE_RHS + " + scale(gravity_resid)"
    elif variant == "C":
        rhs = _BASELINE_RHS + " + scale(gravity_resid) + " + _HGU_SPLINE
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Valid variants: A, B, C")
    return f"I(1 - fcc23) + trial ~ {rhs} + cell"


def _add_hgu_spline_cols(data: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute hgu_b1, hgu_b2 spline basis columns from hgu_signed_dist."""
    if "hgu_signed_dist" not in data.columns:
        return data
    from patsy import dmatrix
    import numpy as np
    x = data["hgu_signed_dist"].values
    dm = dmatrix("cr(x, knots=(-5000, 0, 5000)) - 1", {"x": x}, return_type="matrix")
    dm_arr = np.asarray(dm)
    data = data.copy()
    data["hgu_b1"] = dm_arr[:, 0]
    if dm_arr.shape[1] > 1:
        data["hgu_b2"] = dm_arr[:, 1]
    else:
        data["hgu_b2"] = 0.0
    return data


def prepare_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (HGU spline basis) to the sample DataFrame."""
    data = _add_hgu_spline_cols(data)
    return data


def fit_model(variant: str, ctx: "RunContext") -> Path:
    """Fit one iCAR model variant. Returns path to saved .pkl file."""
    import forestatrisk as far
    formula = build_formula(variant, ctx)
    sample_path = ctx.output_dir / "sample.csv"
    data = pd.read_csv(sample_path)
    data = prepare_sample(data)

    model_dir = ctx.output_dir / "models" / f"model_{variant}"
    model_dir.mkdir(parents=True, exist_ok=True)

    cfg = ctx.config
    mod = far.model_icar(
        suitab_formula=formula,
        data=data,
        n_neighbors=1,
        Vbeta=cfg.Vbeta,
        beta_start=-99,   # initialise from logistic MLE
        burnin=cfg.burnin,
        mcmc=cfg.mcmc,
        thin=cfg.thin,
        seed=cfg.seed,
        save_rho=True,
        verbose=False,
    )

    # Pickle only safe subset — never pickle patsy.DesignInfo
    pkl_path = model_dir / f"mod_{variant}.pkl"
    safe_state = {
        "betas": mod.betas,
        "rho": mod.rho,
        "formula": formula,
        "betas_mcmc": mod.betas_mcmc,
        "deviance": mod.deviance,
        "variant": variant,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(safe_state, f)
    logger.info("Model %s fitted and saved to %s", variant, pkl_path)
    return pkl_path


def fit_all(ctx: "RunContext") -> list[Path]:
    """Fit all configured model variants in parallel."""
    from palmdef_risk.parallel import run_parallel
    variants = ctx.config.model_variants
    tasks = [(v, ctx) for v in variants]
    results = run_parallel(_fit_worker, tasks,
                           ram_per_task_gb=ctx.config.ram_per_icar_gb, cfg=ctx.config)
    return [r for r in results if r is not None]


def _fit_worker(args: tuple) -> Path | None:
    variant, ctx = args
    try:
        return fit_model(variant, ctx)
    except Exception as e:
        logger.error("Model %s failed: %s", variant, e)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/model/test_icar.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/model/icar.py tests/model/test_icar.py
git commit -m "feat: rewrite icar.py — A/B/C formulas only, protected, no dist_mill/LQ"
```

---

## Task 14 — `model/diagnostics.py` rewrite (VIF + Moran's I)

**Files:**
- Rewrite: `palmdef_risk/model/diagnostics.py`
- Rewrite: `tests/model/test_diagnostics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/model/test_diagnostics.py
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def test_compute_vif_writes_json(tmp_path):
    from palmdef_risk.model.diagnostics import compute_vif
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "altitude": rng.normal(0, 1, n),
        "slope": rng.normal(0, 1, n),
        "gravity_resid": rng.normal(0, 1, n),
    })
    sample = tmp_path / "sample.csv"
    df.to_csv(sample, index=False)
    out = tmp_path / "vif.json"
    compute_vif(["altitude", "slope", "gravity_resid"], sample, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "altitude" in data
    # Uncorrelated columns → VIF near 1
    assert data["altitude"] < 3.0


def test_compute_vif_flags_high_vif(tmp_path, caplog):
    import logging
    from palmdef_risk.model.diagnostics import compute_vif
    rng = np.random.default_rng(42)
    n = 200
    base = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "x1": base,
        "x2": base + rng.normal(0, 0.01, n),  # near-duplicate → very high VIF
    })
    sample = tmp_path / "sample.csv"
    df.to_csv(sample, index=False)
    out = tmp_path / "vif.json"
    with caplog.at_level(logging.WARNING, logger="palmdef_risk"):
        compute_vif(["x1", "x2"], sample, out)
    assert any("VIF" in r.message for r in caplog.records)


def test_morans_i_output_has_required_keys(tmp_path):
    """compute_morans_i writes moran.json with I and p_value per variant."""
    from palmdef_risk.model.diagnostics import compute_morans_i
    # Minimal mock: spatial weights from tiny grid, random residuals
    rng = np.random.default_rng(0)
    residuals = {
        "A": rng.normal(0, 1, 100),
        "B": rng.normal(0, 0.5, 100),
    }
    coords = [(i * 30, j * 30) for i in range(10) for j in range(10)]
    out = tmp_path / "moran.json"
    compute_morans_i(residuals, coords, out)
    assert out.exists()
    data = json.loads(out.read_text())
    for v in ["A", "B"]:
        assert v in data
        assert "I" in data[v]
        assert "p_value" in data[v]
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/model/test_diagnostics.py -v
```
Expected: `ImportError` — functions not found

- [ ] **Step 3: Rewrite `palmdef_risk/model/diagnostics.py`**

```python
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_vif(
    covariates: list[str],
    sample_csv: Path | str,
    out_json: Path | str,
) -> dict[str, float]:
    """
    Compute Variance Inflation Factor for each covariate.
    Writes results to out_json. Warns for VIF > 5.
    """
    data = pd.read_csv(sample_csv)[covariates].dropna()
    X = data.values
    vif = {}
    for j, col in enumerate(covariates):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        others = np.column_stack([np.ones(len(y)), others])
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        y_hat = others @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        v = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        vif[col] = round(v, 3)
        if v > 10:
            logger.warning("VIF(%s)=%.1f > 10 (high multicollinearity)", col, v)
        elif v > 5:
            logger.warning("VIF(%s)=%.1f > 5 (moderate multicollinearity)", col, v)

    Path(out_json).write_text(json.dumps(vif, indent=2))
    return vif


def compute_morans_i(
    residuals: dict[str, np.ndarray],
    coords: list[tuple[float, float]],
    out_json: Path | str,
) -> dict:
    """
    Compute Moran's I on deviance residuals for each model variant.
    Uses inverse-distance weights (k=8 neighbours).
    """
    try:
        from libpysal.weights import KNN
        from esda.moran import Moran
    except ImportError:
        logger.warning("libpysal/esda not installed — Moran's I skipped")
        results = {v: {"I": None, "p_value": None, "note": "esda not installed"}
                   for v in residuals}
        Path(out_json).write_text(json.dumps(results, indent=2))
        return results

    import geopandas as gpd
    from shapely.geometry import Point
    pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coords])
    w = KNN.from_dataframe(pts, k=min(8, len(coords) - 1))
    w.transform = "R"

    results = {}
    for variant, resid in residuals.items():
        mi = Moran(resid, w)
        results[variant] = {
            "I": round(float(mi.I), 4),
            "p_value": round(float(mi.p_norm), 4),
        }
        logger.info("Moran's I [%s]: I=%.4f p=%.4f", variant, mi.I, mi.p_norm)

    Path(out_json).write_text(json.dumps(results, indent=2))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/model/test_diagnostics.py -v
```
Expected: PASS (Moran's I test may skip if libpysal not installed)

- [ ] **Step 5: Commit**

```
git add palmdef_risk/model/diagnostics.py tests/model/test_diagnostics.py
git commit -m "feat: rewrite diagnostics.py — VIF + Moran's I"
```

---

## Task 15 — `model/sensitivity.py` (NEW — gravity bandwidth sensitivity)

**Files:**
- Create: `palmdef_risk/model/sensitivity.py`
- Create: `tests/model/test_sensitivity.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/model/test_sensitivity.py
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_sensitivity_json_has_entry_per_sigma(tmp_path, minimal_config_yaml):
    from palmdef_risk.io.run import create_run
    from palmdef_risk.model.sensitivity import run_gravity_sensitivity

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")

    def _fake_compute(ctx_arg, sigma_km):
        return tmp_path / "gravity_raw.tif"

    def _fake_ortho(ctx_arg):
        return tmp_path / "gravity_resid.tif"

    def _fake_fit(variant, ctx_arg):
        return tmp_path / "mod.pkl"

    def _fake_load(pkl):
        m = MagicMock()
        m.betas = [0.1, 0.2, 0.3]
        m.deviance = [100.0]
        return m

    with patch("palmdef_risk.model.sensitivity.compute_gravity_raw", _fake_compute):
        with patch("palmdef_risk.model.sensitivity.orthogonalize_gravity_ctx", _fake_ortho):
            with patch("palmdef_risk.model.sensitivity.fit_model", _fake_fit):
                with patch("palmdef_risk.model.sensitivity._load_model", _fake_load):
                    out = ctx.output_dir / "diagnostics" / "gravity_sensitivity.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    result = run_gravity_sensitivity(ctx)

    assert out.exists()
    data = json.loads(out.read_text())
    # One entry per sigma in config
    assert len(data) == len(ctx.config.sensitivity_sigmas)
    for entry in data:
        assert "sigma_km" in entry
        assert "accessibility_coef" in entry
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/model/test_sensitivity.py -v
```
Expected: `ModuleNotFoundError: No module named 'palmdef_risk.model.sensitivity'`

- [ ] **Step 3: Create `palmdef_risk/model/sensitivity.py`**

```python
from __future__ import annotations
import json
import logging
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)


def compute_gravity_raw(ctx: "RunContext", sigma_km: float) -> Path:
    """Compute gravity_raw.tif at a given sigma (overwrites ctx.data_dir/gravity_raw.tif)."""
    from palmdef_risk.process.gravity import _apply_gaussian_filter
    from palmdef_risk.io.helpers import get_mask_properties, rasterize_vector
    d = ctx.data_dir
    ref = d / "forest_t2.tif"
    mill_gpkg = ctx.raw_dir / "mill" / "mill_t2.gpkg"
    mask_props = get_mask_properties(str(ref))
    tmp = d / "_mill_density_sensitivity_tmp.tif"
    rasterize_vector(str(mill_gpkg), str(tmp), burn_value=1, mask_props=mask_props)
    out = d / "gravity_raw.tif"
    _apply_gaussian_filter(tmp, out, sigma_km=sigma_km, radius_km=ctx.config.radius_km)
    tmp.unlink(missing_ok=True)
    return out


def _load_model(pkl_path: Path) -> object:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def run_gravity_sensitivity(ctx: "RunContext") -> Path:
    """
    For each sigma in config.sensitivity_sigmas: refit Model B, extract
    accessibility coefficient + deviance. Writes gravity_sensitivity.json.
    """
    from palmdef_risk.process.gravity import orthogonalize_gravity_ctx
    from palmdef_risk.model.icar import fit_model, build_formula

    out_dir = ctx.output_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "gravity_sensitivity.json"

    # Backup current gravity files
    d = ctx.data_dir
    backup_raw = d / "_gravity_raw_backup.tif"
    backup_resid = d / "_gravity_resid_backup.tif"
    if (d / "gravity_raw.tif").exists():
        shutil.copy2(d / "gravity_raw.tif", backup_raw)
    if (d / "gravity_resid.tif").exists():
        shutil.copy2(d / "gravity_resid.tif", backup_resid)

    results = []
    for sigma in ctx.config.sensitivity_sigmas:
        logger.info("Gravity sensitivity: sigma=%.0f km", sigma)
        compute_gravity_raw(ctx, sigma_km=sigma)
        orthogonalize_gravity_ctx(ctx)
        pkl = fit_model("B", ctx)
        state = _load_model(pkl)
        formula = build_formula("B", ctx)
        # Accessibility coefficient is the last beta before 'cell'
        # beta index for gravity_resid depends on formula term order
        coef_idx = _gravity_coef_index(formula, state)
        entry = {
            "sigma_km": sigma,
            "accessibility_coef": float(state["betas"][coef_idx])
            if coef_idx is not None else None,
            "mean_deviance": float(np.mean(state["deviance"]))
            if hasattr(state["deviance"], "__len__") else float(state["deviance"]),
        }
        results.append(entry)

    # Restore original gravity files
    if backup_raw.exists():
        shutil.move(str(backup_raw), d / "gravity_raw.tif")
    if backup_resid.exists():
        shutil.move(str(backup_resid), d / "gravity_resid.tif")

    out_json.write_text(json.dumps(results, indent=2))
    logger.info("Gravity sensitivity written to %s", out_json)
    return out_json


def _gravity_coef_index(formula: str, state: dict) -> int | None:
    """Find index of gravity_resid beta in the betas array."""
    try:
        terms = formula.split("~")[1].split("+")
        terms = [t.strip() for t in terms if "cell" not in t]
        for i, t in enumerate(terms):
            if "gravity_resid" in t:
                return i + 1  # +1 for intercept
    except Exception:
        pass
    return None


import numpy as np  # noqa: E402 — needed for mean_deviance calculation above
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/model/test_sensitivity.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/model/sensitivity.py tests/model/test_sensitivity.py
git commit -m "feat: add sensitivity.py — gravity bandwidth sensitivity (σ=15/25/40 km)"
```

---

## Task 16 — `model/predict.py`: Risk raster dtype UInt16

**Files:**
- Modify: `palmdef_risk/model/predict.py`
- Modify: `tests/model/test_predict.py`

- [ ] **Step 1: Write failing test**

```python
# tests/model/test_predict.py — add this test
def test_risk_raster_is_uint16_nodata_zero(tmp_path, write_raster):
    """predict_risk must write UInt16 with NoData=0 (0=NoData, 1-65535=prob)."""
    from palmdef_risk.model.predict import _write_risk_raster
    import numpy as np
    from osgeo import gdal
    arr = np.random.uniform(0, 1, (10, 10)).astype(np.float32)
    ref = write_raster(tmp_path / "ref.tif", np.ones((10, 10), dtype=np.uint8),
                       gt=[500000, 30, 0, 9000300, 0, -30], epsg=32750)
    out = tmp_path / "risk.tif"
    _write_risk_raster(arr, str(ref), str(out))
    ds = gdal.Open(str(out))
    band = ds.GetRasterBand(1)
    assert band.DataType == gdal.GDT_UInt16
    assert band.GetNoDataValue() == 0
    result = band.ReadAsArray()
    # Valid pixels must be in [1, 65535]
    assert result[result > 0].min() >= 1
    ds = None
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/model/test_predict.py::test_risk_raster_is_uint16_nodata_zero -v
```
Expected: `ImportError` or dtype mismatch

- [ ] **Step 3: Add `_write_risk_raster` to `palmdef_risk/model/predict.py`**

```python
def _write_risk_raster(
    prob_arr: np.ndarray,
    ref_tif: str,
    out_tif: str,
) -> None:
    """Write probability [0,1] as UInt16. NoData=0, valid range=[1, 65535]."""
    from osgeo import gdal
    ds = gdal.Open(ref_tif)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ny, nx = prob_arr.shape
    ds = None

    # Scale: 0 → 0 (NoData), 1 → 65535
    scaled = np.round(prob_arr * 65535).astype(np.uint16)
    scaled = np.clip(scaled, 1, 65535)   # ensure valid pixels are ≥ 1
    # pixels that were exactly 0.0 stay 0 (treated as NoData)
    scaled[prob_arr == 0.0] = 0

    out_ds = gdal.GetDriverByName("GTiff").Create(
        out_tif, nx, ny, 1, gdal.GDT_UInt16,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(scaled)
    out_ds.GetRasterBand(1).SetNoDataValue(0)
    out_ds.FlushCache()
    out_ds = None
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/model/test_predict.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/model/predict.py tests/model/test_predict.py
git commit -m "fix: risk raster output as UInt16, NoData=0"
```

---

## Task 17 — Delete dead modules

**Files:**
- Delete: `palmdef_risk/process/lq.py`
- Delete: `palmdef_risk/process/correlation.py`
- Delete: `palmdef_risk/model/gwr.py`
- Delete: `tests/process/test_lq.py`
- Delete: `tests/model/test_gwr.py`

- [ ] **Step 1: Delete files**

```powershell
Remove-Item palmdef_risk\process\lq.py
Remove-Item palmdef_risk\process\correlation.py
Remove-Item palmdef_risk\model\gwr.py
Remove-Item tests\process\test_lq.py
Remove-Item tests\model\test_gwr.py
```

- [ ] **Step 2: Verify no remaining imports of deleted modules**

```
grep -r "from palmdef_risk.process.lq" .
grep -r "from palmdef_risk.process.correlation" .
grep -r "from palmdef_risk.model.gwr" .
grep -r "import lq" palmdef_risk/
grep -r "lq_direction" palmdef_risk/
grep -r "kde_bandwidth" palmdef_risk/
```

Fix any remaining references. Typically in `__init__.py` files.

- [ ] **Step 3: Run full test suite**

```
pytest tests/ -q
```
Expected: all PASS (zero imports of lq/correlation/gwr)

- [ ] **Step 4: Commit**

```
git add -A
git commit -m "refactor: delete lq.py, correlation.py, gwr.py and their tests"
```

---

## Task 18 — Simplify notebooks

**Files:**
- Rewrite: `notebooks/01_download.ipynb`
- Rewrite: `notebooks/02_process.ipynb`
- Rewrite: `notebooks/03_model.ipynb`

Each notebook is replaced with a minimal, clean version matching the spec (Section 7). Use `nbformat` to write programmatically, or edit cells directly. Key structure for each:

**01_download.ipynb** — 6 cells:
```
[0] config_path = "configs/my_run.yaml"
    use_cache = {"forest": False, "variables": False, "mill": True}
    from palmdef_risk.io.run import create_run
    ctx = create_run(config_path)

[1] from palmdef_risk.cache import CacheManager
    # Print per-dataset status
    cm = CacheManager(ctx.config.cache_dir)
    ...print status_report...

[2] from palmdef_risk.data.forest import download_forest
    download_forest(ctx, use_cache=use_cache["forest"])

[3] from palmdef_risk.data.variables import download_variables
    download_variables(ctx, use_cache=use_cache["variables"])

[4] from palmdef_risk.data.mill import download_mill
    download_mill(ctx, use_cache=use_cache["mill"])

[5] from palmdef_risk.data.user_inputs import ingest_user_inputs
    ingest_user_inputs(ctx)
```

**02_process.ipynb** — 5 cells:
```
[0] from palmdef_risk.io.run import load_run
    ctx = load_run("runs/my_run_dir")

[1] from palmdef_risk.process.align import align_all
    align_all(ctx, inputs=ctx.raw_dir)

[2] from palmdef_risk.process.distances import compute_all_distances
    compute_all_distances(ctx)

[3] from palmdef_risk.process.gravity import compute_gravity_accessibility, orthogonalize_gravity_ctx
    compute_gravity_accessibility(ctx)
    r2 = orthogonalize_gravity_ctx(ctx)
    print(f"Gravity R² = {r2:.3f}")

[4] from palmdef_risk.process.align import compute_hgu_signed_distance
    compute_hgu_signed_distance(
        str(ctx.raw_dir / "user_inputs" / "hgu.gpkg"),
        str(ctx.data_dir / "forest_t2.tif"),
        str(ctx.data_dir / "hgu_signed_dist.tif"),
    )
```

**03_model.ipynb** — 7 cells:
```
[0] from palmdef_risk.io.run import load_run
    ctx = load_run("runs/my_run_dir")

[1] import forestatrisk as far
    far.make_sample(...)  # build_sample_data

[2] from palmdef_risk.model.diagnostics import compute_vif
    compute_vif([...covariates...], ctx.output_dir / "sample.csv",
                ctx.output_dir / "diagnostics" / "vif.json")

[3] from palmdef_risk.model.icar import fit_all
    fit_all(ctx)

[4] from palmdef_risk.model.predict import predict_all
    predict_all(ctx)

[5] from palmdef_risk.model.diagnostics import compute_morans_i
    compute_morans_i(residuals, coords,
                     ctx.output_dir / "diagnostics" / "moran.json")

[6] from palmdef_risk.model.sensitivity import run_gravity_sensitivity
    run_gravity_sensitivity(ctx)
```

- [ ] **Step 1: Rewrite each notebook** (edit cells or use `nbformat.write`)

- [ ] **Step 2: Verify notebooks parse without import errors**

```
python -c "import nbformat; nbformat.read('notebooks/01_download.ipynb', as_version=4)"
python -c "import nbformat; nbformat.read('notebooks/02_process.ipynb', as_version=4)"
python -c "import nbformat; nbformat.read('notebooks/03_model.ipynb', as_version=4)"
```
Expected: no exception

- [ ] **Step 3: Commit**

```
git add notebooks/
git commit -m "refactor: simplify notebooks for gravity/icar pipeline"
```

---

## Task 19 — `.claude` folder updates

**Files:**
- Create: `.claude/CLAUDE.md`
- Create: `.claude/skills/pipeline-run.md`
- Create: `.claude/skills/pipeline-check.md`

- [ ] **Step 1: Create `.claude/CLAUDE.md`**

```markdown
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
```

- [ ] **Step 2: Create `.claude/skills/pipeline-run.md`**

```markdown
---
name: pipeline-run
description: Run the full palmdef_risk pipeline (all 3 notebooks) via papermill for a given config file.
---

# pipeline-run

Execute all three pipeline notebooks in sequence using papermill.

## Usage

```
/pipeline-run configs/my_run.yaml
```

## What this does

1. Runs `notebooks/01_download.ipynb` with `config_path` injected
2. Runs `notebooks/02_process.ipynb` loading the run created in step 1
3. Runs `notebooks/03_model.ipynb` loading the same run

## Implementation

When the user invokes `/pipeline-run <config_path>`, execute:

```bash
conda activate conda-far
python run.py --config <config_path>
```

Or, to run a single stage:
```bash
python run.py --config <config_path> --notebook 01_download
python run.py --config <config_path> --notebook 02_process --run-dir runs/<run_dir>
python run.py --config <config_path> --notebook 03_model --run-dir runs/<run_dir>
```

Progress is logged to `runs/<run_dir>/logs/run.log`.
```

- [ ] **Step 3: Create `.claude/skills/pipeline-check.md`**

```markdown
---
name: pipeline-check
description: Inspect a palmdef_risk run folder and report which stages are complete, partial, or missing.
---

# pipeline-check

Check the status of a run folder.

## Usage

```
/pipeline-check runs/wri_kalteng_20250501_120000
```

## What this does

Inspects the run folder and reports per-stage completion:

| Stage | Done if | Key files |
|---|---|---|
| Stage 1 (download) | raw/forest/ + raw/variables/ + raw/mill/ populated | `forest_t2.tif`, `protected.gpkg`, `mill_t2.gpkg` |
| Stage 2 (process) | data/ flat rasters present | `dist_road.tif`, `gravity_resid.tif`, `hgu_signed_dist.tif` |
| Stage 3 (model) | output/models/ populated | `mod_A.pkl`, `vif.json`, `moran.json` |

## Implementation

When the user invokes `/pipeline-check <run_dir>`, use Read and Glob tools to check:

```python
# Stage 1 checks
raw = Path(run_dir) / "data" / "raw"
s1_ok = all([
    (raw / "forest" / "forest_t2.tif").exists(),
    (raw / "variables" / "protected.gpkg").exists(),
    (raw / "mill" / "mill_t2.gpkg").exists(),
])

# Stage 2 checks
data = Path(run_dir) / "data"
s2_ok = all([
    (data / "dist_road.tif").exists(),
    (data / "gravity_resid.tif").exists(),
    (data / "hgu_signed_dist.tif").exists(),
])

# Stage 3 checks
out = Path(run_dir) / "output"
s3_ok = (out / "models").exists() and any((out / "models").glob("*/mod_*.pkl"))
```

Report as a table:
```
Stage 1 (download):  ✓ complete
Stage 2 (process):   ✓ complete
Stage 3 (model):     ✗ missing  — no mod_*.pkl found
```
```

- [ ] **Step 4: Create `.claude/` structure and commit**

```powershell
New-Item -ItemType Directory -Force .claude\skills
```

Write the three files, then:

```
git add .claude/
git commit -m "docs: add project CLAUDE.md and pipeline-run/pipeline-check skills"
```

---

## Final Check

- [ ] **Run full test suite**

```
pytest tests/ -q
```
Expected: all PASS

- [ ] **Verify no `palmoil_risk` references remain**

```
grep -r "palmoil_risk" . --include="*.py" --include="*.toml" --include="*.yaml" --include="*.md"
```
Expected: zero results

- [ ] **Verify no `pa.tif` or `"pa"` references remain in active code**

```
grep -rn '"pa"' palmdef_risk/
grep -rn "'pa'" palmdef_risk/
grep -rn "pa\.tif" palmdef_risk/
grep -rn "pa\.gpkg" palmdef_risk/
```
Expected: zero results (only `_PROTECTED_FILENAME = "protected"` and `protected.tif/gpkg` references)

- [ ] **Final commit**

```
git add -A
git commit -m "feat: palmdef_risk refactor complete — gravity/icar pipeline, Part 2"
```

---

## Implementation Phases Summary

| Phase | Tasks | Key deliverables |
|---|---|---|
| **Part 1 — Ph.1** | 1–3 | Package rename, config schema, run.py CRS auto-detect |
| **Part 1 — Ph.2** | 4–6 | utm.py, parallel.py, cache.py |
| **Part 1 — Ph.3** | 7–9 | mill.py (Trase/cumulative), forest.py/variables.py UTM |
| **Part 2 — Ph.4** | 10–12 | HGU signed-dist, distances.py, gravity.py |
| **Part 2 — Ph.5** | 13–16 | icar.py A/B/C, diagnostics.py VIF/Moran, sensitivity.py |
| **Part 2 — Ph.6** | 17–19 | Delete dead code, notebooks, .claude folder |
