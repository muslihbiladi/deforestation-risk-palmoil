# BIG RBI River Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Badan Informasi Geospasial (BIG) RBI as a selectable river-data source for the `dist_river` covariate, alongside the existing OSM download and user-file options.

**Architecture:** A three-way `user_inputs.river.source` (`big` | `osm` | `user`, default `big`) selects the river provider. A new `get_rivers_big()` downloader fetches geometry-only features from the BIG RBI ArcGIS REST MapServer (Layer 237 centrelines + Layer 257 wide-river polygons), paginates, merges, clips to the AOI, and writes the same `river.gpkg` contract every downstream stage already consumes. Stages 2 and 3 are untouched.

**Tech Stack:** Python, geopandas, shapely, GDAL/OGR, `requests` (ArcGIS REST GeoJSON), pytest.

**Spec:** `docs/superpowers/specs/2026-06-12-big-river-source-design.md`

**Working dir / commands:** run tests from the repo root inside the `palmdef-risk` conda env: `python -m pytest`. The pipeline runs with CWD=`active/`.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `active/palmdef_risk/io/config.py` | `RunConfig` dataclass + YAML parse + validation | Add `river_source` field, parse, validate |
| `active/configs/schema.json` | JSON-schema for config validation | Add `source` enum to `user_inputs.river` |
| `active/configs/template.yaml` | User-facing config template | Add `river.source: big` |
| `active/configs/central-kalimantan.yaml` | Run config | Set `river.source: big` |
| `active/configs/east-kotawaringin.yaml` | Run config | Set `river.source: big` |
| `active/palmdef_risk/data/variables.py` | Stage-1 downloaders | Add BIG constants, `_big_query_layer`, `get_rivers_big`; river dispatch in `download_variables`; river check in `_variables_complete`; pass `river_source` to cache key |
| `active/palmdef_risk/cache.py` | Cross-run cache keys | Add `river_source` to `variables_key` |
| `active/palmdef_risk/data/user_inputs.py` | Stage-1 user-file ingest | Route river off `river_source == "user"` |
| `active/notebooks/01_download.ipynb` | Stage-1 notebook | Pass `river_source` to `variables_key` call |
| `active/tests/test_cache.py` | Cache tests | Update `variables_key` call signatures |
| `active/tests/data/test_variables_river_big.py` | New | Offline tests for BIG downloader + dispatch |
| `active/tests/io/test_config_river.py` | New | Config field + validation tests |

---

## Task 1: Add `river_source` to RunConfig (field, parse, validation)

**Files:**
- Modify: `active/palmdef_risk/io/config.py`
- Test: `active/tests/io/test_config_river.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `active/tests/io/test_config_river.py`:

```python
import yaml
import pytest
from palmdef_risk.io.config import RunConfig


def _base_cfg_dict(tmp_path):
    """Minimal valid config dict; river block omitted on purpose."""
    aoi = tmp_path / "aoi.gpkg"
    aoi.write_text("")  # presence only; from_yaml does not open it
    return {
        "run": {"project": "p", "area": "a", "task": "t"},
        "aoi": {"source": str(aoi), "buffer": 0.0},
        "crs": "EPSG:32750",
        "forest": {"source": "tmf", "years": [2015, 2020, 2024], "perc": 75},
        "variables": {"use_ghsl_towns": False, "ghsl_years": None, "osm_timeout": 180},
        "user_inputs": {
            "peatland": {"path": "x", "type": "binary"},
            "hgu": {"path": "x"},
            "plantation": {"t2": None, "t3": None},
        },
        "mill": {"source": "trase", "path": None},
        "process": {"gravity": {"sigma_km": 25.0, "radius_km": 80.0}},
        "model": {"variants": ["A"]},
        "output": {"project_future": False, "projection_year": 2035},
    }


def _write(tmp_path, d):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(d))
    return p


def test_river_source_defaults_to_big(tmp_path):
    cfg = RunConfig.from_yaml(_write(tmp_path, _base_cfg_dict(tmp_path)))
    assert cfg.river_source == "big"


def test_river_source_parsed_from_yaml(tmp_path):
    d = _base_cfg_dict(tmp_path)
    d["user_inputs"]["river"] = {"source": "osm", "path": None}
    cfg = RunConfig.from_yaml(_write(tmp_path, d))
    assert cfg.river_source == "osm"
    assert cfg.river_path is None


def test_invalid_river_source_rejected(tmp_path):
    d = _base_cfg_dict(tmp_path)
    d["user_inputs"]["river"] = {"source": "nope", "path": None}
    cfg = RunConfig.from_yaml(_write(tmp_path, d))
    errs = cfg.validate()
    assert any("river.source" in e for e in errs)


def test_user_source_requires_path(tmp_path):
    d = _base_cfg_dict(tmp_path)
    d["user_inputs"]["river"] = {"source": "user", "path": None}
    cfg = RunConfig.from_yaml(_write(tmp_path, d))
    errs = cfg.validate()
    assert any("river.path required" in e for e in errs)


def test_user_source_with_path_ok(tmp_path):
    rv = tmp_path / "myriver.gpkg"
    rv.write_text("")
    d = _base_cfg_dict(tmp_path)
    d["user_inputs"]["river"] = {"source": "user", "path": str(rv)}
    cfg = RunConfig.from_yaml(_write(tmp_path, d))
    errs = cfg.validate()
    assert not any("river" in e for e in errs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest active/tests/io/test_config_river.py -v`
Expected: FAIL — `AttributeError: 'RunConfig' object has no attribute 'river_source'`.

- [ ] **Step 3: Add the field to the dataclass**

In `active/palmdef_risk/io/config.py`, in the `# User inputs` block, add `river_source` immediately before `river_path` (line 40):

```python
    plantation_industrial_value: int
    plantation_smallholder_value: int
    river_source: str
    river_path: Optional[str]
```

- [ ] **Step 4: Parse it in `from_yaml`**

In the `cls(...)` call, add the parse line immediately before the existing `river_path=...` (line 107):

```python
            plantation_smallholder_value=int(plant.get("smallholder_value", 2)),
            river_source=str(riv.get("source", "big")),
            river_path=str(riv["path"]) if riv.get("path") else None,
```

- [ ] **Step 5: Add validation**

In `validate()`, add these checks (place them next to the existing mill checks around line 161):

```python
        if self.river_source not in ("big", "osm", "user"):
            errors.append("user_inputs.river.source must be 'big', 'osm', or 'user'")
        if self.river_source == "user" and not self.river_path:
            errors.append("user_inputs.river.path required when river.source: 'user'")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest active/tests/io/test_config_river.py -v`
Expected: PASS (5 passed).

- [ ] **Step 7: Commit**

```bash
git add active/palmdef_risk/io/config.py active/tests/io/test_config_river.py
git commit -m "feat(config): add river_source field with validation"
```

---

## Task 2: Update config schema, template, and run configs

**Files:**
- Modify: `active/configs/schema.json`
- Modify: `active/configs/template.yaml`
- Modify: `active/configs/central-kalimantan.yaml`
- Modify: `active/configs/east-kotawaringin.yaml`

- [ ] **Step 1: Add `source` to the schema's river block**

In `active/configs/schema.json`, replace the `river` property (lines 101–107):

```json
        "river": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "source": {"type": "string", "enum": ["big", "osm", "user"]},
            "path": {"type": ["string", "null"]}
          }
        }
```

- [ ] **Step 2: Update the template**

In `active/configs/template.yaml`, replace the `river:` block (lines 52–53):

```yaml
  river:
    source: big   # "big" (BIG RBI REST, default) | "osm" (OpenStreetMap) | "user" (local file)
    path: null    # required only when source: user; GeoPackage or shapefile of river lines
```

- [ ] **Step 3: Update central-kalimantan.yaml**

In `active/configs/central-kalimantan.yaml`, replace the `river:` block (lines 51–52):

```yaml
  river:
    source: big   # BIG RBI national topographic rivers (better Kalimantan coverage than OSM)
    path: null
```

- [ ] **Step 4: Update east-kotawaringin.yaml**

In `active/configs/east-kotawaringin.yaml`, replace the `river:` block (lines 51–52):

```yaml
  river:
    source: big   # BIG RBI national topographic rivers (better Kalimantan coverage than OSM)
    path: null
```

- [ ] **Step 5: Validate the configs parse and pass schema**

Run: `cd active && python run.py --config configs/central-kalimantan.yaml --dry-run`
Expected: dry-run completes, no schema/validation error; prints planned run folder. Then `cd ..`.

- [ ] **Step 6: Commit**

```bash
git add active/configs/schema.json active/configs/template.yaml active/configs/central-kalimantan.yaml active/configs/east-kotawaringin.yaml
git commit -m "feat(config): river.source option in schema, template, run configs"
```

---

## Task 3: BIG REST pagination helper `_big_query_layer`

**Files:**
- Modify: `active/palmdef_risk/data/variables.py`
- Test: `active/tests/data/test_variables_river_big.py` (create)

- [ ] **Step 1: Write the failing test**

Create `active/tests/data/test_variables_river_big.py`:

```python
from unittest.mock import patch, MagicMock
import geopandas as gpd
import pytest


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _line_feature(oid, x0=110.0):
    return {
        "type": "Feature",
        "properties": {"OBJECTID": oid},
        "geometry": {"type": "LineString",
                     "coordinates": [[x0, -1.0], [x0 + 0.01, -1.01]]},
    }


def test_big_query_layer_paginates():
    from palmdef_risk.data import variables as v

    page1 = {"features": [_line_feature(i) for i in range(1000)],
             "properties": {"exceededTransferLimit": True}}
    page2 = {"features": [_line_feature(1000 + i) for i in range(5)],
             "properties": {"exceededTransferLimit": False}}
    responses = [_FakeResp(page1), _FakeResp(page2)]

    with patch.object(v, "requests", create=True) as mock_requests:
        mock_requests.get = MagicMock(side_effect=responses)
        feats = v._big_query_layer(237, (110.0, -2.0, 111.0, 0.0),
                                   timeout=10, verbose=False)
    assert len(feats) == 1005
    assert mock_requests.get.call_count == 2


def test_big_query_layer_hard_fails():
    from palmdef_risk.data import variables as v
    with patch.object(v, "requests", create=True) as mock_requests:
        mock_requests.get = MagicMock(side_effect=Exception("boom"))
        with pytest.raises(RuntimeError):
            v._big_query_layer(237, (110.0, -2.0, 111.0, 0.0),
                               timeout=1, verbose=False)
```

> Note: `_big_query_layer` does `import requests` at the top of the function, so
> the module attribute `variables.requests` does not exist by default. The test
> patches it with `create=True`; the implementation in Step 3 imports `requests`
> at **module level** (add `import requests` to the import block) so the patch
> target exists. Update the import block accordingly in Step 3.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest active/tests/data/test_variables_river_big.py -v`
Expected: FAIL — `AttributeError: module 'palmdef_risk.data.variables' has no attribute '_big_query_layer'`.

- [ ] **Step 3: Implement the helper and constants**

In `active/palmdef_risk/data/variables.py`, add `import requests` to the top import block (after `import time`):

```python
import time
import requests
import multiprocessing
```

Then add the BIG section just above the `# osmnx-based OSM downloader` section (before line 951, the `_OVERPASS_ENDPOINTS` block):

```python
# ============================================================
# BIG RBI river downloader (ArcGIS REST MapServer)
# ============================================================

# Rupa Bumi Indonesia 1:50,000 national basemap (public, no auth).
_BIG_RBI_URL = (
    "https://geoservices.big.go.id/rbi/rest/services/"
    "BASEMAP/Rupabumi_Indonesia/MapServer"
)
_BIG_RIVER_LINE_LAYER = 237   # Sungai (Garis) — centrelines (polyline)
_BIG_RIVER_AREA_LAYER = 257   # Sungai (area)  — wide rivers (polygon)
_BIG_PAGE_SIZE = 1000         # service max records per query
_BIG_UA = "palmdef_risk/1.0 (deforestation risk research; +https://www.wri.org)"


def _big_query_layer(layer_id, bbox, timeout=180, verbose=True):
    """Fetch all GeoJSON features from one BIG RBI MapServer layer in a bbox.

    Geometry + OBJECTID only (dist_river uses presence, not attributes).
    Paginates on resultOffset until the service stops setting
    exceededTransferLimit. Both layers are requested with outSR=4326 so the
    response geometry is already in EPSG:4326.

    Raises RuntimeError if any page fails on every retry — the caller does NOT
    fall back to OSM (the user explicitly chose source=big).

    :param layer_id: BIG MapServer layer id (237 or 257).
    :param bbox: (xmin, ymin, xmax, ymax) in EPSG:4326.
    :return: list of GeoJSON feature dicts.
    """
    xmin, ymin, xmax, ymax = bbox
    url = f"{_BIG_RBI_URL}/{layer_id}/query"
    headers = {"User-Agent": _BIG_UA}
    params = {
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "true",
        "outFields": "OBJECTID",
        "geometryPrecision": "5",
        "resultRecordCount": _BIG_PAGE_SIZE,
        "f": "geojson",
    }

    features = []
    offset = 0
    while True:
        params["resultOffset"] = offset
        data = None
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, headers=headers,
                                 timeout=timeout + 30)
                if r.status_code == 200:
                    data = r.json()
                    break
                if r.status_code in (429, 503, 504) and attempt < 2:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                break  # other status — give up on this page
            except Exception:
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 10))
                    continue
        if data is None:
            raise RuntimeError(
                f"BIG RBI layer {layer_id} query failed at offset {offset} "
                f"(no successful response after retries)."
            )
        page = data.get("features", []) or []
        features.extend(page)
        if verbose:
            print(f"    BIG layer {layer_id}: +{len(page)} "
                  f"(total {len(features)})")
        exceeded = (data.get("properties", {}) or {}).get(
            "exceededTransferLimit", False)
        if not exceeded or len(page) < _BIG_PAGE_SIZE:
            break
        offset += _BIG_PAGE_SIZE
        time.sleep(1)  # be polite to the public service
    return features
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest active/tests/data/test_variables_river_big.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add active/palmdef_risk/data/variables.py active/tests/data/test_variables_river_big.py
git commit -m "feat(variables): BIG RBI paginated layer query helper"
```

---

## Task 4: `get_rivers_big` — merge 237+257, clip, write `river.gpkg`

**Files:**
- Modify: `active/palmdef_risk/data/variables.py`
- Test: `active/tests/data/test_variables_river_big.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `active/tests/data/test_variables_river_big.py`:

```python
def _poly_feature(oid, x0=110.2):
    return {
        "type": "Feature",
        "properties": {"OBJECTID": oid},
        "geometry": {"type": "Polygon",
                     "coordinates": [[[x0, -1.0], [x0 + 0.02, -1.0],
                                      [x0 + 0.02, -1.02], [x0, -1.02],
                                      [x0, -1.0]]]},
    }


def test_get_rivers_big_merges_layers(tmp_path):
    from palmdef_risk.data import variables as v

    lines = [_line_feature(i, x0=110.1) for i in range(3)]
    polys = [_poly_feature(100 + i, x0=110.2) for i in range(2)]

    def fake_query(layer_id, bbox, timeout=180, verbose=True):
        return lines if layer_id == v._BIG_RIVER_LINE_LAYER else polys

    aoi = (110.0, -1.5, 110.5, -0.5)  # bbox AOI covering all features
    with patch.object(v, "_big_query_layer", side_effect=fake_query):
        out = v.get_rivers_big(aoi, output_dir=str(tmp_path),
                               output_crs="EPSG:32749", verbose=False)

    gpkg = out["river"]
    gdf = gpd.read_file(gpkg)
    assert len(gdf) == 5  # 3 lines + 2 polygons merged
    geom_types = set(gdf.geometry.geom_type)
    assert {"LineString"}.issubset(geom_types)
    assert any("Polygon" in t for t in geom_types)
    assert gdf.crs.to_epsg() == 32749  # reprojected to requested CRS


def test_get_rivers_big_empty_writes_empty_gpkg(tmp_path):
    from palmdef_risk.data import variables as v
    with patch.object(v, "_big_query_layer", return_value=[]):
        out = v.get_rivers_big((110.0, -1.5, 110.5, -0.5),
                               output_dir=str(tmp_path), verbose=False)
    assert out["river"].endswith("river.gpkg")
    import os
    assert os.path.exists(out["river"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest active/tests/data/test_variables_river_big.py -k get_rivers_big -v`
Expected: FAIL — `AttributeError: ... has no attribute 'get_rivers_big'`.

- [ ] **Step 3: Implement `get_rivers_big`**

In `active/palmdef_risk/data/variables.py`, add directly after `_big_query_layer` (still in the BIG section):

```python
def get_rivers_big(aoi, output_dir="data", buff=0.0, output_crs=None,
                   timeout=180, verbose=True):
    """Download river features from BIG RBI (Layers 237 lines + 257 polygons).

    Fetches geometry only, merges both layers into one generic-geometry
    GeoDataFrame, clips to the AOI polygon, optionally reprojects to
    output_crs, and writes <output_dir>/river.gpkg. Wide-river polygons (257)
    are kept as polygons: process/distances.py burns them as a presence area,
    which is the correct representation for dist_river.

    Hard-fails (RuntimeError, raised by _big_query_layer) if the BIG service
    cannot be reached — no silent OSM fallback.

    Signature mirrors get_rivers() so download_variables can dispatch with the
    same kwargs. Returns {"river": path}.
    """
    from shapely.geometry import shape

    if verbose:
        print("=" * 60)
        print("Downloading BIG RBI rivers (Layers 237 + 257)...")

    os.makedirs(output_dir, exist_ok=True)
    polygon = _load_aoi_polygon(aoi, buff)
    bbox = polygon.bounds  # (xmin, ymin, xmax, ymax) in EPSG:4326
    gpkg_path = os.path.join(output_dir, "river.gpkg")

    geoms = []
    for layer_id in (_BIG_RIVER_LINE_LAYER, _BIG_RIVER_AREA_LAYER):
        feats = _big_query_layer(layer_id, bbox, timeout=timeout,
                                 verbose=verbose)
        for ft in feats:
            g = ft.get("geometry")
            if g:
                geoms.append(shape(g))

    if not geoms:
        if verbose:
            print("  No BIG river features in AOI — writing empty river.gpkg.")
        _create_empty_gpkg(gpkg_path, ogr.wkbLineString)
        return {"river": gpkg_path}

    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")

    # Clip to the AOI polygon (in EPSG:4326 before reprojection)
    aoi_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    before = len(gdf)
    gdf = gpd.clip(gdf, aoi_gdf)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    if verbose:
        print(f"  Clipped to AOI: {before} -> {len(gdf)} features")

    if gdf.empty:
        _create_empty_gpkg(gpkg_path, ogr.wkbLineString)
        return {"river": gpkg_path}

    if output_crs is not None:
        gdf = gdf.to_crs(output_crs)

    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    gdf.to_file(gpkg_path, driver="GPKG")

    if verbose:
        print(f"  {len(gdf)} features -> {gpkg_path}")
    return {"river": gpkg_path}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest active/tests/data/test_variables_river_big.py -v`
Expected: PASS (4 passed total).

- [ ] **Step 5: Commit**

```bash
git add active/palmdef_risk/data/variables.py active/tests/data/test_variables_river_big.py
git commit -m "feat(variables): get_rivers_big downloader (merge 237+257, clip, write)"
```

---

## Task 5: Add `river_source` to the variables cache key

**Files:**
- Modify: `active/palmdef_risk/cache.py:46-47`
- Modify: `active/tests/test_cache.py:51`
- Test: `active/tests/test_cache.py` (add a discrimination test)

- [ ] **Step 1: Write the failing test**

Add to `active/tests/test_cache.py`:

```python
def test_variables_key_differs_by_river_source(tmp_path):
    from palmdef_risk.cache import CacheManager
    cm = CacheManager(tmp_path / "cache")
    k_big = cm.variables_key((0, 0, 1, 1), 0, False, None, 180, "big")
    k_osm = cm.variables_key((0, 0, 1, 1), 0, False, None, 180, "osm")
    assert k_big != k_osm
```

Also update the existing call in `test_status_report_keys` (line 51) to pass a river source:

```python
    kv = cm.variables_key((0, 0, 1, 1), 0, False, None, 180, "big")
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `python -m pytest active/tests/test_cache.py -v`
Expected: FAIL — `test_variables_key_differs_by_river_source` raises `TypeError: variables_key() takes 6 positional arguments but 7 were given`.

- [ ] **Step 3: Add the parameter to `variables_key`**

In `active/palmdef_risk/cache.py`, replace lines 46–47:

```python
    def variables_key(self, aoi_bbox, buffer, use_ghsl, ghsl_years, timeout,
                      river_source="big") -> str:
        return _hash(aoi_bbox, buffer, use_ghsl, ghsl_years, timeout, river_source)
```

(The `="big"` default keeps any not-yet-updated caller working; explicit callers below pass it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest active/tests/test_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add active/palmdef_risk/cache.py active/tests/test_cache.py
git commit -m "feat(cache): include river_source in variables cache key"
```

---

## Task 6: Wire dispatch in `download_variables` + `_variables_complete` + cache-key call

**Files:**
- Modify: `active/palmdef_risk/data/variables.py` (`_variables_complete` ~1390, `download_variables` cache-key ~1418 and river block ~1481)
- Test: `active/tests/data/test_variables_river_big.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `active/tests/data/test_variables_river_big.py`:

```python
from palmdef_risk.io.run import create_run


class _Cfg:
    """Duck-typed config for _variables_complete river logic."""
    def __init__(self, river_source, use_ghsl_towns=False):
        self.river_source = river_source
        self.use_ghsl_towns = use_ghsl_towns


def test_variables_complete_river_required_for_big(tmp_path):
    from palmdef_risk.data.variables import _variables_complete
    # All non-river required files present, but river.gpkg missing.
    for name in ("altitude.tif", "slope.tif", "protected.gpkg",
                 "road.gpkg", "town.gpkg"):
        (tmp_path / name).write_text("x")
    assert _variables_complete(tmp_path, _Cfg("big")) is False
    (tmp_path / "river.gpkg").write_text("x")
    assert _variables_complete(tmp_path, _Cfg("big")) is True


def test_variables_complete_river_not_required_for_user(tmp_path):
    from palmdef_risk.data.variables import _variables_complete
    for name in ("altitude.tif", "slope.tif", "protected.gpkg",
                 "road.gpkg", "town.gpkg"):
        (tmp_path / name).write_text("x")
    # river.gpkg absent, but source=user → still complete.
    assert _variables_complete(tmp_path, _Cfg("user")) is True


def test_download_variables_dispatches_big(minimal_config_yaml, tmp_path):
    from palmdef_risk.data import variables as v
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    ctx.config.river_source = "big"  # minimal_config_yaml omits river → default big

    called = {}

    def fake_big(**kw):
        called["big"] = True
        out = kw["output_dir"]
        import os
        os.makedirs(out, exist_ok=True)
        p = os.path.join(out, "river.gpkg")
        open(p, "w").close()
        return {"river": p}

    def fake_osm(**kw):
        called["osm"] = True
        return {}

    # Stub every other downloader so only river logic runs.
    with patch.object(v, "get_srtm", return_value={}), \
         patch.object(v, "get_wdpa", return_value={}), \
         patch.object(v, "get_roads", return_value={}), \
         patch.object(v, "get_towns", return_value={}), \
         patch.object(v, "ee"), \
         patch.object(v, "get_rivers_big", side_effect=fake_big), \
         patch.object(v, "get_rivers", side_effect=fake_osm):
        v.download_variables(ctx, use_cache=False)

    assert called.get("big") is True
    assert "osm" not in called
```

> The `minimal_config_yaml` fixture lives in `active/tests/conftest.py` and omits
> a river block, so `cfg.river_source` defaults to `"big"`. The test sets it
> explicitly for clarity. `create_run` is imported from `palmdef_risk.io.run`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest active/tests/data/test_variables_river_big.py -k "complete or dispatch" -v`
Expected: FAIL — `_variables_complete` still keys off `river_path`; `download_variables` still calls `get_rivers` only (no `get_rivers_big` dispatch).

- [ ] **Step 3: Update `_variables_complete`**

In `active/palmdef_risk/data/variables.py`, replace the river check (lines 1390–1392):

```python
    # river: required unless the user supplies their own file
    # (user file lives in user_inputs/, not variables/)
    if cfg.river_source != "user":
        required.append(out_dir / "river.gpkg")
```

- [ ] **Step 4: Pass `river_source` into the cache key**

Replace the `variables_key` call (line 1418):

```python
    _vkey = _cm.variables_key(_bbox, cfg.aoi_buffer, cfg.use_ghsl_towns,
                              cfg.ghsl_years, cfg.osm_timeout, cfg.river_source)
```

- [ ] **Step 5: Replace the river dispatch block**

Replace the rivers block in `download_variables` (lines 1481–1487):

```python
    # Rivers — source-dependent: user file (ingested separately), BIG RBI, or OSM
    if cfg.river_source == "user":
        print("Variables: river.source=user — river ingested from user_inputs, "
              "skipping download.")
    elif (out_dir / "river.gpkg").exists():
        print("Variables: river.gpkg already present, skipping rivers.")
    elif cfg.river_source == "big":
        result.update(get_rivers_big(**osm_kwargs))
    else:  # "osm"
        result.update(get_rivers(**osm_kwargs))
```

> `osm_kwargs` (defined at line ~1438) is `dict(aoi, output_dir, buff,
> output_crs, timeout, verbose)` — it matches `get_rivers_big`'s signature, so
> no new kwargs dict is needed.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest active/tests/data/test_variables_river_big.py -v`
Expected: PASS (all tests in the module).

- [ ] **Step 7: Commit**

```bash
git add active/palmdef_risk/data/variables.py active/tests/data/test_variables_river_big.py
git commit -m "feat(variables): dispatch river download on river_source"
```

---

## Task 7: Route user-supplied river off `river_source` in `ingest_user_inputs`

**Files:**
- Modify: `active/palmdef_risk/data/user_inputs.py:41-46`
- Test: `active/tests/data/test_user_inputs.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `active/tests/data/test_user_inputs.py` (reuse existing fixtures in that file / conftest; mirror the file's existing style):

```python
from unittest.mock import patch
from palmdef_risk.data.user_inputs import ingest_user_inputs
from palmdef_risk.io.run import create_run


def test_river_source_user_copies_file(minimal_config_yaml, tmp_path, tiny_vector):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    ctx.config.river_source = "user"
    ctx.config.river_path = str(tiny_vector)
    result = ingest_user_inputs(ctx)
    assert result["river"] is not None
    assert result["river"].exists()


def test_river_source_big_skips_copy(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    ctx.config.river_source = "big"
    result = ingest_user_inputs(ctx)
    assert result["river"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest active/tests/data/test_user_inputs.py -k river -v`
Expected: FAIL — current code keys off `cfg.river_path` truthiness, not `river_source`; `test_river_source_big_skips_copy` may pass by luck but `test_river_source_user_copies_file` semantics are not guaranteed. (If both already pass, still proceed to Step 3 to make the intent explicit.)

- [ ] **Step 3: Update the river routing**

In `active/palmdef_risk/data/user_inputs.py`, replace lines 41–46:

```python
    if cfg.river_source == "user":
        result["river"] = _copy_vector(cfg.river_path, dst, "river")
        log.info("User-supplied river (source=user) will override downloaded river")
    else:
        log.info("river.source=%s — river will be downloaded in Stage 1",
                 cfg.river_source)
        result["river"] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest active/tests/data/test_user_inputs.py -k river -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add active/palmdef_risk/data/user_inputs.py active/tests/data/test_user_inputs.py
git commit -m "feat(user_inputs): route river ingest on river_source"
```

---

## Task 8: Update the download notebook's cache-key call

**Files:**
- Modify: `active/notebooks/01_download.ipynb` (cell id `fd34e018`)

- [ ] **Step 1: Update the `variables_key` call in the notebook**

In `active/notebooks/01_download.ipynb`, cell `fd34e018`, replace the `vars_key = ...` call so it passes the river source (use NotebookEdit):

```python
vars_key = cm.variables_key(bbox, ctx.config.aoi_buffer,
                            ctx.config.use_ghsl_towns,
                            ctx.config.ghsl_years, ctx.config.osm_timeout,
                            ctx.config.river_source)
```

- [ ] **Step 2: Verify the notebook cell parses**

Run: `python -c "import json; nb=json.load(open(r'active/notebooks/01_download.ipynb', encoding='utf-8')); print('cells', len(nb['cells']))"`
Expected: prints a cell count, no JSON error.

- [ ] **Step 3: Commit**

```bash
git add active/notebooks/01_download.ipynb
git commit -m "chore(notebook): pass river_source to variables_key"
```

---

## Task 9: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest`
Expected: all tests pass (existing suite green + new river/config/cache tests). If any pre-existing test fails for an unrelated reason, note it but do not mask it.

- [ ] **Step 2: Confirm no stray `river_path`-keyed logic remains**

Run: `python -m pytest active/tests -k "river" -v` and grep:
Run: `git grep -n "river_path" -- active/palmdef_risk`
Expected: `river_path` appears only in `config.py` (field + parse) and `user_inputs.py` (the `_copy_vector(cfg.river_path, ...)` call inside the `source == "user"` branch). No `download_variables`/`_variables_complete` references to `river_path` remain.

- [ ] **Step 3: Final commit (if any verification fixups were needed)**

```bash
git add -A
git commit -m "test: verify BIG river source end-to-end"
```

---

## Notes for the implementer

- **TDD discipline:** every task writes the test first, watches it fail, then implements. Do not skip the "verify it fails" step.
- **`requests` import:** Task 3 moves `requests` to a module-level import in `variables.py` so tests can patch `variables.requests`. The existing `_overpass_post` does `import requests` locally; leave that as-is (harmless shadow) or remove it — do not change OSM behavior.
- **Mixed-geometry GPKG:** geopandas writes a merged lines+polygons GeoDataFrame as a generic-geometry GPKG layer. `process/distances.py::rasterize_vector` burns all features to presence=1 regardless of geometry type, so `dist_river` is correct without polygon→line conversion.
- **Default change:** `river_source` defaults to `big`. Existing configs without a `river` block now use BIG. This is intended (decision #4 in the spec).
- **No network in tests:** all BIG access is patched. Never hit `geoservices.big.go.id` in the suite.
- **Hard-fail contract:** `_big_query_layer` raises `RuntimeError` on unreachable service; `get_rivers_big` does not catch it, so a failed BIG download stops the run (decision #5). Do not add an OSM fallback.
