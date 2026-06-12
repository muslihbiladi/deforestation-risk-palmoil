# BIG RBI River Source — Design Spec

**Date:** 2026-06-12
**Status:** Approved (brainstorming)
**Topic:** Add Badan Informasi Geospasial (BIG) RBI as a river-data source for the
`dist_river` covariate, alongside the existing OSM and user-file sources.
**Source note:** `notes/note_4.md`

---

## 1. Problem

OSM waterway coverage in rural Kalimantan is patchy (volunteer-driven), so the
`dist_river` covariate is unreliable. BIG's Rupa Bumi Indonesia (RBI) 1:50,000
national topographic basemap is a systematic government survey with nationwide
coverage and no rural gaps. We add BIG as a selectable river source.

**Service:** `https://geoservices.big.go.id/rbi/rest/services/BASEMAP/Rupabumi_Indonesia/MapServer`
(ArcGIS REST v10.81, public, no auth).

- **Layer 237** — Sungai (Garis): river centrelines, polyline. Native EPSG:4326.
- **Layer 257** — Sungai (area): wide rivers as polygons. Native EPSG:3857.

Both layers are fetched with `outSR=4326` so the service returns geometry already
in EPSG:4326, unifying CRS before merge.

---

## 2. Decisions (locked in brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Config shape | Three-way `user_inputs.river.source` = `big` \| `osm` \| `user` |
| 2 | Layer 257 polygons | Merge with Layer 237 lines into one `river.gpkg` |
| 3 | KLSSNG stream-order filter | **Not** implemented (field mostly null in Kalimantan) |
| 4 | Default source | **`big`** (behavior change from implicit OSM) |
| 5 | BIG fetch failure | **Hard fail** — raise, stop the run; no silent OSM fallback |

### Why merge 257 as polygons (not convert to lines)

`process/distances.py::_proximity_from_vector` rasterizes the river vector to a
presence mask (`burn_value=1`) and computes a distance transform from those
cells. Burning a wide-river **polygon as a filled area** is the correct presence
representation for `dist_river` — converting it to a boundary line would make the
river interior register as "far from river". So Layer 257 polygons are kept as
polygons and merged into a single generic-geometry `river.gpkg` layer; rasterize
burns all features (lines and polygons) identically. No skeletonization needed.

---

## 3. Architecture

The change is localized to Stage 1 (download). Stage 2 (process) and Stage 3
(model) are untouched because the output contract is unchanged: a single
`variables/river.gpkg` in the run CRS, consumed exactly as today.

### 3.1 Config layer

**`configs/schema.json`** — extend `user_inputs.river`:
```json
"river": {
  "type": "object",
  "properties": {
    "source": { "type": "string", "enum": ["big", "osm", "user"], "default": "big" },
    "path":   { "type": ["string", "null"], "default": null }
  }
}
```

**`configs/template.yaml`**:
```yaml
user_inputs:
  river:
    source: big        # "big" (BIG RBI REST, default) | "osm" | "user"
    path: null         # required only when source: user
```

**`palmdef_risk/io/config.py`** — `RunConfig`:
- Add field `river_source: str` (default `"big"`), parsed from `riv.get("source", "big")`.
- Keep existing `river_path`.
- Validation (in the existing validate step):
  - `river_source` must be one of `{big, osm, user}`.
  - `source == "user"` ⇒ `river_path` set and the file readable (reuse existing
    user-input readability check).
  - `source in {big, osm}` ⇒ `river_path` ignored (not an error if present, but
    not used).

### 3.2 Downloader — `get_rivers_big()`

New function in `palmdef_risk/data/variables.py`, signature mirroring
`get_rivers()`:

```python
def get_rivers_big(aoi, output_dir="data", buff=0.0, output_crs=None,
                   timeout=180, verbose=True):
```

Behavior:
1. `polygon = _load_aoi_polygon(aoi, buff)`; bbox = `polygon.bounds` (EPSG:4326).
2. For each layer in (237, 257): paginated REST query (see §3.3), collect GeoJSON
   feature pages. Geometry only — no attribute fields are fetched (`dist_river`
   uses presence, not names/classes).
3. Build a GeoDataFrame per layer (EPSG:4326), concat into one GeoDataFrame
   (generic geometry — lines + polygons coexist).
4. Clip to the AOI polygon (`gpd.clip`), drop empty/null geometries.
5. If `output_crs` is set, reproject.
6. Write `output_dir/river.gpkg` (driver GPKG). Return `{"river": path}`.
7. On total fetch failure (every retry on a needed page fails) → **raise**
   `RuntimeError` (hard fail).

Module constants:
```python
_BIG_RBI_URL = "https://geoservices.big.go.id/rbi/rest/services/BASEMAP/Rupabumi_Indonesia/MapServer"
_BIG_RIVER_LINE_LAYER = 237
_BIG_RIVER_AREA_LAYER = 257
_BIG_PAGE_SIZE = 1000
_BIG_UA = "palmdef_risk/1.0 (deforestation risk research; +https://www.wri.org)"
```

### 3.3 Pagination helper

A small private helper `_big_query_layer(layer_id, bbox, timeout, verbose)`:
- Loops `resultOffset` in steps of `_BIG_PAGE_SIZE`.
- Query params:
  ```
  geometry={xmin},{ymin},{xmax},{ymax}
  geometryType=esriGeometryEnvelope
  inSR=4326
  outSR=4326
  spatialRel=esriSpatialRelIntersects
  returnGeometry=true
  outFields=OBJECTID
  geometryPrecision=5
  resultOffset={offset}
  resultRecordCount=1000
  f=geojson
  ```
  **Minimal payload:** only geometry + `OBJECTID` are requested (OID kept for
  stable pagination ordering; all other attributes dropped — unused downstream).
  `geometryPrecision=5` trims coordinates to ~1 m to further shrink each page.
- Uses `requests.get` with `_BIG_UA` header and retry/backoff mirroring
  `_overpass_post` (one retry per attempt, short sleep on transient failure).
- Continues until response `properties.exceededTransferLimit` is false/absent
  **and** the page returned `< _BIG_PAGE_SIZE` features.
- Returns the accumulated list of GeoJSON features. Raises on unrecoverable
  failure (so `get_rivers_big` can hard-fail).

### 3.4 Routing

**`palmdef_risk/data/variables.py::download_variables`** — replace the current
river block with a 3-way dispatch on `cfg.river_source`:
- `user` → user-supplied file is copied by `ingest_user_inputs` (Stage-1 user
  inputs); `download_variables` skips river (as it does today when `river_path`
  is set).
- `big` → `get_rivers_big(**osm_kwargs)` (same kwargs shape as OSM getters).
- `osm` → `get_rivers(**osm_kwargs)` (existing behavior).
- Skip if `river.gpkg` already present (resumability preserved).

**`palmdef_risk/data/user_inputs.py::ingest_user_inputs`** — river handling keys
off `cfg.river_source == "user"` instead of `cfg.river_path` truthiness:
- `source == "user"` → copy `cfg.river_path` to `user_inputs/river.gpkg`
  (existing `_copy_vector` path), log override.
- else → `result["river"] = None`, river comes from `download_variables`.

**`_variables_complete(out_dir, cfg)`** — river is a required output unless
`cfg.river_source == "user"` (user file lives under `user_inputs/`, not
`variables/`). Replaces the current `getattr(cfg, "river_path", None)` check.

### 3.5 Cache

**`palmdef_risk/cache.py::variables_key`** — add `river_source` to the hashed key
inputs so switching `osm` ↔ `big` invalidates the cached variables directory.
`user` does not download into `variables/`, so it also gets its own key bucket.

---

## 4. Data flow

```
config.yaml (user_inputs.river.source)
        │
        ▼
RunConfig.river_source ──► download_variables ──► dispatch
        │                         │
        │                         ├─ big  → get_rivers_big → REST 237+257 → merge → clip → river.gpkg
        │                         ├─ osm  → get_rivers (Overpass)        → river.gpkg
        │                         └─ user → (skipped; ingest_user_inputs copies file)
        ▼
variables/river.gpkg  ──►  process/distances.py  ──►  dist_river.tif  ──► model
```

Output contract (`variables/river.gpkg`, run CRS, generic geometry) is identical
across all three sources, so no downstream code changes.

---

## 5. Error handling

- **BIG unreachable / HTTP error / timeout** after retries → `RuntimeError`,
  run stops (decision #5). No silent OSM fallback.
- **Empty result** (valid response, zero features in AOI) → write an empty
  `river.gpkg` (reuse `_create_empty_gpkg` with line geom type) and warn, matching
  how OSM "no features" is handled. This is distinct from a fetch failure.
- **CRS:** both layers requested with `outSR=4326`; reprojection to run CRS via
  the existing geopandas `to_crs` path.

---

## 6. Testing

All tests offline (monkeypatch `requests.get`), mirroring the existing OSM test
strategy. New tests in `active/tests/data/`:

1. **Pagination** — mock returns two pages (1000 + N<1000 features with
   `exceededTransferLimit` true then false); assert both pages accumulated.
2. **Merge** — mock 237 (lines) + 257 (polygons); assert merged `river.gpkg`
   contains both geometry types and is clipped to AOI.
3. **CRS** — assert output written in requested `output_crs`.
4. **Hard fail** — mock persistent failure; assert `RuntimeError` raised.
5. **Routing/dispatch** — `download_variables` calls the correct getter for each
   `river_source`; `source=user` skips download.
6. **`_variables_complete`** — river required for big/osm, not for user.
7. **Cache key** — `variables_key` differs across `river_source` values.

Run from repo root: `python -m pytest active/tests/data -k river`.

---

## 7. Files to change

| File | Change |
|------|--------|
| `active/configs/schema.json` | Add `source` enum to `user_inputs.river` |
| `active/configs/template.yaml` | Add `river.source: big` |
| `active/configs/central-kalimantan.yaml` | Set `river.source: big` |
| `active/configs/east-kotawaringin.yaml` | Set `river.source: big` |
| `active/palmdef_risk/io/config.py` | Add `river_source` field + validation |
| `active/palmdef_risk/data/variables.py` | `get_rivers_big`, `_big_query_layer`, BIG constants, dispatch in `download_variables`, `_variables_complete` |
| `active/palmdef_risk/data/user_inputs.py` | River routing keys off `river_source == "user"` |
| `active/palmdef_risk/cache.py` | Add `river_source` to `variables_key` |
| `active/tests/data/test_variables_river_big.py` | New offline test module |

---

## 8. Out of scope (YAGNI)

- KLSSNG stream-order filtering (field mostly null).
- Polygon-to-centreline skeletonization (area burn is correct for distance).
- Caching BIG data Indonesia-wide (mill-style); AOI bbox query is sufficient.
- Any change to Stage 2 / Stage 3.

## 9. Open items resolved

The three open questions in `notes/note_4.md` are resolved by the decisions:
- Layer 257 merge → **yes**, as polygons (decision #2).
- KLSSNG sufficiency → **moot**, no filter (decision #3).
- Service headers/referrer → public, no auth; a User-Agent header is sent as a
  courtesy (mirrors the Overpass UA convention).
