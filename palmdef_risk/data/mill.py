"""Download Universal Mill List (UML) palm oil mill locations.

Primary source: Trase Indonesia dataset (GeoJSON, ~950KB, uml_id field).
Fallback source: GFW ArcGIS REST API (global, more frequently updated).
Source is configured via config.mill_source: "trase" | "gfw".

NOTE: Verify Trase direct download URL before first run — the URL below
was current as of 2026-01. Check https://trase.earth/open-data/datasets/
indonesia-palm-oil-mills for the latest download link.
"""
from __future__ import annotations
from pathlib import Path
import logging

import geopandas as gpd
import requests
from shapely.geometry import box as shapely_box

from palmdef_risk.io.run import RunContext

log = logging.getLogger(__name__)

_TRASE_DOWNLOAD_URL = (
    "https://trase.earth/open-data/datasets/indonesia-palm-oil-mills/"
    "download?format=geojson"
)

_GFW_QUERY_URL = (
    "https://services.arcgis.com/nGt4QxSblgDfeJn9/arcgis/rest/services/"
    "Universal_Mill_List/FeatureServer/0/query"
)


def download_mill(ctx: RunContext) -> dict[str, Path]:
    """Download UML mills and filter to AOI. Returns {"mill": path}."""
    dst = ctx.raw_dir / "mill"
    dst.mkdir(parents=True, exist_ok=True)
    out_path = dst / "mill.gpkg"

    aoi_extent = _parse_aoi_extent(ctx.config.aoi_source, ctx.config.aoi_buffer)
    log.info("Downloading mill data from source: %s", ctx.config.mill_source)

    if ctx.config.mill_source == "trase":
        gdf = _fetch_trase()
    elif ctx.config.mill_source == "gfw":
        gdf = _fetch_gfw(aoi_extent)
    else:
        raise ValueError(f"Unknown mill source: {ctx.config.mill_source}")

    gdf = _filter_to_aoi(gdf, aoi_extent)
    log.info("  %d mills in AOI", len(gdf))

    if len(gdf) == 0:
        log.warning("No mills found in AOI. mill.gpkg will be empty.")

    if out_path.exists():
        out_path.unlink()
    gdf.to_file(out_path, driver="GPKG")
    log.info("  Written: %s", out_path)
    return {"mill": out_path}


def _fetch_trase() -> gpd.GeoDataFrame:
    log.info("  Fetching from Trase (%s)...", _TRASE_DOWNLOAD_URL)
    resp = requests.get(_TRASE_DOWNLOAD_URL, timeout=120)
    resp.raise_for_status()
    import io
    gdf = gpd.read_file(io.BytesIO(resp.content))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def _fetch_gfw(aoi_extent: tuple) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = aoi_extent
    params = {
        "where": "1=1",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": 5000,
    }
    log.info("  Querying GFW REST API...")
    resp = requests.get(_GFW_QUERY_URL, params=params, timeout=120)
    resp.raise_for_status()
    import io
    gdf = gpd.read_file(io.StringIO(resp.text))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def _filter_to_aoi(gdf: gpd.GeoDataFrame, aoi_extent: tuple) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = aoi_extent
    aoi_poly = shapely_box(xmin, ymin, xmax, ymax)
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_poly], crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    result = gpd.clip(gdf.to_crs("EPSG:4326"), aoi_gdf)
    return result[~result.geometry.is_empty & result.geometry.notna()].copy()


def _parse_aoi_extent(aoi_source: str, buffer: float) -> tuple:
    from pathlib import Path as _P
    src = _P(aoi_source)
    if src.exists():
        from osgeo import ogr, osr
        ds = ogr.Open(str(src))
        layer = ds.GetLayer()
        xmin, xmax, ymin, ymax = layer.GetExtent()
        srs_src = layer.GetSpatialRef()
        srs_4326 = osr.SpatialReference()
        srs_4326.ImportFromEPSG(4326)
        srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if srs_src:
            srs_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            if not srs_src.IsSame(srs_4326):
                ct = osr.CoordinateTransformation(srs_src, srs_4326)
                corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
                xs, ys = [], []
                for cx, cy in corners:
                    tx, ty, _ = ct.TransformPoint(cx, cy)
                    xs.append(tx)
                    ys.append(ty)
                xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        ds = None
    else:
        xmin, ymin, xmax, ymax = [float(v) for v in aoi_source.split(",")]
    return (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)
