from __future__ import annotations
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import requests

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)

_TRASE_URL = (
    "https://trase.earth/open-data/datasets/"
    "indonesia-palm-oil-mills/download?format=geojson"
)


def _fetch_trase() -> gpd.GeoDataFrame:
    logger.info("Downloading Trase mill data ...")
    resp = requests.get(_TRASE_URL, timeout=120)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False, mode="wb") as f:
        f.write(resp.content)
        tmp = f.name
    gdf = gpd.read_file(tmp)
    Path(tmp).unlink(missing_ok=True)
    return gdf


def _filter_mills(gdf: gpd.GeoDataFrame, year: int) -> gpd.GeoDataFrame:
    """Keep mills where earliest_year_of_existence <= year OR is null."""
    col = "earliest_year_of_existence"
    if col not in gdf.columns:
        return gdf.copy()
    mask = gdf[col].isna() | (gdf[col] <= year)
    return gdf[mask].copy()


def _filter_to_aoi(
    gdf: gpd.GeoDataFrame,
    aoi_extent: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = aoi_extent
    return gdf.cx[xmin:xmax, ymin:ymax].copy()


def _aoi_extent_4326(ctx: "RunContext") -> tuple[float, float, float, float]:
    from palmdef_risk.io.helpers import aoi_bbox_4326
    return aoi_bbox_4326(ctx.config.aoi_source)


def download_mill(
    ctx: "RunContext",
    use_cache: bool = True,
) -> dict[str, Path]:
    """Download Trase mills, apply cumulative year filter, write mill_t2.gpkg + mill_t3.gpkg."""
    from palmdef_risk.cache import CacheManager
    t2 = ctx.config.forest_years[1]
    t3 = ctx.config.forest_years[2] if len(ctx.config.forest_years) > 2 else t2

    cm = CacheManager(ctx.config.cache_dir)
    cache_dir = cm.mill_dir(t2, t3)

    if use_cache and cm.mill_valid(t2, t3):
        logger.info("Mill cache hit (t2=%d, t3=%d)", t2, t3)
        raw_t2 = cache_dir / "mill_t2.gpkg"
        raw_t3 = cache_dir / "mill_t3.gpkg"
    else:
        raw = _fetch_trase()
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Cache stores AOI-unfiltered (Indonesia-wide) filtered by year only
        _filter_mills(raw, t2).to_file(str(cache_dir / "mill_t2.gpkg"), driver="GPKG")
        _filter_mills(raw, t3).to_file(str(cache_dir / "mill_t3.gpkg"), driver="GPKG")
        raw_t2 = cache_dir / "mill_t2.gpkg"
        raw_t3 = cache_dir / "mill_t3.gpkg"

    # AOI clip + reproject to run CRS
    aoi_ext = _aoi_extent_4326(ctx)
    out_dir = ctx.raw_dir / "mill"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_t2 = out_dir / "mill_t2.gpkg"
    out_t3 = out_dir / "mill_t3.gpkg"

    for src, dst in [(raw_t2, out_t2), (raw_t3, out_t3)]:
        gdf = gpd.read_file(str(src))
        clipped = _filter_to_aoi(gdf, aoi_ext)
        clipped.to_crs(ctx.config.crs).to_file(str(dst), driver="GPKG")

    logger.info("Mill files written: %s, %s", out_t2, out_t3)
    return {"mill_t2": out_t2, "mill_t3": out_t3}
