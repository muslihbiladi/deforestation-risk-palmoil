from __future__ import annotations
from pathlib import Path
import shutil
import logging

from osgeo import gdal, ogr

from palmdef_risk.io.run import RunContext

log = logging.getLogger(__name__)


def ingest_user_inputs(ctx: RunContext) -> dict[str, Path | None]:
    """Validate and copy user-provided files into the run folder.

    Must be called before any downloads. Fails fast if any required
    file is missing, unreadable, or missing a CRS definition.

    Returns dict with keys: peatland, hgu, plantation_t2, plantation_t3, river.
    plantation_t3 is None if not configured; river is None unless river.source=user
    (otherwise river is downloaded in Stage 1).
    """
    dst = ctx.raw_dir / "user_inputs"
    cfg = ctx.config
    result: dict[str, Path | None] = {}

    result["peatland"] = _copy_vector(cfg.peatland_path, dst, "peatland")
    result["hgu"] = _copy_vector(cfg.hgu_path, dst, "hgu")

    if cfg.plantation_t2:
        result["plantation_t2"] = _copy_raster(cfg.plantation_t2, dst, "plantation_t2")
    else:
        log.info("plantation.t2 not configured — dist_plantation_edge will be skipped")
        result["plantation_t2"] = None

    if cfg.plantation_t3:
        result["plantation_t3"] = _copy_raster(cfg.plantation_t3, dst, "plantation_t3")
    else:
        log.info("plantation.t3 not configured — dist_plantation_edge_forecast will be skipped")
        result["plantation_t3"] = None

    if cfg.river_source == "user":
        if not cfg.river_path:
            raise ValueError(
                "user_inputs.river.path is required when river.source: 'user'"
            )
        result["river"] = _copy_vector(cfg.river_path, dst, "river")
        log.info("User-supplied river (source=user) will override downloaded river")
    else:
        log.info("river.source=%s — river will be downloaded in Stage 1",
                 cfg.river_source)
        result["river"] = None

    log.info("User inputs ingested to %s", dst)
    return result


def _copy_vector(src_path: str, dst_dir: Path, label: str) -> Path:
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"{label}: file not found: {src}")
    ds = ogr.Open(str(src))
    if ds is None:
        raise ValueError(f"{label}: GDAL/OGR cannot open: {src}")
    layer = ds.GetLayer()
    if layer.GetSpatialRef() is None:
        raise ValueError(f"CRS undefined in {label}: {src}")
    ds = None
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    log.info("  %s -> %s", label, dst)
    return dst


def _copy_raster(src_path: str, dst_dir: Path, label: str) -> Path:
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"{label}: file not found: {src}")
    ds = gdal.Open(str(src))
    if ds is None:
        raise ValueError(f"{label}: GDAL cannot open: {src}")
    ds = None
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    log.info("  %s -> %s", label, dst)
    return dst
