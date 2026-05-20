from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from osgeo import gdal

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

    for name, src, tgt in [
        ("dist_edge",           "forest_t2.tif", 0),
        ("dist_defor",          "fcc12.tif",      0),
        ("dist_edge_forecast",  "forest_t3.tif", 0),
        ("dist_defor_forecast", "fcc23.tif",      0),
    ]:
        src_path = d / src
        out_path = d / f"{name}.tif"
        if src_path.exists() and not out_path.exists():
            tasks.append(("raster", src_path, out_path, tgt))

    for name, vec in [
        ("dist_road",            "road.gpkg"),
        ("dist_river",           "river.gpkg"),
        ("dist_town",            "town.gpkg"),
        ("dist_plantation_edge", "plantation.tif"),
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
