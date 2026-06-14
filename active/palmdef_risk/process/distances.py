from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from osgeo import gdal

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)


def _raster_shape(path: Path):
    ds = gdal.Open(str(path))
    if ds is None:
        return None
    s = (ds.RasterYSize, ds.RasterXSize)
    ds = None
    return s


def _needs_recompute(out_path: Path, ref_shape: tuple) -> bool:
    """Return True if out_path is missing or has wrong dimensions."""
    if not out_path.exists():
        return True
    shape = _raster_shape(out_path)
    if shape != ref_shape:
        logger.warning(
            "%s shape %s != reference %s — recomputing",
            out_path.name, shape, ref_shape,
        )
        out_path.unlink()
        return True
    return False


def _proximity_from_raster(src_path: Path, out_path: Path, target_value: int = 0) -> None:
    """Compute GDAL proximity (metres) to pixels where value == target_value.

    The binary target mask is built directly into the MEM Byte raster that
    ComputeProximity consumes, streaming the source block-by-block. The full
    source array and a separate full uint8 mask never coexist with the MEM
    copy (the prior code held all three); only the MEM mask — which proximity
    requires whole — stays resident. The mask is pixel-identical to
    ``(arr == target_value)``, so the proximity output is unchanged.
    """
    ds = gdal.Open(str(src_path))
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    nx, ny = band.XSize, band.YSize
    bx, by = band.GetBlockSize()

    src_ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    src_ds.SetGeoTransform(gt)
    src_ds.SetProjection(proj)
    mask_band = src_ds.GetRasterBand(1)
    for yoff in range(0, ny, by):
        ywin = min(by, ny - yoff)
        for xoff in range(0, nx, bx):
            xwin = min(bx, nx - xoff)
            blk = band.ReadAsArray(xoff, yoff, xwin, ywin)
            mask_band.WriteArray((blk == target_value).astype(np.uint8), xoff, yoff)
    ds = None

    out_ds = gdal.GetDriverByName("GTiff").Create(
        str(out_path), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    gdal.ComputeProximity(mask_band, out_ds.GetRasterBand(1),
                          options=["DISTUNITS=GEO"])
    out_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    out_ds = None
    src_ds = None


def _resample_to_ref(src: Path, ref: Path, out: Path) -> None:
    """Resample src to exactly match ref's extent, resolution and CRS."""
    ref_ds = gdal.Open(str(ref))
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    nx, ny = ref_ds.RasterXSize, ref_ds.RasterYSize
    ref_ds = None
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * nx
    ymin = ymax + gt[5] * ny
    gdal.Warp(
        str(out), str(src),
        options=gdal.WarpOptions(
            format="GTiff",
            dstSRS=proj,
            outputBounds=[xmin, ymin, xmax, ymax],
            width=nx, height=ny,
            resampleAlg=gdal.GRA_NearestNeighbour,
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
        ),
    )


def _proximity_from_vector(vec_path: Path, ref_path: Path, out_path: Path) -> None:
    """Rasterize vector then compute proximity from burned pixels."""
    from palmdef_risk.io.helpers import get_mask_properties, rasterize_vector, reproject_vector
    mask_props = get_mask_properties(str(ref_path))
    # Reproject vector to reference CRS if needed (handles UTM↔4326 mismatch).
    # reproject_vector returns input_path unchanged when CRS already matches.
    proj_vec = out_path.parent / f"_vec_proj_{out_path.stem}.gpkg"
    vec_src = Path(reproject_vector(str(vec_path), str(proj_vec), mask_props["srs"]))
    tmp = out_path.parent / f"_vec_tmp_{out_path.stem}.tif"
    rasterize_vector(str(vec_src), str(tmp), burn_value=1, mask_props=mask_props)
    if vec_src == proj_vec:
        proj_vec.unlink(missing_ok=True)
    _proximity_from_raster(tmp, out_path, target_value=1)
    tmp.unlink(missing_ok=True)


def compute_all_distances(ctx: "RunContext") -> None:
    """Compute distance rasters for modelling (data/) and forecasting (data/forecast/).

    Modelling distances (data/):
        dist_edge, dist_defor, dist_road, dist_river, dist_town,
        dist_plantation_edge

    Forecast distances (data/forecast/):
        dist_edge, dist_defor, dist_town, dist_plantation_edge

    dist_town source priority (both periods):
        1. raw_dir/variables/town.gpkg            (OSM points)
        2. raw_dir/variables/ghsl_built_t2.tif    (model period — proximity to value=1)
           raw_dir/variables/ghsl_built_t3.tif    (forecast period)

    dist_mill is NOT computed here (used only as gravity input).
    """
    d = ctx.data_dir
    fcast = d / "forecast"
    fcast.mkdir(parents=True, exist_ok=True)

    vec_dir = ctx.raw_dir / "variables"
    ref = d / "forest_t2.tif"
    ref_shape = _raster_shape(ref)
    tasks = []

    # ── Modelling distances (data/) ───────────────────────────────────────────

    for name, src, tgt in [
        ("dist_edge",  "forest_t2.tif", 0),
        ("dist_defor", "fcc12.tif",     0),
    ]:
        src_path = d / src
        out_path = d / f"{name}.tif"
        if src_path.exists() and _needs_recompute(out_path, ref_shape):
            tasks.append(("raster", src_path, out_path, tgt))

    for name, ras_name in [
        ("dist_road",  "road.tif"),
        ("dist_river", "river.tif"),
    ]:
        ras_path = d / ras_name
        out_path = d / f"{name}.tif"
        if ras_path.exists() and _needs_recompute(out_path, ref_shape):
            tasks.append(("raster", ras_path, out_path, 1))  # target_value=1 = presence
        elif not ras_path.exists():
            # Fallback: re-rasterize from vector if binary raster missing
            vec_path = vec_dir / (name.replace("dist_", "") + ".gpkg")
            if vec_path.exists() and _needs_recompute(out_path, ref_shape):
                tasks.append(("vector", vec_path, out_path, ref))

    # dist_town (model): OSM points preferred, fallback to GHSL t2 raster
    dist_town = d / "dist_town.tif"
    if _needs_recompute(dist_town, ref_shape):
        town_vec = vec_dir / "town.gpkg"
        town_ras = vec_dir / "ghsl_built_t2.tif"
        if town_vec.exists():
            tasks.append(("vector", town_vec, dist_town, ref))
        elif town_ras.exists():
            # Resample GHSL (100m) to reference resolution (30m) before proximity
            town_30m = d / "intermediate" / "ghsl_built_t2_30m.tif"
            tasks.append(("ghsl", town_ras, dist_town, ref, town_30m))
        else:
            logger.warning("No town source found for dist_town — skipping")

    # dist_plantation_edge
    plant_tif = d / "plantation.tif"
    dist_plant = d / "dist_plantation_edge.tif"
    if plant_tif.exists() and _needs_recompute(dist_plant, ref_shape):
        tasks.append(("raster", plant_tif, dist_plant, 1))

    # ── Forecast distances (data/forecast/) ───────────────────────────────────

    for name, src, tgt in [
        ("dist_edge",  "forest_t3.tif", 0),
        ("dist_defor", "fcc23.tif",     0),
    ]:
        src_path = d / src
        out_path = fcast / f"{name}.tif"
        if src_path.exists() and _needs_recompute(out_path, ref_shape):
            tasks.append(("raster", src_path, out_path, tgt))

    # Forecast plantation edge distance (t3 plantation raster), model name under forecast/
    plant_t3 = d / "plantation_t3.tif"
    dist_plant_fc = fcast / "dist_plantation_edge.tif"
    if plant_t3.exists() and _needs_recompute(dist_plant_fc, ref_shape):
        tasks.append(("raster", plant_t3, dist_plant_fc, 1))

    # dist_town (forecast): OSM points (static) or GHSL t3 raster
    dist_town_fc = fcast / "dist_town.tif"
    if _needs_recompute(dist_town_fc, ref_shape):
        town_vec = vec_dir / "town.gpkg"
        town_ras = vec_dir / "ghsl_built_t3.tif"
        if town_vec.exists():
            tasks.append(("vector", town_vec, dist_town_fc, ref))
        elif town_ras.exists():
            town_30m = d / "intermediate" / "ghsl_built_t3_30m.tif"
            tasks.append(("ghsl", town_ras, dist_town_fc, ref, town_30m))
        else:
            logger.warning("No town source found for forecast/dist_town — skipping")

    from palmdef_risk.parallel import run_parallel
    run_parallel(_dist_worker, tasks,
                 ram_per_task_gb=ctx.config.ram_per_dist_gb, cfg=ctx.config,
                 desc="Computing distance rasters")
    logger.info("All distance rasters computed")


def _dist_worker(task: tuple) -> None:
    kind = task[0]
    if kind == "raster":
        _, src, out, target = task
        _proximity_from_raster(src, out, target_value=target)
    elif kind == "vector":
        _, src, out, ref = task
        _proximity_from_vector(src, ref, out)
    elif kind == "ghsl":
        _, src, out, ref, tmp_30m = task
        _resample_to_ref(src, ref, tmp_30m)
        _proximity_from_raster(tmp_30m, out, target_value=1)
        tmp_30m.unlink(missing_ok=True)
