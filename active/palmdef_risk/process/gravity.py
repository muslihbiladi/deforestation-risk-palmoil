from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from osgeo import gdal
from scipy.signal import oaconvolve

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)


def _gaussian_kernel(sigma_px: float, radius_px: int) -> np.ndarray:
    """Area-normalized circular Gaussian kernel, zeroed beyond radius_px.

    Built in float32 so the downstream FFT convolution runs in float32
    (complex64) rather than float64 — halving the peak memory of the gravity
    surface. exp/normalize in [0, 1] are well within float32 precision.
    """
    offs = np.arange(-radius_px, radius_px + 1, dtype=np.float32)
    yy, xx = np.meshgrid(offs, offs, indexing="ij")
    d2 = xx ** 2 + yy ** 2  # squared pixel distance from kernel centre
    kernel = np.exp(-d2 / (2.0 * np.float32(sigma_px) ** 2))
    kernel[d2 > np.float32(radius_px) ** 2] = 0.0
    kernel /= kernel.sum()  # area-normalize (matches prior DC-gain-1 FFT)
    return kernel.astype(np.float32)


def _apply_gaussian_filter(
    mill_raster: Path | str,
    out_path: Path | str,
    sigma_km: float,
    radius_km: float,
) -> None:
    """Gaussian-kernel accessibility surface, truncated at radius_km.

    A_i = Σ over mills m with d(i,m) ≤ radius_km of exp(-d²/2σ²) (WORKFLOW §3.3,
    pre-registered "within 80 km"). The Gaussian kernel is zeroed beyond
    radius_km — a hard *circular* catchment — so mills past radius_km contribute
    exactly 0. A full-support kernel instead leaks its tail (≈14 % of the kernel
    mass sits beyond 2σ, e.g. at the sensitivity-sweep bandwidth σ=40 km with
    radius=80 km), silently widening the catchment as σ grows.

    Overlap-add FFT convolution (`scipy.signal.oaconvolve`): O(N log N) with
    bounded memory, and a *linear* (zero-padded) convolution — no frequency-
    domain wrap-around. The kernel is area-normalized (Σ=1), matching the
    previous DC-gain-1 FFT, so where the catchment covers the full support
    (small σ) the result matches the untruncated method; only the large-σ tail
    is removed.
    """
    ds = gdal.Open(str(mill_raster))
    # Read as float32 (mill density is 0/1) so oaconvolve runs in float32 —
    # halves the peak RAM of the FFT versus the prior float64 upcast.
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    pixel_size_m = abs(gt[1])
    ds = None

    sigma_px = (sigma_km * 1000.0) / pixel_size_m
    radius_px = int(np.ceil((radius_km * 1000.0) / pixel_size_m))

    # Circular Gaussian kernel (float32), zeroed beyond radius_px (hard catchment).
    kernel = _gaussian_kernel(sigma_px, radius_px)

    ny, nx = arr.shape
    result = np.clip(oaconvolve(arr, kernel, mode="same"), 0.0, None).astype(np.float32)

    logger.info(
        "Gaussian filter applied (truncated): sigma=%.1f km (%.0f px), "
        "radius=%.1f km (%d px), raster=%dx%d",
        sigma_km, sigma_px, radius_km, radius_px, nx, ny,
    )
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
    """OLS: A_i ~ dist_road + dist_town. Residual → out_path. Returns R²."""
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

    mask = (g_arr != g_nd) & (r_arr != r_nd) & (t_arr != t_nd)
    g = g_arr[mask]
    r = r_arr[mask]
    t = t_arr[mask]

    X = np.column_stack([np.ones(len(g)), r, t])
    beta, *_ = np.linalg.lstsq(X, g, rcond=None)
    g_hat = X @ beta
    residual_flat = g - g_hat

    ss_res = np.sum(residual_flat ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if r2 > 0.85:
        logger.warning(
            "Gravity R²=%.3f > 0.85: accessibility largely collinear with infrastructure.",
            r2,
        )

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


def _rasterize_points_numpy(points_gpkg: Path, ref_path: Path, out_path: Path) -> int:
    """Burn point features into a Float32 raster via pixel-index arithmetic.

    Bypasses gdal.RasterizeLayer, which silently produces all-zero output for
    sparse point data in some GDAL/OGR builds. Handles both Point and MultiPoint.
    Returns the number of points burned into the grid.
    """
    from osgeo import ogr

    ds = gdal.Open(str(ref_path))
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ds = None

    arr = np.zeros((ny, nx), dtype=np.float32)

    vec_ds = ogr.Open(str(points_gpkg))
    if vec_ds is None:
        logger.warning("Cannot open mill file for rasterization: %s", points_gpkg)
        return 0

    layer = vec_ds.GetLayer()
    n_burned = 0
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        sub_geoms = (
            [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]
            if geom.GetGeometryCount() > 0
            else [geom]
        )
        for pt in sub_geoms:
            col = int((pt.GetX() - gt[0]) / gt[1])
            row = int((pt.GetY() - gt[3]) / gt[5])
            if 0 <= row < ny and 0 <= col < nx:
                arr[row, col] = 1.0
                n_burned += 1
    vec_ds = None

    logger.info("Burned %d mill point(s) into density raster from %s", n_burned, points_gpkg.name)

    out_ds = gdal.GetDriverByName("GTiff").Create(
        str(out_path), nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(arr)
    out_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    out_ds = None
    return n_burned


def _raster_shape(path: Path):
    ds = gdal.Open(str(path))
    if ds is None:
        return None
    shape = (ds.RasterYSize, ds.RasterXSize)
    ds = None
    return shape


def _is_zero_raster(path: Path) -> bool:
    """Return True if raster max ≈ 0 (mill burn failed — CRS mismatch)."""
    ds = gdal.Open(str(path))
    if ds is None:
        return True
    try:
        stats = ds.GetRasterBand(1).GetStatistics(0, 1)
    except RuntimeError:
        ds = None
        return True
    ds = None
    return stats[1] < 1e-6


def _compute_gravity_for_period(
    ctx: "RunContext",
    mill_gpkg: Path,
    out_raw: Path,
    out_resid: Path,
    dist_road: Path,
    dist_town: Path,
    force: bool = False,
) -> float:
    """Shared logic: rasterize mills → Gaussian filter → orthogonalize."""
    from palmdef_risk.io.helpers import (
        get_mask_properties, rasterize_vector, reproject_vector,
    )
    d = ctx.data_dir
    ref = ctx.raw_dir / "forest" / "forest_t2.tif"
    if not ref.exists():
        ref = d / "forest_t2.tif"  # fallback
    r2 = 0.0

    if not mill_gpkg.exists():
        logger.warning("Mill file not found, skipping gravity: %s", mill_gpkg)
        return r2

    ref_shape = _raster_shape(ref)
    if out_raw.exists() and not force:
        if _raster_shape(out_raw) != ref_shape:
            logger.warning(
                "%s shape %s != reference %s — recomputing",
                out_raw.name, _raster_shape(out_raw), ref_shape,
            )
            out_raw.unlink()
            if out_resid.exists():
                out_resid.unlink()
        elif _is_zero_raster(out_raw):
            logger.warning(
                "%s max≈0 (mill CRS mismatch in prior run) — recomputing",
                out_raw.name,
            )
            out_raw.unlink()
            if out_resid.exists():
                out_resid.unlink()

    if force or not out_raw.exists():
        # Reproject mill to reference CRS if needed (handles UTM↔4326 mismatch).
        # reproject_vector returns input_path unchanged when CRS already matches.
        mask_props = get_mask_properties(str(ref))
        proj_mill = out_raw.parent / f"_mill_proj_{out_raw.stem}.gpkg"
        mill_src = Path(reproject_vector(str(mill_gpkg), str(proj_mill), mask_props["srs"]))
        tmp = out_raw.parent / f"_mill_density_tmp_{out_raw.stem}.tif"
        n_burned = _rasterize_points_numpy(mill_src, ref, tmp)
        if mill_src == proj_mill:
            proj_mill.unlink(missing_ok=True)
        if n_burned == 0:
            logger.error(
                "No mill points burned into grid — check mill GPKG extent vs reference: %s",
                mill_gpkg,
            )
            if tmp.exists():
                tmp.unlink()
            return r2
        _apply_gaussian_filter(tmp, out_raw,
                               sigma_km=ctx.config.sigma_km,
                               radius_km=ctx.config.radius_km)
        tmp.unlink(missing_ok=True)
    else:
        logger.info("skip (exists): %s", out_raw.name)

    # Always rerun orthogonalize — it is cheap (<1 s OLS) and depends on
    # dist_road and dist_town which may have changed since gravity_raw was cached.
    r2 = orthogonalize_gravity(out_raw, dist_road, dist_town, out_resid)
    return r2


def compute_gravity_accessibility(ctx: "RunContext", force: bool = False) -> float:
    """Compute gravity for modelling period (mill_t2) → data/gravity_resid.tif.

    Returns R² of the OLS orthogonalization.
    """
    d = ctx.data_dir
    return _compute_gravity_for_period(
        ctx,
        mill_gpkg=ctx.raw_dir / "mill" / "mill_t2.gpkg",
        out_raw=d / "gravity_raw.tif",
        out_resid=d / "gravity_resid.tif",
        dist_road=d / "dist_road.tif",
        dist_town=d / "dist_town.tif",
        force=force,
    )


def compute_gravity_forecast(ctx: "RunContext", force: bool = False) -> float:
    """Compute gravity for forecast period (mill_t3) → data/forecast/gravity_resid.tif.

    Returns R² of the OLS orthogonalization.
    """
    d = ctx.data_dir
    fcast = d / "forecast"
    fcast.mkdir(parents=True, exist_ok=True)
    return _compute_gravity_for_period(
        ctx,
        mill_gpkg=ctx.raw_dir / "mill" / "mill_t3.gpkg",
        out_raw=fcast / "gravity_raw.tif",
        out_resid=fcast / "gravity_resid.tif",
        dist_road=d / "dist_road.tif",
        dist_town=fcast / "dist_town.tif",
        force=force,
    )


def orthogonalize_gravity_ctx(ctx: "RunContext", force: bool = False) -> float:
    """Alias kept for backward compatibility — calls compute_gravity_accessibility."""
    return compute_gravity_accessibility(ctx, force=force)
