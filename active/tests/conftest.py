import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Fix PROJ database conflict (must happen before any osgeo import).
#
# On Windows machines that also have PostgreSQL/PostGIS installed, PROJ_LIB
# may be set to a stale postgis proj directory (DATABASE.LAYOUT.VERSION.MINOR=2)
# which causes GetSpatialRef() to return None.  We override PROJ_LIB and
# PROJ_DATA to point at the palmdef-risk PROJ database before osgeo is imported.
# ---------------------------------------------------------------------------
# On Windows, conda envs have python.exe at <prefix>\python.exe (no bin/ dir).
# On Linux/Mac, it's at <prefix>/bin/python, so .parent.parent is needed.
# Try both to handle both platforms.
_exe = Path(sys.executable)
for _conda_prefix in [_exe.parent, _exe.parent.parent]:
    for _candidate in [
        _conda_prefix / "Library" / "share" / "proj",
        _conda_prefix / "share" / "proj",
    ]:
        if (_candidate / "proj.db").exists():
            os.environ["PROJ_LIB"] = str(_candidate)
            os.environ["PROJ_DATA"] = str(_candidate)
            break
    else:
        continue
    break

import pytest
import numpy as np
import yaml
from osgeo import gdal, ogr, osr


def _write_raster(path: Path, arr: np.ndarray, gt: list, epsg: int,
                  dtype=None, nodata=None) -> Path:
    if dtype is None:
        dtype = gdal.GDT_Byte
    driver = gdal.GetDriverByName("GTiff")
    ny, nx = arr.shape
    ds = driver.Create(str(path), nx, ny, 1, dtype)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(arr)
    if nodata is not None:
        ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds.FlushCache()
    ds = None
    return path


def _write_vector(path: Path, epsg: int = 32750) -> Path:
    driver = ogr.GetDriverByName("GPKG")
    if path.exists():
        driver.DeleteDataSource(str(path))
    ds = driver.CreateDataSource(str(path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    layer = ds.CreateLayer("layer", srs, ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for pt in [(500000, 8999700), (500300, 8999700),
               (500300, 9000000), (500000, 9000000), (500000, 8999700)]:
        ring.AddPoint(*pt)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(poly)
    layer.CreateFeature(feat)
    ds.FlushCache()
    ds = None
    return path


@pytest.fixture
def write_raster():
    """Injectable fixture returning the _write_raster helper callable."""
    return _write_raster


@pytest.fixture
def write_vector():
    """Injectable fixture returning the _write_vector helper callable."""
    return _write_vector


@pytest.fixture
def tiny_raster(tmp_path) -> Path:
    """10x10 Byte raster, UTM 50S (EPSG:32750), 30m pixels, NoData=255."""
    arr = np.ones((10, 10), dtype=np.uint8)
    arr[0, :] = 255
    arr[:, 0] = 255
    return _write_raster(
        tmp_path / "tiny.tif", arr,
        gt=[500000, 30, 0, 9000000, 0, -30], epsg=32750,
        dtype=gdal.GDT_Byte, nodata=255,
    )


@pytest.fixture
def tiny_vector(tmp_path) -> Path:
    """Single-polygon GPKG in UTM 50S."""
    return _write_vector(tmp_path / "tiny.gpkg", epsg=32750)


@pytest.fixture
def user_input_files(tmp_path) -> dict:
    """Minimal user-provided input files for testing."""
    raw = tmp_path / "raw_data"
    raw.mkdir()
    arr = np.ones((10, 10), dtype=np.uint8)
    arr[0, :] = 255
    gt = [500000, 30, 0, 9000000, 0, -30]
    peat = _write_vector(raw / "peatland.gpkg", epsg=32750)
    hgu = _write_vector(raw / "hgu.gpkg", epsg=32750)
    plant_t2 = _write_raster(raw / "plantation_t2.tif", arr, gt, 32750,
                              dtype=gdal.GDT_Byte, nodata=255)
    return {"peatland": peat, "hgu": hgu, "plantation_t2": plant_t2}


@pytest.fixture
def minimal_config_yaml(tmp_path, user_input_files) -> Path:
    (tmp_path / "configs").mkdir(exist_ok=True)
    cfg = {
        "run": {"project": "test_proj", "area": "test_area", "task": "test"},
        "aoi": {"source": str(user_input_files["hgu"]), "buffer": 0.0},
        "crs": "EPSG:32750",
        "cache_dir": "cache/",
        "forest": {"source": "tmf", "years": [2015, 2020, 2024], "perc": 75},
        "variables": {"use_ghsl_towns": False, "ghsl_years": None, "osm_timeout": 180},
        "user_inputs": {
            "peatland": {"path": str(user_input_files["peatland"]), "type": "binary"},
            "hgu": {"path": str(user_input_files["hgu"])},
            "plantation": {"t2": str(user_input_files["plantation_t2"]),
                           "t3": None, "industrial_value": 1, "smallholder_value": 2},
        },
        "mill": {"source": "trase", "path": None},
        "process": {
            "gravity": {"sigma_km": 25.0, "radius_km": 80.0},
            "sensitivity": {"sigmas_km": [15.0, 25.0, 40.0]},
        },
        "model": {
            "variants": ["A", "B"], "nsamp": 10000, "csize": 10,
            "Vbeta": 1000, "burnin": 100, "mcmc": 100, "thin": 1, "seed": 42,
        },
        "parallel": {"max_workers": None, "cpu_fraction": 0.9,
                     "ram_per_dist_gb": 0.5, "ram_per_icar_gb": 1.0,
                     "ram_per_predict_gb": 0.75},
        "output": {"project_future": False, "projection_year": 2035},
    }
    path = tmp_path / "configs" / "test.yaml"
    path.write_text(yaml.dump(cfg))
    return path
