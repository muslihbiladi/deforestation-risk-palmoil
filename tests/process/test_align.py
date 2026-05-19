import pytest
import numpy as np
from pathlib import Path
from osgeo import gdal
from palmdef_risk.io.helpers import verify_alignment


def test_rasterize_vector_produces_aligned_raster(tiny_raster, tiny_vector, tmp_path):
    from palmdef_risk.io.helpers import get_mask_properties, rasterize_vector
    mask_props = get_mask_properties(str(tiny_raster))
    out = tmp_path / "burned.tif"
    ok = rasterize_vector(str(tiny_vector), str(out), burn_value=1, mask_props=mask_props)
    assert ok
    assert out.exists()
    assert verify_alignment(str(out), str(tiny_raster))


def test_merge_plantation_raster(tmp_path):
    from palmdef_risk.process.align import merge_plantation
    import numpy as np
    from osgeo import gdal
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:4, 2:4] = 1   # industrial
    arr[6:8, 6:8] = 2   # smallholder
    from tests.conftest import _write_raster
    src = tmp_path / "plantation_raw.tif"
    _write_raster(src, arr, gt=[500000, 30, 0, 9000000, 0, -30],
                  epsg=32750, dtype=gdal.GDT_Byte, nodata=255)
    out = tmp_path / "plantation.tif"
    merge_plantation(str(src), str(out), industrial_value=1, smallholder_value=2)
    ds = gdal.Open(str(out))
    result = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    assert result[3, 3] == 1
    assert result[7, 7] == 1
    assert result[0, 9] == 0
