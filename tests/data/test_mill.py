import pytest
import geopandas as gpd
from shapely.geometry import Point
from unittest.mock import patch
from palmdef_risk.data.mill import _filter_mills, _filter_to_aoi, download_mill
from palmdef_risk.io.run import create_run


def _make_gdf(years):
    return gpd.GeoDataFrame(
        {"earliest_year_of_existence": years,
         "geometry": [Point(110 + i * 0.5, -1.0) for i in range(len(years))]},
        crs="EPSG:4326",
    )


def test_filter_mills_keeps_null_years():
    gdf = _make_gdf([None, 2015, 2021])
    result = _filter_mills(gdf, year=2020)
    assert len(result) == 2   # null + 2015; drops 2021


def test_filter_mills_keeps_equal_year():
    gdf = _make_gdf([2020, 2021])
    result = _filter_mills(gdf, year=2020)
    assert len(result) == 1   # keeps 2020, drops 2021


def test_filter_mills_no_year_column_keeps_all():
    gdf = gpd.GeoDataFrame(
        {"mill_id": [1, 2], "geometry": [Point(110, -1), Point(111, -1)]},
        crs="EPSG:4326",
    )
    result = _filter_mills(gdf, year=2020)
    assert len(result) == 2


def test_filter_to_aoi_clips_correctly():
    gdf = _make_gdf([2015, 2015, 2015])
    # Points at 110, 110.5, 111 — clip to 109.5-111.3
    result = _filter_to_aoi(gdf, aoi_extent=(109.5, -2.0, 111.3, 0.0))
    assert len(result) == 3  # all three within extent


def test_filter_to_aoi_excludes_outside():
    gdf = _make_gdf([2015, 2015, 2015])
    # Points at 110, 110.5, 111; clip to 109.5-110.7
    result = _filter_to_aoi(gdf, aoi_extent=(109.5, -2.0, 110.7, 0.0))
    assert len(result) == 2  # 110 and 110.5; drops 111


def test_download_mill_writes_t2_and_t3(minimal_config_yaml, tmp_path):
    mock_gdf = _make_gdf([2010, 2015, 2022])
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    with patch("palmdef_risk.data.mill._fetch_trase", return_value=mock_gdf):
        with patch("palmdef_risk.data.mill._aoi_extent_4326",
                   return_value=(109.0, -3.0, 115.0, 1.0)):
            result = download_mill(ctx, use_cache=False)
    assert result["mill_t2"].exists()
    assert result["mill_t3"].exists()
    # t2=2020 (forest_years[1]): keeps 2010 + 2015 (null not in mock_gdf)
    t2 = gpd.read_file(result["mill_t2"])
    assert len(t2) == 2   # 2010 <= 2020 and 2015 <= 2020; drops 2022
