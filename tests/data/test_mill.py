import pytest
from unittest.mock import patch
from pathlib import Path
from palmoil_risk.io.run import create_run
from palmoil_risk.data.mill import download_mill, _filter_to_aoi


def test_filter_to_aoi_returns_geodataframe(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point
    mills = gpd.GeoDataFrame(
        {"uml_id": [1, 2, 3], "geometry": [
            Point(110.0, -1.0),
            Point(115.0, -2.0),
            Point(120.0, -3.0),
        ]},
        crs="EPSG:4326",
    )
    result = _filter_to_aoi(mills, aoi_extent=(109.0, -2.5, 116.0, 0.5))
    assert len(result) == 2
    assert "uml_id" in result.columns


def test_filter_to_aoi_empty_returns_empty(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point
    mills = gpd.GeoDataFrame(
        {"uml_id": [1], "geometry": [Point(130.0, -5.0)]},
        crs="EPSG:4326",
    )
    result = _filter_to_aoi(mills, aoi_extent=(100.0, -1.0, 105.0, 1.0))
    assert len(result) == 0


def test_download_mill_writes_gpkg(minimal_config_yaml, tmp_path):
    """Mock HTTP response; verify GPKG is written to raw/mill/."""
    import geopandas as gpd
    from shapely.geometry import Point

    mock_gdf = gpd.GeoDataFrame(
        {"uml_id": ["ID001"], "mill_name": ["Test Mill"],
         "geometry": [Point(112.5, -1.5)]},
        crs="EPSG:4326",
    )

    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")

    with patch("palmoil_risk.data.mill._parse_aoi_extent", return_value=(111.0, -3.0, 114.0, 0.0)):
        with patch("palmoil_risk.data.mill._fetch_trase", return_value=mock_gdf):
            result = download_mill(ctx)

    assert result["mill"].exists()
    assert result["mill"].suffix == ".gpkg"
    loaded = gpd.read_file(result["mill"])
    assert len(loaded) >= 1
