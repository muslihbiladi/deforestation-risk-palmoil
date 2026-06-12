from unittest.mock import patch, MagicMock
import geopandas as gpd
import pytest


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _line_feature(oid, x0=110.0):
    return {
        "type": "Feature",
        "properties": {"OBJECTID": oid},
        "geometry": {"type": "LineString",
                     "coordinates": [[x0, -1.0], [x0 + 0.01, -1.01]]},
    }


def test_big_query_layer_paginates():
    from palmdef_risk.data import variables as v

    page1 = {"features": [_line_feature(i) for i in range(1000)],
             "properties": {"exceededTransferLimit": True}}
    page2 = {"features": [_line_feature(1000 + i) for i in range(5)],
             "properties": {"exceededTransferLimit": False}}
    responses = [_FakeResp(page1), _FakeResp(page2)]

    with patch.object(v, "requests", create=True) as mock_requests:
        mock_requests.get = MagicMock(side_effect=responses)
        feats = v._big_query_layer(237, (110.0, -2.0, 111.0, 0.0),
                                   timeout=10, verbose=False)
    assert len(feats) == 1005
    assert mock_requests.get.call_count == 2


def test_big_query_layer_hard_fails():
    from palmdef_risk.data import variables as v
    with patch.object(v, "requests", create=True) as mock_requests:
        mock_requests.get = MagicMock(side_effect=Exception("boom"))
        with pytest.raises(RuntimeError):
            v._big_query_layer(237, (110.0, -2.0, 111.0, 0.0),
                               timeout=1, verbose=False)


def _poly_feature(oid, x0=110.2):
    return {
        "type": "Feature",
        "properties": {"OBJECTID": oid},
        "geometry": {"type": "Polygon",
                     "coordinates": [[[x0, -1.0], [x0 + 0.02, -1.0],
                                      [x0 + 0.02, -1.02], [x0, -1.02],
                                      [x0, -1.0]]]},
    }


def test_get_rivers_big_merges_layers(tmp_path):
    from palmdef_risk.data import variables as v

    lines = [_line_feature(i, x0=110.1) for i in range(3)]
    polys = [_poly_feature(100 + i, x0=110.2) for i in range(2)]

    def fake_query(layer_id, bbox, timeout=180, verbose=True):
        return lines if layer_id == v._BIG_RIVER_LINE_LAYER else polys

    aoi = (110.0, -1.5, 110.5, -0.5)  # bbox AOI covering all features
    with patch.object(v, "_big_query_layer", side_effect=fake_query):
        out = v.get_rivers_big(aoi, output_dir=str(tmp_path),
                               output_crs="EPSG:32749", verbose=False)

    gpkg = out["river"]
    gdf = gpd.read_file(gpkg)
    assert len(gdf) == 5  # 3 lines + 2 polygons merged
    geom_types = set(gdf.geometry.geom_type)
    assert {"LineString"}.issubset(geom_types)
    assert "Polygon" in geom_types
    assert gdf.crs.to_epsg() == 32749  # reprojected to requested CRS


def test_get_rivers_big_empty_writes_empty_gpkg(tmp_path):
    from palmdef_risk.data import variables as v
    with patch.object(v, "_big_query_layer", return_value=[]):
        out = v.get_rivers_big((110.0, -1.5, 110.5, -0.5),
                               output_dir=str(tmp_path), verbose=False)
    assert out["river"].endswith("river.gpkg")
    import os
    assert os.path.exists(out["river"])
    assert len(gpd.read_file(out["river"])) == 0
