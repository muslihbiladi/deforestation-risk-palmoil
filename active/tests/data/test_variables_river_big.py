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
             "exceededTransferLimit": True}
    page2 = {"features": [_line_feature(1000 + i) for i in range(5)],
             "exceededTransferLimit": False}
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


from palmdef_risk.io.run import create_run


class _Cfg:
    """Duck-typed config for _variables_complete river logic."""
    def __init__(self, river_source, use_ghsl_towns=False,
                 plantation_source="user"):
        self.river_source = river_source
        self.use_ghsl_towns = use_ghsl_towns
        self.plantation_source = plantation_source


def test_variables_complete_river_required_for_big(tmp_path):
    from palmdef_risk.data.variables import _variables_complete
    for name in ("altitude.tif", "slope.tif", "protected.gpkg",
                 "road.gpkg", "town.gpkg"):
        (tmp_path / name).write_text("x")
    assert _variables_complete(tmp_path, _Cfg("big")) is False
    (tmp_path / "river.gpkg").write_text("x")
    assert _variables_complete(tmp_path, _Cfg("big")) is True


def test_variables_complete_river_not_required_for_user(tmp_path):
    from palmdef_risk.data.variables import _variables_complete
    for name in ("altitude.tif", "slope.tif", "protected.gpkg",
                 "road.gpkg", "town.gpkg"):
        (tmp_path / name).write_text("x")
    assert _variables_complete(tmp_path, _Cfg("user")) is True


def test_download_variables_dispatches_big(minimal_config_yaml, tmp_path):
    from palmdef_risk.data import variables as v
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    ctx.config.river_source = "big"

    called = {}

    def fake_big(**kw):
        called["big"] = True
        out = kw["output_dir"]
        import os
        os.makedirs(out, exist_ok=True)
        p = os.path.join(out, "river.gpkg")
        open(p, "w").close()
        return {"river": p}

    def fake_osm(**kw):
        called["osm"] = True
        return {}

    with patch.object(v, "get_srtm", return_value={}), \
         patch.object(v, "get_wdpa", return_value={}), \
         patch.object(v, "get_roads", return_value={}), \
         patch.object(v, "get_towns", return_value={}), \
         patch.object(v, "ee"), \
         patch.object(v, "get_rivers_big", side_effect=fake_big), \
         patch.object(v, "get_rivers", side_effect=fake_osm):
        v.download_variables(ctx, use_cache=False)

    assert called.get("big") is True
    assert "osm" not in called


def test_big_query_layer_continues_on_full_page_without_flag():
    from palmdef_risk.data import variables as v

    # Page 1: exactly 1000 features, NO exceededTransferLimit key at all.
    # Page 2: short page → terminates.
    page1 = {"features": [_line_feature(i) for i in range(1000)]}
    page2 = {"features": [_line_feature(1000 + i) for i in range(2)]}
    responses = [_FakeResp(page1), _FakeResp(page2)]

    with patch.object(v, "requests", create=True) as mock_requests:
        mock_requests.get = MagicMock(side_effect=responses)
        feats = v._big_query_layer(237, (110.0, -2.0, 111.0, 0.0),
                                   timeout=10, verbose=False)
    assert len(feats) == 1002
    assert mock_requests.get.call_count == 2
