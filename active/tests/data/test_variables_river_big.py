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
