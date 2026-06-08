from palmdef_risk.data.utm import detect_utm_zones, primary_utm_zone


def test_primary_utm_kalimantan_tengah():
    bbox = (108.5, -3.0, 116.0, 2.0)
    assert primary_utm_zone(bbox) == "EPSG:32749"


def test_primary_utm_north_of_equator():
    bbox = (108.0, 1.0, 112.0, 4.0)
    assert primary_utm_zone(bbox) == "EPSG:32649"


def test_detect_single_zone():
    bbox = (109.0, -2.0, 113.0, 1.0)
    zones = detect_utm_zones(bbox)
    assert zones == ["EPSG:32749"]


def test_detect_spans_two_zones():
    bbox = (107.5, -1.0, 109.0, 1.0)
    zones = detect_utm_zones(bbox)
    assert "EPSG:32748" in zones
    assert "EPSG:32749" in zones
    assert len(zones) == 2
