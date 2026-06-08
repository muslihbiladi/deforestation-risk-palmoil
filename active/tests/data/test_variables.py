from palmdef_risk.data.variables import _WDPA_OUTPUT_NAME


def test_wdpa_output_name_is_protected():
    assert _WDPA_OUTPUT_NAME == "protected"


def test_no_pa_string_in_module():
    import palmdef_risk.data.variables as v
    import inspect
    source = inspect.getsource(v)
    # Check no hardcoded "pa.gpkg" or "pa.tif" in the source
    assert '"pa.gpkg"' not in source
    assert '"pa.tif"' not in source
    assert "'pa.gpkg'" not in source
    assert "'pa.tif'" not in source
