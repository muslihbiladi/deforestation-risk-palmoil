def test_package_imports_as_palmdef_risk():
    import palmdef_risk
    from palmdef_risk.io.config import RunConfig
    from palmdef_risk.io.run import RunContext, create_run, load_run
    assert True


def test_old_package_name_not_importable():
    import importlib
    import sys
    # Remove from cache if accidentally imported earlier
    sys.modules.pop("palmoil_risk", None)
    spec = importlib.util.find_spec("palmoil_risk")
    assert spec is None, "palmoil_risk should not be importable after rename"
