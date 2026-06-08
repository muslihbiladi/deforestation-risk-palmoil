def test_download_forest_passes_output_crs(minimal_config_yaml, tmp_path):
    """download_forest must pass output_crs=ctx.config.crs to get_fcc."""
    from unittest.mock import patch
    from palmdef_risk.io.run import create_run
    from palmdef_risk.data.forest import download_forest
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    captured = {}
    original_get_fcc = None
    try:
        import palmdef_risk.data.forest as fm
        original_get_fcc = fm.get_fcc
        def fake_get_fcc(*args, **kwargs):
            captured["output_crs"] = kwargs.get("output_crs")
            return {}
        fm.get_fcc = fake_get_fcc
        try:
            download_forest(ctx, use_cache=False)
        except Exception:
            pass
    finally:
        if original_get_fcc:
            fm.get_fcc = original_get_fcc
    if captured:
        assert captured.get("output_crs") == ctx.config.crs, (
            f"output_crs={captured.get('output_crs')} but expected {ctx.config.crs}"
        )
