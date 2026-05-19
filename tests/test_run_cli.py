import sys
import subprocess
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def test_run_cli_dry_run_succeeds(tmp_path):
    """--dry-run should exit 0 and not execute any notebooks."""
    config = _ROOT / "tests" / "conftest_template.yaml"
    if not config.exists():
        # Use the bundled template; it has placeholder paths so we only test
        # that the CLI arg parsing and dry-run path work.
        config = _ROOT / "configs" / "template.yaml"

    result = subprocess.run(
        [sys.executable, str(_ROOT / "run.py"),
         "--config", str(config),
         "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "dry-run" in result.stdout


def test_run_cli_missing_config_exits_nonzero():
    result = subprocess.run(
        [sys.executable, str(_ROOT / "run.py"),
         "--config", "nonexistent.yaml",
         "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
