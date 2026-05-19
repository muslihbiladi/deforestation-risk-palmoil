from unittest.mock import MagicMock
from palmdef_risk.parallel import adaptive_workers, run_parallel


def _square(x):
    return x * x


def _cfg(max_workers=None, cpu_fraction=0.9):
    c = MagicMock()
    c.max_workers = max_workers
    c.cpu_fraction = cpu_fraction
    return c


def test_adaptive_workers_returns_at_least_one():
    assert adaptive_workers(1.0, _cfg()) >= 1


def test_adaptive_workers_respects_max_workers():
    assert adaptive_workers(0.001, _cfg(max_workers=2)) <= 2


def test_run_parallel_returns_correct_results():
    results = run_parallel(_square, [1, 2, 3, 4], ram_per_task_gb=0.001, cfg=_cfg())
    assert results == [1, 4, 9, 16]


def test_run_parallel_sequential_fallback():
    results = run_parallel(_square, [5, 6], ram_per_task_gb=999.0, cfg=_cfg())
    assert results == [25, 36]


def test_run_parallel_empty_tasks():
    assert run_parallel(_square, [], ram_per_task_gb=1.0, cfg=_cfg()) == []
