from __future__ import annotations
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

logger = logging.getLogger(__name__)


def adaptive_workers(ram_per_task_gb: float, cfg) -> int:
    """Return worker count bounded by available RAM, CPU fraction, and max_workers."""
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        ram_workers = max(1, int(avail_gb / ram_per_task_gb))
    except ImportError:
        ram_workers = 4

    cpu_count = os.cpu_count() or 1
    cpu_fraction = getattr(cfg, "cpu_fraction", 0.9)
    cpu_workers = max(1, math.floor(cpu_count * cpu_fraction))

    n = min(ram_workers, cpu_workers)
    max_w = getattr(cfg, "max_workers", None)
    if max_w:
        n = min(n, max_w)
    return max(1, n)


def run_parallel(fn: Callable, tasks: list, ram_per_task_gb: float, cfg) -> list[Any]:
    """Run fn(task) for each task. Falls back to sequential when workers=1."""
    if not tasks:
        return []
    n = adaptive_workers(ram_per_task_gb, cfg)
    if n == 1:
        return [fn(t) for t in tasks]
    results: list[Any] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(fn, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    logger.info("run_parallel: %d tasks on %d workers", len(tasks), n)
    return results
