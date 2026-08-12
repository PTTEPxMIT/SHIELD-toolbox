"""Run loading: ``load_run(path) -> PermeationRun`` across all raw formats."""

from shield_toolbox.io.loader import detect_run_format, load_run
from shield_toolbox.io.run import PermeationRun, RunFormat

__all__ = [
    "PermeationRun",
    "RunFormat",
    "detect_run_format",
    "load_run",
]
