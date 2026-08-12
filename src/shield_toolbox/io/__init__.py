"""Run loading and format conversion.

``load_run(path) -> PermeationRun`` reads a local run directory (rig CSV or
stored parquet layout); ``fetch_run(run_id)`` downloads a stored run from
SHIELD-Data via the ``shield_data`` package; ``convert_run`` upgrades
old-generation run directories, once.
"""

from shield_toolbox.io.convert import convert_run
from shield_toolbox.io.fetch import fetch_run
from shield_toolbox.io.furnace import load_furnace_log
from shield_toolbox.io.loader import load_run, run_from_frame
from shield_toolbox.io.run import PermeationRun

__all__ = [
    "PermeationRun",
    "convert_run",
    "fetch_run",
    "load_furnace_log",
    "load_run",
    "run_from_frame",
]
