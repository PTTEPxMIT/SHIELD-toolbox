"""Run loading and format conversion.

``load_run(path) -> PermeationRun`` reads the canonical layout
(``shield_data.csv`` + ``run_metadata.json``); ``convert_run`` upgrades
old-generation run directories to it, once.
"""

from shield_toolbox.io.convert import convert_run
from shield_toolbox.io.loader import load_run
from shield_toolbox.io.run import PermeationRun

__all__ = [
    "PermeationRun",
    "convert_run",
    "load_run",
]
