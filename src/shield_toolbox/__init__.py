"""Analysis toolbox for the SHIELD gas-driven permeation rig.

Extracts permeability, diffusivity, and solubility of materials and coatings
from time-lag permeation measurements.
"""

from importlib.metadata import version as _version

from shield_toolbox.config import (
    RigConfig,
    get_rig_config,
    get_rig_config_for_date,
    list_rig_versions,
)
from shield_toolbox.io import PermeationRun, RunFormat, detect_run_format, load_run

__version__ = _version("shield-toolbox")

__all__ = [
    "PermeationRun",
    "RigConfig",
    "RunFormat",
    "__version__",
    "detect_run_format",
    "get_rig_config",
    "get_rig_config_for_date",
    "list_rig_versions",
    "load_run",
]
