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
from shield_toolbox.io import (
    PermeationRun,
    convert_run,
    fetch_run,
    load_furnace_log,
    load_run,
)

__version__ = _version("shield-toolbox")

from shield_toolbox.campaign import (  # noqa: E402  (needs processing)
    arrhenius,
    load_results,
)
from shield_toolbox.processing import (  # noqa: E402  (needs __version__)
    ProcessedRun,
    SampleInfo,
    process_run,
)

__all__ = [
    "PermeationRun",
    "ProcessedRun",
    "RigConfig",
    "SampleInfo",
    "__version__",
    "arrhenius",
    "convert_run",
    "fetch_run",
    "get_rig_config",
    "get_rig_config_for_date",
    "list_rig_versions",
    "load_furnace_log",
    "load_results",
    "load_run",
    "process_run",
]
