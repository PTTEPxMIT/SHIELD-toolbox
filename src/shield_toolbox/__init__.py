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

__version__ = _version("shield-toolbox")

__all__ = [
    "RigConfig",
    "__version__",
    "get_rig_config",
    "get_rig_config_for_date",
    "list_rig_versions",
]
