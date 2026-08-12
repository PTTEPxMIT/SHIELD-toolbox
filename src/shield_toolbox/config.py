"""Versioned SHIELD rig configurations.

Every rig-specific hardware constant lives here, one frozen ``RigConfig`` per
hardware version of the rig. Analysis code must never hard-code rig constants —
take a ``RigConfig`` argument (or call :func:`get_rig_config`) instead.

The rig is periodically upgraded (e.g. changed downstream volume and sample
area). When that happens, fill in the next draft config with the measured
values, set its ``valid_from`` commissioning date, and leave earlier versions
untouched so historical runs keep analyzing with the hardware they were
recorded on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from uncertainties import UFloat, ufloat

from shield_toolbox.gauges import CVM211, WGM701, Baratron626D, TypeKThermocouple


@dataclass(frozen=True)
class GaugeSpec:
    """A pressure gauge installed on the rig, with its calibration."""

    name: str
    model: str
    location: str  # "upstream" or "downstream"
    calibration: object
    full_scale_torr: float | None = None


@dataclass(frozen=True)
class RigConfig:
    """Hardware constants for one version of the SHIELD rig."""

    version: str
    valid_from: date | None
    """First run date this configuration applies to; None while draft."""
    downstream_volume_m3: UFloat
    sample_diameter_m: float
    """Diameter of the exposed sample area set by the fitting."""
    ambient_temperature_K: float
    v1_v2_split_ratio: UFloat
    """Fraction of the downstream volume on the hot (V1) side, used by the
    Takaishi–Sensui thermal-transpiration correction."""
    furnace_setpoint_offset_K: float
    """Sample temperature minus furnace setpoint, used as a fallback for old
    runs recorded without a sample thermocouple."""
    gauges: tuple[GaugeSpec, ...]
    thermocouple: TypeKThermocouple
    notes: str = ""

    @property
    def sample_area_m2(self) -> float:
        return 0.25 * math.pi * self.sample_diameter_m**2

    @property
    def is_draft(self) -> bool:
        return self.valid_from is None

    def gauge(self, name: str) -> GaugeSpec:
        for spec in self.gauges:
            if spec.name == name:
                return spec
        raise KeyError(
            f"No gauge named {name!r} in rig {self.version}; "
            f"available: {[g.name for g in self.gauges]}"
        )


SHIELD_V1 = RigConfig(
    version="v1",
    valid_from=date(2000, 1, 1),  # covers all runs recorded before the upgrade
    downstream_volume_m3=ufloat(7.9e-5, 9.8e-6),
    sample_diameter_m=0.0155,
    ambient_temperature_K=300.0,
    v1_v2_split_ratio=ufloat(0.35, 0.1),
    furnace_setpoint_offset_K=-18.0,
    gauges=(
        GaugeSpec(
            name="baratron_upstream",
            model="MKS Baratron 626D",
            location="upstream",
            calibration=Baratron626D(full_scale_torr=1000.0),
            full_scale_torr=1000.0,
        ),
        GaugeSpec(
            name="baratron_downstream",
            model="MKS Baratron 626D",
            location="downstream",
            calibration=Baratron626D(full_scale_torr=1.0),
            full_scale_torr=1.0,
        ),
        GaugeSpec(
            name="wasp_downstream",
            model="Inficon WGM701",
            location="downstream",
            calibration=WGM701(),
        ),
        GaugeSpec(
            name="cvm211_upstream",
            model="InstruTech CVM211",
            location="upstream",
            calibration=CVM211(),
        ),
    ),
    thermocouple=TypeKThermocouple(),
    notes="Original SHIELD rig as commissioned.",
)

SHIELD_V2 = RigConfig(
    version="v2",
    valid_from=None,  # DRAFT — set to the commissioning date at upgrade time
    downstream_volume_m3=ufloat(7.9e-5, 9.8e-6),  # PLACEHOLDER: measure
    sample_diameter_m=0.0155,  # PLACEHOLDER: measure
    ambient_temperature_K=300.0,
    v1_v2_split_ratio=ufloat(0.35, 0.1),  # PLACEHOLDER: re-estimate
    furnace_setpoint_offset_K=0.0,
    gauges=SHIELD_V1.gauges,
    thermocouple=TypeKThermocouple(),
    notes=(
        "DRAFT for the upcoming rig upgrade. Downstream volume, sample "
        "diameter, and volume split are placeholders copied from v1 — "
        "measure and update them, then set valid_from."
    ),
)

_RIG_VERSIONS: dict[str, RigConfig] = {
    "v1": SHIELD_V1,
    "v2": SHIELD_V2,
}


def list_rig_versions() -> tuple[str, ...]:
    """Known rig version identifiers, oldest first."""
    return tuple(_RIG_VERSIONS)


def get_rig_config(version: str = "v1") -> RigConfig:
    """Return the rig configuration for an explicit version identifier."""
    try:
        return _RIG_VERSIONS[version]
    except KeyError:
        raise KeyError(
            f"Unknown rig version {version!r}; known versions: {list(_RIG_VERSIONS)}"
        ) from None


def get_rig_config_for_date(run_date: date) -> RigConfig:
    """Return the rig configuration in service on ``run_date``.

    Draft configurations (``valid_from is None``) are never selected.
    """
    candidates = [
        cfg
        for cfg in _RIG_VERSIONS.values()
        if cfg.valid_from is not None and cfg.valid_from <= run_date
    ]
    if not candidates:
        raise ValueError(
            f"No rig configuration is valid on {run_date.isoformat()}; "
            f"the earliest starts {min(c.valid_from for c in _RIG_VERSIONS.values() if c.valid_from)}"
        )
    return max(candidates, key=lambda cfg: cfg.valid_from)
