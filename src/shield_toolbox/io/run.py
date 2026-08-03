"""The :class:`PermeationRun` container for a recorded SHIELD run.

``PermeationRun`` holds the raw recorded data exactly as loaded from disk —
timestamps, gauge voltages (V), thermocouple voltages (mV) — plus the run
metadata. It contains no analysis logic and no unit conversions; calibrations
live in :mod:`shield_toolbox.gauges` and physics in
:mod:`shield_toolbox.analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PermeationRun:
    """A recorded SHIELD run, as loaded from disk.

    Everything shares one time axis: seconds relative to the first sample
    (``time_s[0] == 0``). Voltages are stored raw: pressure gauges in volts,
    thermocouples in millivolts.
    """

    path: Path
    run_id: str
    metadata: dict
    timestamps: np.ndarray
    """Absolute sample times, ``datetime64[ns]``."""
    time_s: np.ndarray
    """Seconds since the first sample."""
    gauge_voltages: dict[str, np.ndarray]
    """Raw gauge voltage traces in volts, keyed by gauge name."""
    gauge_locations: dict[str, str]
    """Gauge name -> ``"upstream"`` / ``"downstream"``, from the metadata."""
    thermocouple_mv: dict[str, np.ndarray] = field(default_factory=dict)
    """Raw thermocouple voltage traces in millivolts, keyed by name."""
    local_temperature_c: np.ndarray | None = None
    """Cold-junction (LabJack ambient) temperature in °C, if recorded."""
    valve_times_s: dict[str, float] = field(default_factory=dict)
    """Valve event name (e.g. ``"v3_open_time"``) -> seconds since the
    first sample."""

    @property
    def start_time(self) -> datetime | None:
        """The recorder's ``start_time`` from the metadata, if present."""
        raw = self.metadata.get("run_info", {}).get("start_time")
        if not raw:
            return None
        return datetime.fromisoformat(raw)

    @property
    def furnace_setpoint(self) -> float | None:
        """Furnace setpoint as recorded in the metadata."""
        value = self.metadata.get("run_info", {}).get("furnace_setpoint")
        return None if value is None else float(value)

    def gauges_at(self, location: str) -> list[str]:
        """Names of the gauges installed at ``"upstream"`` or ``"downstream"``."""
        return [name for name, loc in self.gauge_locations.items() if loc == location]

    def voltage(self, gauge_name: str) -> np.ndarray:
        """Raw voltage trace (V) for one gauge, by metadata gauge name."""
        try:
            return self.gauge_voltages[gauge_name]
        except KeyError:
            raise KeyError(
                f"No gauge named {gauge_name!r} in run {self.run_id}; "
                f"available: {sorted(self.gauge_voltages)}"
            ) from None
