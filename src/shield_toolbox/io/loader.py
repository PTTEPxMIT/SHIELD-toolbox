"""Load a recorded SHIELD run from disk.

Only the canonical layout is supported — ``shield_data.csv`` +
``run_metadata.json``, as written by the current DAS recorder. Old-generation
directories (``pressure_gauge_data.csv`` [+ ``thermocouple_data.csv``]) must
be converted once with :func:`shield_toolbox.io.convert_run` (or
``scripts/convert_runs.py``).

Column headers carry units on disk (``WGM701_Voltage (V)``,
``type K thermocouple_Voltage (mV)``, ``Local_temperature (C)``); some tools
re-export them with sanitized names (``WGM701_Voltage_V``), so both spellings
are accepted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from shield_toolbox.io.convert import (
    CANONICAL_CSV,
    LEGACY_PRESSURE_CSV,
    METADATA_FILENAME,
)
from shield_toolbox.io.run import PermeationRun

_TIMESTAMP_COLUMN = "RealTimestamp"
_GAUGE_VOLTAGE_RE = re.compile(r"^(?P<name>.+?)_Voltage(?: \(V\)|_V)$")
_TC_VOLTAGE_RE = re.compile(r"^(?P<name>.+?)_Voltage(?: \(mV\)|_mV)$")
_LOCAL_TEMP_RE = re.compile(r"^Local_temperature(?: \(C\)|_C)$")


def load_run(path: str | Path) -> PermeationRun:
    """Load a canonical run directory into a :class:`PermeationRun`.

    Args:
        path: Run directory containing ``shield_data.csv`` and
            ``run_metadata.json``.

    Returns:
        The loaded run. The time axis is seconds since the first sample;
        voltages are raw (V for gauges, mV for thermocouples).

    Raises:
        FileNotFoundError: If the directory is not a canonical run — with a
            pointer to the converter when it holds an old-generation run.
    """
    path = Path(path)
    csv_path = path / CANONICAL_CSV
    if not csv_path.is_file():
        if (path / LEGACY_PRESSURE_CSV).is_file():
            raise FileNotFoundError(
                f"{path} is an old-format run ({LEGACY_PRESSURE_CSV}); "
                "convert it once with shield_toolbox.io.convert_run() or "
                "scripts/convert_runs.py"
            )
        raise FileNotFoundError(
            f"{path} is not a SHIELD run directory (no {CANONICAL_CSV})"
        )
    metadata = _load_metadata(path)

    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip() for column in frame.columns]
    if _TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(
            f"{csv_path} has no {_TIMESTAMP_COLUMN!r} column; "
            f"columns: {list(frame.columns)}"
        )

    timestamps = pd.DatetimeIndex(pd.to_datetime(frame[_TIMESTAMP_COLUMN]))
    time_s = ((timestamps - timestamps[0]).total_seconds()).to_numpy(dtype=float)

    gauge_voltages: dict[str, np.ndarray] = {}
    thermocouple_mv: dict[str, np.ndarray] = {}
    local_temperature_c: np.ndarray | None = None
    for column in frame.columns:
        if match := _GAUGE_VOLTAGE_RE.match(column):
            gauge_voltages[match["name"]] = frame[column].to_numpy(dtype=float)
        elif match := _TC_VOLTAGE_RE.match(column):
            thermocouple_mv[match["name"]] = frame[column].to_numpy(dtype=float)
        elif _LOCAL_TEMP_RE.match(column):
            local_temperature_c = frame[column].to_numpy(dtype=float)

    return PermeationRun(
        path=path,
        run_id=path.name,
        metadata=metadata,
        timestamps=timestamps.to_numpy(),
        time_s=time_s,
        gauge_voltages=gauge_voltages,
        gauge_locations=_gauge_locations(metadata),
        thermocouple_mv=thermocouple_mv,
        local_temperature_c=local_temperature_c,
        valve_times_s=_valve_times_s(metadata, timestamps[0]),
    )


def _load_metadata(path: Path) -> dict:
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No {METADATA_FILENAME} in {path}")
    with open(metadata_path) as f:
        return json.load(f)


def _gauge_locations(metadata: dict) -> dict[str, str]:
    return {
        gauge["name"]: gauge["gauge_location"]
        for gauge in metadata.get("gauges", [])
        if "name" in gauge and "gauge_location" in gauge
    }


def _valve_times_s(metadata: dict, t0: pd.Timestamp) -> dict[str, float]:
    """Valve events from ``run_info`` as seconds since the first sample.

    The recorder stores absolute timestamps under keys like
    ``"v3_open_time"``; ``start_time`` is the recording start, not a valve
    event, and is skipped.
    """
    valve_times: dict[str, float] = {}
    for key, value in metadata.get("run_info", {}).items():
        if key == "start_time" or not key.endswith("_time"):
            continue
        if not isinstance(value, str) or not value:
            continue
        try:
            event_time = pd.Timestamp(value)
        except ValueError:
            continue
        valve_times[key] = float((event_time - t0).total_seconds())
    return valve_times
