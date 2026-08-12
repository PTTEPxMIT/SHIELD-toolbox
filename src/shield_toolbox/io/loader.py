"""Load recorded SHIELD runs from disk, across all raw-data generations.

The DAS has written three on-disk formats over time (newest first):

1. ``shield_data.csv`` — combined pressure + thermocouple + local temperature.
2. ``pressure_gauge_data.csv`` + ``thermocouple_data.csv``.
3. ``pressure_gauge_data.csv`` only.

:func:`load_run` hides the differences behind one entry point and returns a
:class:`~shield_toolbox.io.run.PermeationRun`. Column headers carry units on
disk (``WGM701_Voltage (V)``, ``type K thermocouple_Voltage (mV)``,
``Local_temperature (C)``); some tools re-export them with sanitized names
(``WGM701_Voltage_V``), so both spellings are accepted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from shield_toolbox.io.run import PermeationRun, RunFormat

METADATA_FILENAME = "run_metadata.json"
COMBINED_CSV = "shield_data.csv"
PRESSURE_CSV = "pressure_gauge_data.csv"
THERMOCOUPLE_CSV = "thermocouple_data.csv"

_TIMESTAMP_COLUMN = "RealTimestamp"
_GAUGE_VOLTAGE_RE = re.compile(r"^(?P<name>.+?)_Voltage(?: \(V\)|_V)$")
_TC_VOLTAGE_RE = re.compile(r"^(?P<name>.+?)_Voltage(?: \(mV\)|_mV)$")
_LOCAL_TEMP_RE = re.compile(r"^Local_temperature(?: \(C\)|_C)$")


def detect_run_format(path: str | Path) -> RunFormat:
    """Determine which raw-data generation a run directory was recorded in."""
    path = Path(path)
    if (path / COMBINED_CSV).is_file():
        return RunFormat.COMBINED
    if (path / PRESSURE_CSV).is_file():
        if (path / THERMOCOUPLE_CSV).is_file():
            return RunFormat.SPLIT
        return RunFormat.PRESSURE_ONLY
    raise FileNotFoundError(
        f"{path} is not a SHIELD run directory: found neither "
        f"{COMBINED_CSV!r} nor {PRESSURE_CSV!r}"
    )


def load_run(path: str | Path) -> PermeationRun:
    """Load a recorded run directory into a :class:`PermeationRun`.

    Args:
        path: Run directory containing the data CSV(s) and
            ``run_metadata.json``.

    Returns:
        The loaded run. All time axes are seconds since the first pressure
        sample; voltages are raw (V for gauges, mV for thermocouples).
    """
    path = Path(path)
    run_format = detect_run_format(path)
    metadata = _load_metadata(path)

    if run_format is RunFormat.COMBINED:
        frame = _read_data_csv(path / COMBINED_CSV)
        tc_frame = frame
    else:
        frame = _read_data_csv(path / PRESSURE_CSV)
        tc_frame = (
            _read_data_csv(path / THERMOCOUPLE_CSV)
            if run_format is RunFormat.SPLIT
            else None
        )

    timestamps = _parse_timestamps(frame)
    t0 = timestamps[0]
    time_s = _seconds_since(timestamps, t0)

    gauge_voltages: dict[str, np.ndarray] = {}
    local_temperature_c: np.ndarray | None = None
    for column in frame.columns:
        if match := _GAUGE_VOLTAGE_RE.match(column):
            gauge_voltages[match["name"]] = frame[column].to_numpy(dtype=float)
        elif _LOCAL_TEMP_RE.match(column):
            local_temperature_c = frame[column].to_numpy(dtype=float)

    thermocouple_mv: dict[str, np.ndarray] = {}
    thermocouple_time_s: np.ndarray | None = None
    if tc_frame is not None:
        for column in tc_frame.columns:
            if match := _TC_VOLTAGE_RE.match(column):
                thermocouple_mv[match["name"]] = tc_frame[column].to_numpy(dtype=float)
        if thermocouple_mv:
            thermocouple_time_s = _seconds_since(_parse_timestamps(tc_frame), t0)

    return PermeationRun(
        path=path,
        run_id=path.name,
        format=run_format,
        metadata=metadata,
        timestamps=timestamps.to_numpy(),
        time_s=time_s,
        gauge_voltages=gauge_voltages,
        gauge_locations=_gauge_locations(metadata),
        thermocouple_mv=thermocouple_mv,
        thermocouple_time_s=thermocouple_time_s,
        local_temperature_c=local_temperature_c,
        valve_times_s=_valve_times_s(metadata, t0),
    )


def _load_metadata(path: Path) -> dict:
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No {METADATA_FILENAME} in {path}")
    with open(metadata_path) as f:
        return json.load(f)


def _read_data_csv(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip() for column in frame.columns]
    if _TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(
            f"{csv_path} has no {_TIMESTAMP_COLUMN!r} column; "
            f"columns: {list(frame.columns)}"
        )
    return frame


def _parse_timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(frame[_TIMESTAMP_COLUMN]))


def _seconds_since(timestamps: pd.DatetimeIndex, t0: pd.Timestamp) -> np.ndarray:
    return ((timestamps - t0).total_seconds()).to_numpy(dtype=float)


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
