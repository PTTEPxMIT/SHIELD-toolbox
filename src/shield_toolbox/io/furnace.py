"""Load Eurotherm furnace-controller log exports.

The SHIELD furnace controller (Eurotherm) exports CSV logs (``LOG*.csv`` /
``TCCOMP*.csv``) with an integrity-hash first line, then a header row of
``Date``, ``Time``, ``Main_Controller_PV``, ``Main_Controller_Working_SP``
and per-column ``QF`` quality flags. These logs are recorded by the furnace
itself, independently of the DAS — they are what the rig's furnace actually
did (measured temperature vs its own working setpoint), used to check
heating/cooling rates and to calibrate the sample-vs-furnace temperature
offset (:func:`shield_toolbox.analysis.furnace_temperature_offset`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATE_COLUMN = "Date"
_TIME_COLUMN = "Time"
_PV_COLUMN = "Main_Controller_PV"
_SETPOINT_COLUMN = "Main_Controller_Working_SP"


def load_furnace_log(path: str | Path) -> pd.DataFrame:
    """Load one Eurotherm furnace log into a tidy DataFrame.

    Args:
        path: A ``LOG*.csv`` / ``TCCOMP*.csv`` controller export. A leading
            integrity-hash line (no header fields) is detected and skipped.

    Returns:
        Columns ``timestamp`` (datetime), ``time_s`` (seconds since the
        first row), ``furnace_temperature_C`` (the controller's measured
        process value), ``setpoint_C`` (its working setpoint, which ramps
        during programmed heating). One row per log tick.

    Raises:
        ValueError: If the file lacks the expected controller columns.
    """
    path = Path(path)
    with open(path) as f:
        first_line = f.readline()
    skiprows = 0 if _DATE_COLUMN in first_line.split(",") else 1

    frame = pd.read_csv(path, skiprows=skiprows)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = {_DATE_COLUMN, _TIME_COLUMN, _PV_COLUMN, _SETPOINT_COLUMN} - set(
        frame.columns
    )
    if missing:
        raise ValueError(
            f"{path} does not look like a Eurotherm log export; "
            f"missing columns: {sorted(missing)}"
        )

    # Controller dates are MM/DD/YYYY; combining with the date makes the
    # time axis robust to logs that run past midnight.
    timestamps = pd.to_datetime(
        frame[_DATE_COLUMN].str.strip() + " " + frame[_TIME_COLUMN].str.strip(),
        format="%m/%d/%Y %H:%M:%S",
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "time_s": (timestamps - timestamps.iloc[0]).dt.total_seconds(),
            "furnace_temperature_C": frame[_PV_COLUMN].astype(float),
            "setpoint_C": frame[_SETPOINT_COLUMN].astype(float),
        }
    )
