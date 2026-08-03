"""One-time conversion of old-generation run directories to the canonical
format (what the current DAS recorder writes)::

    <run_dir>/
    ├── shield_data.csv      one row per tick: RealTimestamp, gauge voltages
    │                        (V), thermocouple voltages (mV), local temp (C)
    └── run_metadata.json    version "1.3", run_info.data_filename set

Two older layouts exist and convert losslessly:

- ``pressure_gauge_data.csv`` only (oldest): the canonical CSV is the same
  table — the conversion is a straight copy under the new name.
- ``pressure_gauge_data.csv`` + ``thermocouple_data.csv``: both files were
  written in the same acquisition loop with the same timestamp per tick, so
  the thermocouple columns are joined on timestamp (nearest-match as a
  fallback for any drift).

Conversion is non-destructive: the original CSVs are left in place;
``run_metadata.json`` is upgraded (version bumped, filename keys updated).
After converting, :func:`shield_toolbox.io.load_run` handles only the
canonical layout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

CANONICAL_CSV = "shield_data.csv"
METADATA_FILENAME = "run_metadata.json"
LEGACY_PRESSURE_CSV = "pressure_gauge_data.csv"
LEGACY_TC_CSV = "thermocouple_data.csv"
_TIMESTAMP_COLUMN = "RealTimestamp"


def convert_run(src: str | Path, dest: str | Path | None = None) -> Path:
    """Convert a run directory to the canonical format.

    Args:
        src: Run directory in any generation's layout.
        dest: Output directory; defaults to ``src`` (in place — adds
            ``shield_data.csv`` next to the originals and upgrades the
            metadata file).

    Returns:
        The directory containing the canonical run.

    Raises:
        FileNotFoundError: If ``src`` has no run CSV or no metadata file.
    """
    src = Path(src)
    dest = src if dest is None else Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    metadata_path = src / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No {METADATA_FILENAME} in {src}")
    with open(metadata_path) as f:
        metadata = json.load(f)

    if (src / CANONICAL_CSV).is_file():
        if dest != src:
            shutil.copy2(src / CANONICAL_CSV, dest / CANONICAL_CSV)
            shutil.copy2(metadata_path, dest / METADATA_FILENAME)
        return dest

    if not (src / LEGACY_PRESSURE_CSV).is_file():
        raise FileNotFoundError(
            f"{src} is not a SHIELD run directory: found neither "
            f"{CANONICAL_CSV!r} nor {LEGACY_PRESSURE_CSV!r}"
        )

    frame = _read_csv(src / LEGACY_PRESSURE_CSV)
    if (src / LEGACY_TC_CSV).is_file():
        frame = _join_thermocouple(frame, _read_csv(src / LEGACY_TC_CSV))

    frame.to_csv(dest / CANONICAL_CSV, index=False)
    with open(dest / METADATA_FILENAME, "w") as f:
        json.dump(_upgrade_metadata(metadata), f, indent=2)
    return dest


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    if _TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"{path} has no {_TIMESTAMP_COLUMN!r} column")
    return frame


def _join_thermocouple(pressure: pd.DataFrame, tc: pd.DataFrame) -> pd.DataFrame:
    """Join thermocouple columns onto the pressure rows.

    Both files were written once per acquisition tick with the same
    timestamp, so an exact join normally applies; nearest-match covers any
    row drift (e.g. a truncated final backup row).
    """
    same_stamps = (
        len(pressure) == len(tc)
        and (
            pressure[_TIMESTAMP_COLUMN].to_numpy() == tc[_TIMESTAMP_COLUMN].to_numpy()
        ).all()
    )
    if same_stamps:
        tc_columns = tc.drop(columns=[_TIMESTAMP_COLUMN])
        return pd.concat([pressure, tc_columns], axis=1)

    left = pressure.assign(_ts=pd.to_datetime(pressure[_TIMESTAMP_COLUMN]))
    right = tc.assign(_ts=pd.to_datetime(tc[_TIMESTAMP_COLUMN])).drop(
        columns=[_TIMESTAMP_COLUMN]
    )
    merged = pd.merge_asof(
        left.sort_values("_ts"),
        right.sort_values("_ts"),
        on="_ts",
        direction="nearest",
    )
    return merged.drop(columns=["_ts"])


def _upgrade_metadata(metadata: dict) -> dict:
    upgraded = dict(metadata)
    upgraded["version"] = "1.3"
    run_info = dict(upgraded.get("run_info", {}))
    run_info["data_filename"] = CANONICAL_CSV
    run_info.pop("pressure_data_filename", None)
    upgraded["run_info"] = run_info
    upgraded.pop("temperature_data_filename", None)
    return upgraded
