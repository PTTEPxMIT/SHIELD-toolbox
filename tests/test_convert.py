"""Tests for shield_toolbox.io.convert — upgrading old run directories."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from shield_toolbox.io import convert_run, load_run


def test_pressure_only_converts_to_copy(fixtures_dir, tmp_path):
    out = convert_run(fixtures_dir / "pressure_only_run", tmp_path / "converted")
    canonical = pd.read_csv(out / "shield_data.csv")
    original = pd.read_csv(fixtures_dir / "pressure_only_run/pressure_gauge_data.csv")
    pd.testing.assert_frame_equal(canonical, original)

    with open(out / "run_metadata.json") as f:
        metadata = json.load(f)
    assert metadata["version"] == "1.3"
    assert metadata["run_info"]["data_filename"] == "shield_data.csv"
    # Everything else survives the upgrade.
    assert metadata["run_info"]["furnace_setpoint"] == 500
    assert len(metadata["gauges"]) == 4


def test_split_run_joins_thermocouple_columns(fixtures_dir, tmp_path):
    # Real gen-2 files share one timestamp per tick -> exact column join.
    out = convert_run(fixtures_dir / "split_run", tmp_path / "converted")
    run = load_run(out)
    np.testing.assert_allclose(
        run.thermocouple_mv["type K thermocouple"], [12.10, 12.20, 12.30, 12.40]
    )
    assert len(run.time_s) == 4
    with open(out / "run_metadata.json") as f:
        metadata = json.load(f)
    assert metadata["version"] == "1.3"
    assert "temperature_data_filename" not in metadata
    assert "pressure_data_filename" not in metadata["run_info"]


def test_drifted_thermocouple_rows_join_nearest(fixtures_dir, tmp_path):
    # A truncated TC file (one row short, slightly late stamps) still joins:
    # each pressure row takes the nearest TC row.
    src = tmp_path / "drifted"
    shutil.copytree(fixtures_dir / "split_run", src)
    (src / "thermocouple_data.csv").write_text(
        "RealTimestamp,type K thermocouple_Voltage (mV)\n"
        "2025-11-10 09:00:00.010,12.10\n"
        "2025-11-10 09:00:00.510,12.20\n"
        "2025-11-10 09:00:01.010,12.30\n"
    )
    run = load_run(convert_run(src, tmp_path / "drifted_out"))
    np.testing.assert_allclose(
        run.thermocouple_mv["type K thermocouple"], [12.10, 12.20, 12.30, 12.30]
    )
    assert len(run.time_s) == 4


def test_canonical_run_is_copied_unchanged(fixtures_dir, tmp_path):
    out = convert_run(fixtures_dir / "combined_run", tmp_path / "converted")
    original = pd.read_csv(fixtures_dir / "combined_run/shield_data.csv")
    pd.testing.assert_frame_equal(pd.read_csv(out / "shield_data.csv"), original)


def test_in_place_conversion_keeps_originals(fixtures_dir, tmp_path):
    src = tmp_path / "pressure_only_run"
    shutil.copytree(fixtures_dir / "pressure_only_run", src)
    out = convert_run(src)  # no dest -> in place
    assert out == src
    assert (src / "shield_data.csv").is_file()
    assert (src / "pressure_gauge_data.csv").is_file()  # original kept
    run = load_run(src)
    assert len(run.time_s) == 5


def test_missing_metadata_raises(tmp_path):
    (tmp_path / "pressure_gauge_data.csv").write_text(
        "RealTimestamp,WGM701_Voltage (V)\n2025-10-06 10:41:11.342,3.2\n"
    )
    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        convert_run(tmp_path)


def test_empty_dir_raises(tmp_path):
    (tmp_path / "run_metadata.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="not a SHIELD run directory"):
        convert_run(tmp_path)
