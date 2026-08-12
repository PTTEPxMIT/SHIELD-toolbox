"""Tests for shield_toolbox.processing — process_run and the stored artifact."""

import json

import numpy as np
import pandas as pd
import pytest

from shield_toolbox import get_rig_config, load_run
from shield_toolbox.processing import (
    RESULT_FILENAME,
    TIMESERIES_FILENAME,
    SampleInfo,
    process_run,
)

SAMPLE = SampleInfo(
    substrate="316L", coating="Al2O3", thickness_m=0.00088, sample_id="S-01"
)


@pytest.fixture
def processed(fixtures_dir):
    run = load_run(fixtures_dir / "pressure_only_run")
    return process_run(run, SAMPLE)


def test_rig_resolved_from_run_date(processed):
    assert processed.rig is get_rig_config("v1")


def test_timeseries_columns_and_values(processed):
    ts = processed.timeseries
    expected = {
        "timestamp",
        "time_s",
        "upstream_torr",
        "upstream_err_torr",
        "downstream_torr",
        "downstream_err_torr",
        "temperature_K",
        "in_run",
        "fit_used",
    }
    assert expected <= set(ts.columns)
    # Raw voltages are carried through, one column per gauge.
    assert "Baratron626D_1KT_voltage_V" in ts.columns
    assert "WGM701_voltage_V" in ts.columns
    # Calibration: upstream 1000 Torr FS -> V*100; downstream 1 Torr -> V/10.
    np.testing.assert_allclose(
        ts["upstream_torr"], ts["Baratron626D_1KT_voltage_V"] * 100.0
    )
    np.testing.assert_allclose(
        ts["downstream_torr"], ts["Baratron626D_1T_voltage_V"] / 10.0
    )
    # Fixture never saturates -> whole trace is in-run.
    assert ts["in_run"].all()
    # No thermocouple in the oldest format.
    assert ts["temperature_K"].isna().all()


def test_temperature_fallback_uses_setpoint_offset(processed):
    # v1 offset is -18 K; fixture setpoint is 500.
    assert processed.sample_temperature_K == pytest.approx(482.0)
    assert processed.temperature_source == "furnace_setpoint_offset"


def test_result_dict_contents(processed):
    result = processed.result_dict()
    assert result["run_id"] == "pressure_only_run"
    assert result["sample"] == {
        "substrate": "316L",
        "coating": "Al2O3",
        "thickness_m": 0.00088,
        "sample_id": "S-01",
    }
    assert result["provenance"]["rig_version"] == "v1"
    assert result["results"]["permeability"]["units"] == "H/(m·s·Pa^0.5)"
    assert result["results"]["permeability"]["std_dev"] > 0
    assert result["window"]["n_in_run"] == 5
    assert result["run_info"]["valve_times_s"]["v5_close_time"] == pytest.approx(0.658)


def test_write_creates_material_tree(processed, tmp_path):
    out_dir = processed.write(tmp_path)
    assert out_dir == tmp_path / "316L" / "Al2O3" / "pressure_only_run"
    assert (out_dir / TIMESERIES_FILENAME).is_file()
    assert (out_dir / RESULT_FILENAME).is_file()

    # Parquet round-trips with dtypes intact.
    ts = pd.read_parquet(out_dir / TIMESERIES_FILENAME)
    assert ts["in_run"].dtype == bool
    assert len(ts) == 5

    with open(out_dir / RESULT_FILENAME) as f:
        result = json.load(f)
    assert result["sample"]["substrate"] == "316L"


def test_saturated_window_is_excluded(fixtures_dir):
    # split_run fixture: first two upstream samples at the 10.12 V cap.
    run = load_run(fixtures_dir / "split_run")
    processed = process_run(run, SampleInfo(substrate="316L", thickness_m=0.00088))
    ts = processed.timeseries
    np.testing.assert_array_equal(ts["in_run"], [False, False, True, True])
    assert processed.temperature_source == "thermocouple"
    # Thermocouple mV interpolated then converted; finite everywhere.
    assert np.isfinite(ts["temperature_K"]).all()
