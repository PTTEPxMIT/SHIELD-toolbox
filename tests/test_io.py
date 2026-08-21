"""Tests for shield_toolbox.io — loading canonical runs.

Fixtures in tests/fixtures/ are small synthetic run directories modeled on
real DAS output (same headers and metadata shapes, made-up values);
``combined_run`` is canonical, the other two are old generations used by the
converter tests.
"""

import json
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest

from shield_toolbox.io import (
    PermeationRun,
    convert_run,
    fetch_run,
    find_leak_test_id,
    load_run,
)


@pytest.fixture
def combined_run(fixtures_dir):
    return load_run(fixtures_dir / "combined_run")


@pytest.fixture
def pressure_only_run(fixtures_dir, tmp_path):
    """The oldest-generation fixture, converted then loaded."""
    return load_run(
        convert_run(fixtures_dir / "pressure_only_run", tmp_path / "pressure_only_run")
    )


def test_old_format_dir_points_to_converter(fixtures_dir):
    with pytest.raises(FileNotFoundError, match="convert"):
        load_run(fixtures_dir / "pressure_only_run")


def test_non_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a SHIELD run directory"):
        load_run(tmp_path)


def test_missing_metadata_raises(tmp_path):
    (tmp_path / "shield_data.csv").write_text(
        "RealTimestamp,WGM701_Voltage (V)\n2025-10-06 10:41:11.342,3.2\n"
    )
    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        load_run(tmp_path)


def test_load_combined_run(combined_run):
    run = combined_run
    assert isinstance(run, PermeationRun)
    assert run.run_id == "combined_run"
    assert set(run.gauge_voltages) == {
        "WGM701",
        "Baratron626D_1KT",
        "Baratron626D_1T",
    }
    np.testing.assert_allclose(
        run.thermocouple_mv["type K thermocouple"],
        [20.10, 20.20, 20.30, 20.40, 20.50],
    )
    np.testing.assert_allclose(run.local_temperature_c, [24.9, 25.0, 25.1, 25.2, 25.3])


def test_time_axis_is_relative_to_first_sample(combined_run):
    run = combined_run
    assert run.time_s[0] == 0.0
    np.testing.assert_allclose(run.time_s, [0.0, 0.5, 1.0, 1.5, 2.0])
    assert len(run.timestamps) == len(run.time_s)


def test_gauge_voltages_and_locations(pressure_only_run):
    run = pressure_only_run
    np.testing.assert_allclose(run.voltage("WGM701"), [3.20, 3.21, 3.22, 3.23, 3.24])
    assert run.gauge_locations["Baratron626D_1KT"] == "upstream"
    assert sorted(run.gauges_at("upstream")) == ["Baratron626D_1KT", "CVM211"]
    assert sorted(run.gauges_at("downstream")) == ["Baratron626D_1T", "WGM701"]
    with pytest.raises(KeyError, match="No gauge named"):
        run.voltage("does_not_exist")


def test_valve_times_relative_to_first_sample(pressure_only_run):
    # First sample at 10:41:11.342; v5 closes at 10:41:12.000.
    assert pressure_only_run.valve_times_s["v5_close_time"] == pytest.approx(0.658)
    assert pressure_only_run.valve_times_s["v3_open_time"] == pytest.approx(1.658)
    assert "start_time" not in pressure_only_run.valve_times_s


def test_metadata_accessors(pressure_only_run):
    run = pressure_only_run
    assert run.furnace_setpoint == 500.0
    assert run.start_time is not None
    assert run.start_time.isoformat() == "2025-10-06T10:41:11"


def test_top_level_exports():
    import shield_toolbox

    assert shield_toolbox.load_run is load_run
    assert shield_toolbox.convert_run is convert_run
    assert shield_toolbox.fetch_run is fetch_run
    assert shield_toolbox.PermeationRun is PermeationRun


@pytest.fixture
def stored_parquet_run(fixtures_dir, tmp_path):
    """The combined_run fixture rewritten in the SHIELD-Data stored layout
    (measurements.parquet + run_metadata.json)."""
    run_dir = tmp_path / "25.10.06_run_1_10h41"
    run_dir.mkdir()
    frame = pd.read_csv(fixtures_dir / "combined_run" / "shield_data.csv")
    frame["RealTimestamp"] = pd.to_datetime(frame["RealTimestamp"])
    frame.to_parquet(run_dir / "measurements.parquet", index=False)
    shutil.copy2(
        fixtures_dir / "combined_run" / "run_metadata.json",
        run_dir / "run_metadata.json",
    )
    return run_dir


def test_load_stored_parquet_run_matches_csv(combined_run, stored_parquet_run):
    run = load_run(stored_parquet_run)
    assert run.run_id == "25.10.06_run_1_10h41"
    assert set(run.gauge_voltages) == set(combined_run.gauge_voltages)
    for name, trace in combined_run.gauge_voltages.items():
        np.testing.assert_allclose(run.voltage(name), trace)
    np.testing.assert_allclose(run.time_s, combined_run.time_s)
    np.testing.assert_allclose(
        run.local_temperature_c, combined_run.local_temperature_c
    )


def test_csv_takes_precedence_over_parquet(fixtures_dir, stored_parquet_run):
    # A dir holding both layouts (e.g. a converted checkout) loads the rig CSV.
    shutil.copy2(
        fixtures_dir / "combined_run" / "shield_data.csv",
        stored_parquet_run / "shield_data.csv",
    )
    assert load_run(stored_parquet_run).gauge_voltages


@pytest.fixture
def fake_shield_data(fixtures_dir, monkeypatch):
    """A stand-in shield_data module serving the combined_run fixture."""
    frame = pd.read_csv(fixtures_dir / "combined_run" / "shield_data.csv")
    frame["RealTimestamp"] = pd.to_datetime(frame["RealTimestamp"])
    frame["run_id"] = "combined_run"  # sd.load appends this column
    with open(fixtures_dir / "combined_run" / "run_metadata.json") as f:
        metadata = json.load(f)

    module = types.ModuleType("shield_data")
    module.load = lambda run_id: frame
    module.load_metadata = lambda run_id: metadata
    monkeypatch.setitem(sys.modules, "shield_data", module)
    return module


def test_fetch_run(fake_shield_data, combined_run):
    run = fetch_run("combined_run")
    assert isinstance(run, PermeationRun)
    assert run.run_id == "combined_run"
    assert set(run.gauge_voltages) == set(combined_run.gauge_voltages)
    for name, trace in combined_run.gauge_voltages.items():
        np.testing.assert_allclose(run.voltage(name), trace)
    assert run.valve_times_s == combined_run.valve_times_s
    # The appended run_id column is ignored, not parsed as a channel.
    assert "run_id" not in run.gauge_voltages


def test_fetch_run_without_shield_data_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "shield_data", None)
    with pytest.raises(ImportError, match="SHIELD-Data"):
        fetch_run("combined_run")


# --- leak-test pairing -------------------------------------------------------


def _catalogue_row(run_id, run_type, sample_id, start_time, substrate="316L"):
    return {
        "run_id": run_id,
        "run_type": run_type,
        "sample_id": sample_id,
        "substrate": substrate,
        "coating": "Al2O3",
        "start_time": start_time,
    }


LEAK_CATALOGUE = pd.DataFrame(
    [
        _catalogue_row(
            "26.08.01_run_1_09h00", "leak_test", "S-01", "2026-08-01T09:00:00"
        ),
        _catalogue_row(
            "26.08.10_run_1_09h00", "leak_test", "S-01", "2026-08-10T09:00:00"
        ),
        _catalogue_row(
            "26.08.30_run_1_09h00", "leak_test", "S-01", "2026-08-30T09:00:00"
        ),
        _catalogue_row(
            "26.08.05_run_1_09h00", "leak_test", "S-02", "2026-08-05T09:00:00"
        ),
        _catalogue_row(
            "26.08.11_run_1_10h00", "permeation_exp", "S-01", "2026-08-11T10:00:00"
        ),
    ]
)


def _run_stub(run_info: dict) -> PermeationRun:
    from pathlib import Path

    return PermeationRun(
        path=Path("stub"),
        run_id="stub",
        metadata={"run_info": run_info},
        timestamps=np.array(["2026-08-15T10:00:00"], dtype="datetime64[ns]"),
        time_s=np.zeros(1),
        gauge_voltages={},
        gauge_locations={},
    )


def test_find_leak_test_picks_latest_prior_same_sample():
    run = _run_stub({"sample_id": "S-01", "start_time": "2026-08-15T10:00:00"})
    # Latest leak test before the run wins; the later one and other samples
    # (and permeation rows) are ignored.
    assert find_leak_test_id(LEAK_CATALOGUE, run) == "26.08.10_run_1_09h00"


def test_find_leak_test_none_when_no_prior_match():
    run = _run_stub({"sample_id": "S-03", "start_time": "2026-08-15T10:00:00"})
    assert find_leak_test_id(LEAK_CATALOGUE, run) is None
    early = _run_stub({"sample_id": "S-01", "start_time": "2026-07-01T10:00:00"})
    assert find_leak_test_id(LEAK_CATALOGUE, early) is None


def test_find_leak_test_falls_back_to_substrate_coating_with_warning():
    run = _run_stub(
        {
            "sample_substrate": "316L",
            "sample_coating": "Al2O3",
            "start_time": "2026-08-15T10:00:00",
        }
    )
    with pytest.warns(UserWarning, match="no sample_id"):
        assert find_leak_test_id(LEAK_CATALOGUE, run) == "26.08.10_run_1_09h00"


def test_find_leak_test_none_without_run_type_column():
    run = _run_stub({"sample_id": "S-01", "start_time": "2026-08-15T10:00:00"})
    assert find_leak_test_id(pd.DataFrame({"run_id": []}), run) is None
