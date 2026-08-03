"""Tests for shield_toolbox.io — load_run across all three raw formats.

Fixtures in tests/fixtures/ are small synthetic run directories modeled on
real DAS output (same headers and metadata shapes, made-up values).
"""

import numpy as np
import pytest

from shield_toolbox.io import PermeationRun, RunFormat, detect_run_format, load_run


@pytest.fixture
def pressure_only_run(fixtures_dir):
    return load_run(fixtures_dir / "pressure_only_run")


@pytest.fixture
def split_run(fixtures_dir):
    return load_run(fixtures_dir / "split_run")


@pytest.fixture
def combined_run(fixtures_dir):
    return load_run(fixtures_dir / "combined_run")


def test_detect_format(fixtures_dir):
    assert detect_run_format(fixtures_dir / "pressure_only_run") is (
        RunFormat.PRESSURE_ONLY
    )
    assert detect_run_format(fixtures_dir / "split_run") is RunFormat.SPLIT
    assert detect_run_format(fixtures_dir / "combined_run") is RunFormat.COMBINED


def test_detect_format_rejects_non_run_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a SHIELD run directory"):
        detect_run_format(tmp_path)


def test_missing_metadata_raises(tmp_path):
    (tmp_path / "pressure_gauge_data.csv").write_text(
        "RealTimestamp,WGM701_Voltage (V)\n2025-10-06 10:41:11.342,3.2\n"
    )
    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        load_run(tmp_path)


def test_pressure_only_run(pressure_only_run):
    run = pressure_only_run
    assert isinstance(run, PermeationRun)
    assert run.run_id == "pressure_only_run"
    assert run.format is RunFormat.PRESSURE_ONLY
    assert set(run.gauge_voltages) == {
        "WGM701",
        "CVM211",
        "Baratron626D_1KT",
        "Baratron626D_1T",
    }
    # No thermocouple file in the oldest generation.
    assert run.thermocouple_mv == {}
    assert run.thermocouple_time_s is None
    assert run.local_temperature_c is None


def test_time_axis_is_relative_to_first_sample(pressure_only_run):
    run = pressure_only_run
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
    assert run.metadata["version"] == "1.0"


def test_split_run_thermocouple_own_time_axis(split_run):
    run = split_run
    assert run.format is RunFormat.SPLIT
    np.testing.assert_allclose(
        run.thermocouple_mv["type K thermocouple"], [12.10, 12.20, 12.30, 12.40]
    )
    # TC samples are offset 0.25 s from the pressure samples and keep their
    # own axis, zeroed at the first *pressure* sample.
    np.testing.assert_allclose(run.thermocouple_time_s, [0.25, 0.75, 1.25, 1.75])
    np.testing.assert_allclose(run.time_s, [0.0, 0.5, 1.0, 1.5])
    assert run.local_temperature_c is None


def test_combined_run_has_everything_on_one_axis(combined_run):
    run = combined_run
    assert run.format is RunFormat.COMBINED
    assert set(run.gauge_voltages) == {
        "WGM701",
        "Baratron626D_1KT",
        "Baratron626D_1T",
    }
    np.testing.assert_allclose(
        run.thermocouple_mv["type K thermocouple"],
        [20.10, 20.20, 20.30, 20.40, 20.50],
    )
    np.testing.assert_allclose(run.thermocouple_time_s, run.time_s)
    np.testing.assert_allclose(run.local_temperature_c, [24.9, 25.0, 25.1, 25.2, 25.3])


def test_top_level_exports():
    import shield_toolbox

    assert shield_toolbox.load_run is load_run
    assert shield_toolbox.PermeationRun is PermeationRun
