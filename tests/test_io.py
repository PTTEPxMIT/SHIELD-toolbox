"""Tests for shield_toolbox.io — loading canonical runs.

Fixtures in tests/fixtures/ are small synthetic run directories modeled on
real DAS output (same headers and metadata shapes, made-up values);
``combined_run`` is canonical, the other two are old generations used by the
converter tests.
"""

import numpy as np
import pytest

from shield_toolbox.io import PermeationRun, convert_run, load_run


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
    assert shield_toolbox.PermeationRun is PermeationRun
