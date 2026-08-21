"""Tests for shield_toolbox.processing — process_run and the stored artifact."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shield_toolbox import convert_run, get_rig_config, load_run
from shield_toolbox.io import PermeationRun
from shield_toolbox.processing import (
    RESULT_FILENAME,
    TIMESERIES_FILENAME,
    SampleInfo,
    process_leak_test,
    process_run,
)

SAMPLE = SampleInfo(
    substrate="316L", coating="Al2O3", thickness_m=0.00088, sample_id="S-01"
)


@pytest.fixture
def processed(fixtures_dir, tmp_path):
    run_dir = convert_run(
        fixtures_dir / "pressure_only_run", tmp_path / "pressure_only_run"
    )
    return process_run(load_run(run_dir), SAMPLE)


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
    # Fixture setpoint is 500 °C; v1 offset is -18 K: 500 + 273.15 - 18.
    assert processed.sample_temperature_K == pytest.approx(755.15)
    assert processed.temperature_source == "furnace_setpoint_offset"


def test_result_dict_contents(processed):
    result = processed.result_dict()
    assert result["run_id"] == "pressure_only_run"
    assert result["sample"] == {
        "substrate": "316L",
        "coating": "Al2O3",
        "thickness_m": 0.00088,
        "sample_id": "S-01",
        "coating_layers": [],
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


def test_saturated_window_is_excluded(fixtures_dir, tmp_path):
    # split_run fixture: first two upstream samples at the 10.12 V cap.
    run_dir = convert_run(fixtures_dir / "split_run", tmp_path / "split_run")
    run = load_run(run_dir)
    processed = process_run(run, SampleInfo(substrate="316L", thickness_m=0.00088))
    ts = processed.timeseries
    np.testing.assert_array_equal(ts["in_run"], [False, False, True, True])
    assert processed.temperature_source == "thermocouple"
    # Thermocouple mV interpolated then converted; finite everywhere.
    assert np.isfinite(ts["temperature_K"]).all()


# --- sample description from metadata ---------------------------------------


V14_SAMPLE_FIELDS = {
    "sample_substrate": "carbon steel",
    "sample_coating": "800nm tungsten",
    "sample_coating_layers": [{"material": "tungsten", "thickness_nm": 800}],
    "sample_thickness": 0.00065,
}


def test_sample_info_from_v14_metadata():
    sample = SampleInfo.from_metadata({"run_info": dict(V14_SAMPLE_FIELDS)})
    assert sample == SampleInfo(
        substrate="carbon steel",
        coating="800nm tungsten",
        thickness_m=0.00065,
        coating_layers=({"material": "tungsten", "thickness_nm": 800},),
    )


def test_sample_info_from_legacy_material_fields():
    v13 = SampleInfo.from_metadata(
        {"run_info": {"sample_material": "316", "sample_thickness": 0.008}}
    )
    assert v13.substrate == "316"
    assert v13.coating == "uncoated"
    assert v13.thickness_m == 0.008
    assert v13.coating_layers == ()

    v10 = SampleInfo.from_metadata({"run_info": {"material": "steel"}})
    assert v10.substrate == "steel"


def test_sample_info_from_metadata_without_sample_returns_none():
    assert SampleInfo.from_metadata({"run_info": {"date": "2025-10-06"}}) is None


def test_process_run_defaults_to_metadata_sample(fixtures_dir, tmp_path):
    run_dir = convert_run(
        fixtures_dir / "pressure_only_run", tmp_path / "pressure_only_run"
    )
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["run_info"].update(V14_SAMPLE_FIELDS)
    metadata_path.write_text(json.dumps(metadata))

    processed = process_run(load_run(run_dir))

    assert processed.sample.substrate == "carbon steel"
    assert processed.sample.coating == "800nm tungsten"
    assert processed.sample.thickness_m == 0.00065
    result = processed.result_dict()
    assert result["sample"]["coating_layers"] == [
        {"material": "tungsten", "thickness_nm": 800}
    ]


def test_process_run_without_sample_or_metadata_raises(fixtures_dir, tmp_path):
    run_dir = convert_run(
        fixtures_dir / "pressure_only_run", tmp_path / "pressure_only_run"
    )
    with pytest.raises(ValueError, match="no sample description"):
        process_run(load_run(run_dir))


def test_sample_info_reads_v15_sample_id():
    sample = SampleInfo.from_metadata(
        {"run_info": dict(V14_SAMPLE_FIELDS, sample_id="S-07")}
    )
    assert sample.sample_id == "S-07"


# --- leak tests --------------------------------------------------------------

LEAK_RATE_TRUE = 2e-6  # Torr/s


def _leak_test_run(run_type: str = "leak_test") -> PermeationRun:
    """A synthetic leak-test run: 0.1 Torr + a linear background rise."""
    time_s = np.arange(0.0, 601.0, 1.0)
    downstream_torr = 0.1 + LEAK_RATE_TRUE * time_s
    metadata = {
        "version": "1.5",
        "run_info": {
            "date": "2026-08-20",
            "start_time": "2026-08-20T10:00:00",
            "run_type": run_type,
            "furnace_setpoint": 25,
            "sample_substrate": "316L",
            "sample_coating": "Al2O3",
            "sample_thickness": 0.00088,
            "sample_coating_layers": [],
            "sample_id": "S-01",
            "downstream_setpoint_torr": 0.1,
        },
        "gauges": [
            {
                "name": "Baratron626D_1T",
                "type": "Baratron626D_Gauge",
                "gauge_location": "downstream",
                "full_scale_torr": 1.0,
            }
        ],
        "thermocouples": [],
    }
    return PermeationRun(
        path=Path("leak_run"),
        run_id="leak_run",
        metadata=metadata,
        timestamps=np.datetime64("2026-08-20T10:00:00")
        + time_s.astype("timedelta64[s]"),
        time_s=time_s,
        # 1 Torr full scale: V = torr * 10.
        gauge_voltages={"Baratron626D_1T": downstream_torr * 10.0},
        gauge_locations={"Baratron626D_1T": "downstream"},
        valve_times_s={"downstream_isolated_time": 60.0},
    )


@pytest.fixture
def leak_result():
    return process_leak_test(_leak_test_run(), rig=get_rig_config("v1"))


def test_process_leak_test_fits_background_rate(leak_result):
    assert leak_result.rate_torr_per_s == pytest.approx(LEAK_RATE_TRUE)
    assert leak_result.measurement_start_s == 60.0
    assert leak_result.measurement_start_source == "downstream_isolated_time"
    assert leak_result.downstream_setpoint_torr == 0.1
    assert leak_result.sample.sample_id == "S-01"
    # Samples before the isolation event are excluded from the fit.
    assert not leak_result.timeseries["fit_used"].iloc[0]
    assert leak_result.timeseries["fit_used"].iloc[-1]


def test_process_leak_test_result_dict_and_write(leak_result, tmp_path):
    result = leak_result.result_dict()
    assert result["run_type"] == "leak_test"
    assert result["sample"]["sample_id"] == "S-01"
    assert result["run_info"]["downstream_setpoint_torr"] == 0.1
    assert result["results"]["leak_rate_torr_per_s"] == pytest.approx(LEAK_RATE_TRUE)
    assert result["results"]["leak_molar_rate"]["units"] == "mol/s"
    assert result["results"]["r_squared"] == pytest.approx(1.0)

    out_dir = leak_result.write(tmp_path)
    assert out_dir == tmp_path / "316L" / "Al2O3" / "leak_run"
    assert (out_dir / RESULT_FILENAME).is_file()
    assert (out_dir / TIMESERIES_FILENAME).is_file()


def test_process_leak_test_rejects_permeation_run():
    with pytest.raises(ValueError, match="not a leak test"):
        process_leak_test(
            _leak_test_run(run_type="permeation_exp"), rig=get_rig_config("v1")
        )


def test_process_run_applies_leak_rate_offset(fixtures_dir, tmp_path):
    run_dir = convert_run(
        fixtures_dir / "pressure_only_run", tmp_path / "pressure_only_run"
    )
    run = load_run(run_dir)
    leak_rate = 1e-9  # small enough not to move samples across the fit band
    base = process_run(run, SAMPLE)
    corrected = process_run(run, SAMPLE, leak=leak_rate)

    # Same fitted samples, slope reduced by exactly the leak rate.
    np.testing.assert_array_equal(
        corrected.downstream_fit.used, base.downstream_fit.used
    )
    assert corrected.downstream_fit.slope_torr_per_s == pytest.approx(
        base.downstream_fit.slope_torr_per_s - leak_rate
    )
    assert corrected.permeability.nominal_value < base.permeability.nominal_value
    assert corrected.leak_rate_torr_per_s == leak_rate
    assert corrected.leak_test_run_id is None
    assert "downstream_leak_corrected_torr" in corrected.timeseries.columns
    assert "downstream_leak_corrected_torr" not in base.timeseries.columns

    result = corrected.result_dict()
    assert result["results"]["leak"] == {
        "applied": True,
        "rate_torr_per_s": leak_rate,
        "leak_test_run_id": None,
    }
    assert base.result_dict()["results"]["leak"]["applied"] is False


def test_process_run_takes_leak_test_result_with_provenance(
    fixtures_dir, tmp_path, leak_result
):
    run_dir = convert_run(
        fixtures_dir / "pressure_only_run", tmp_path / "pressure_only_run"
    )
    corrected = process_run(load_run(run_dir), SAMPLE, leak=leak_result)
    assert corrected.leak_rate_torr_per_s == pytest.approx(LEAK_RATE_TRUE)
    assert corrected.leak_test_run_id == "leak_run"
