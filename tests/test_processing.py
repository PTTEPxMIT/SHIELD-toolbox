"""Tests for shield_toolbox.processing — process_run and the stored artifact."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shield_toolbox import get_rig_config
from shield_toolbox.constants import TORR_TO_PA
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
T_STEP_S = 100.0
BASELINE_PA = 5.0


def _fickian_rise(t_rel: np.ndarray, tau_s: float, s_inf: float) -> np.ndarray:
    """Exact permeated pressure for a step at t_rel = 0; its asymptote is
    s_inf·(t_rel − tau_s)."""
    theta = np.clip(t_rel, 0.0, None) / (6.0 * tau_s)  # D·t/e²
    n = np.arange(1, 80)[:, None]
    series = ((-1.0) ** n / n**2) * np.exp(-(n**2) * np.pi**2 * theta[None, :])
    q = theta - 1.0 / 6.0 - (2.0 / np.pi**2) * series.sum(axis=0)
    q[t_rel <= 0] = 0.0
    return s_inf * 6.0 * tau_s * q


def _permeation_run(
    tau_s: float = 3 * 3600,
    s_inf: float = 1.5e-3,
    background: float = 1e-4,
    duration_s: float = 20 * 3600,
    dt_s: float = 10.0,
    noise_pa: float = 0.01,
    date: str = "2026-09-25",
    thermocouple: bool = True,
    saturate_at_s: float | None = None,
) -> PermeationRun:
    """Synthetic run: upstream steps to 500 Torr at T_STEP_S; the downstream is
    a baseline plus a linear background (Pa/s) plus a Fickian rise (S∞ in
    Pa/s, time lag τ), with Gaussian noise."""
    rng = np.random.default_rng(0)
    time_s = np.arange(0.0, duration_s, dt_s)
    t_rel = time_s - T_STEP_S
    upstream_torr = np.where(time_s < T_STEP_S, 0.0, 500.0)
    downstream_pa = (
        BASELINE_PA
        + background * t_rel
        + _fickian_rise(t_rel, tau_s, s_inf)
        + rng.normal(0.0, noise_pa, len(time_s))
    )
    downstream_v = downstream_pa / TORR_TO_PA * 10.0  # 1 Torr full scale
    thermocouple_mv = np.full(len(time_s), 12.2)
    if saturate_at_s is not None:
        saturated = time_s >= saturate_at_s
        downstream_v[saturated] = 10.12
        thermocouple_mv[saturated] = -101.19  # breaks once the gauge saturates
    metadata = {
        "version": "1.4",
        "run_info": {
            "date": date,
            "start_time": f"{date}T10:00:00",
            "run_type": "permeation_exp",
            "furnace_setpoint": 500,
            "sample_substrate": "316L steel",
            "sample_coating": "none",
            "sample_thickness": 0.00098,
            "sample_coating_layers": [],
        },
        "gauges": [
            {
                "name": "Baratron626D_1KT",
                "type": "Baratron626D_Gauge",
                "gauge_location": "upstream",
                "full_scale_torr": 1000.0,
            },
            {
                "name": "Baratron626D_1T",
                "type": "Baratron626D_Gauge",
                "gauge_location": "downstream",
                "full_scale_torr": 1.0,
            },
        ],
        "thermocouples": [{"name": "furnace_thermocouple"}] if thermocouple else [],
    }
    return PermeationRun(
        path=Path("synthetic_run"),
        run_id="synthetic_run",
        metadata=metadata,
        timestamps=np.datetime64(f"{date}T10:00:00")
        + (time_s * 1000).astype("timedelta64[ms]"),
        time_s=time_s,
        gauge_voltages={
            "Baratron626D_1KT": upstream_torr / 100.0,
            "Baratron626D_1T": downstream_v,
        },
        gauge_locations={
            "Baratron626D_1KT": "upstream",
            "Baratron626D_1T": "downstream",
        },
        thermocouple_mv={"furnace_thermocouple": thermocouple_mv}
        if thermocouple
        else {},
        valve_times_s={"v3_open_time": T_STEP_S},
    )


@pytest.fixture(scope="module")
def processed():
    return process_run(_permeation_run(), SAMPLE)


def test_rig_resolved_from_run_date(processed):
    assert processed.rig is get_rig_config("v2")


def test_recovers_background_and_time_lag(processed):
    assert processed.initial_time_s == pytest.approx(T_STEP_S)
    # Slow rise: the 10 min blocks find the onset, the noise recording stops
    # a quarter of the pre-rise time before it.
    noise = processed.noise
    assert noise.onset_from == "10 min blocks"
    assert noise.start_s == 60.0
    assert noise.end_s == pytest.approx(0.75 * noise.onset_s)
    assert 1800 < noise.onset_s < 3 * 3600
    assert processed.background.slope.nominal_value == pytest.approx(1e-4, rel=0.1)
    assert processed.background.level.nominal_value == pytest.approx(
        BASELINE_PA, abs=0.05
    )
    # Steady state from 3 τ_L to the end of the run.
    assert processed.steady_state.slope == pytest.approx(1.5e-3, rel=0.01)
    assert processed.time_lag_s == pytest.approx(3 * 3600, rel=0.03)
    start_s, end_s = processed.steady_state_window_s
    assert start_s == pytest.approx(3 * processed.time_lag_s, abs=10.0)
    assert end_s == pytest.approx(20 * 3600 - T_STEP_S, abs=10.0)


def test_derived_properties(processed):
    e = SAMPLE.thickness_m
    assert processed.diffusivity_m2_per_s == pytest.approx(
        e**2 / (6 * processed.time_lag_s)
    )
    assert processed.solubility.nominal_value == pytest.approx(
        processed.permeability.nominal_value / processed.diffusivity_m2_per_s
    )
    assert processed.permeability.std_dev > 0
    # Upstream used defaults to the measured mean over the window.
    assert processed.upstream_pressure_torr == pytest.approx(500.0)
    assert processed.upstream_pressure_measured_torr == pytest.approx(500.0)
    assert processed.temperature_source == "thermocouple"


def test_fast_rise_uses_one_minute_onset():
    processed = process_run(
        _permeation_run(tau_s=900, s_inf=1e-2, duration_s=3 * 3600, dt_s=2.0),
        SAMPLE,
    )
    noise = processed.noise
    assert noise.onset_from == "1 min blocks"
    assert noise.onset_s < 1200
    assert noise.end_s == pytest.approx(0.75 * noise.onset_s)
    assert processed.time_lag_s == pytest.approx(900, rel=0.05)


def test_settings_and_refit(processed):
    fixed = processed.refit(upstream_pressure_torr=400.0)
    assert fixed.upstream_pressure_torr == 400.0
    assert fixed.upstream_pressure_measured_torr == pytest.approx(500.0)
    # Φ ∝ 1/√P_up
    assert fixed.permeability.nominal_value == pytest.approx(
        processed.permeability.nominal_value * np.sqrt(500.0 / 400.0), rel=1e-6
    )

    no_background = processed.refit(background_slope_pa_per_s=0.0)
    assert no_background.background_slope_pa_per_s == 0.0
    assert no_background.steady_state.slope == pytest.approx(
        processed.steady_state.slope + processed.background_slope_pa_per_s, rel=0.01
    )

    early = processed.refit(steady_state_start_taus=2.0)
    assert early.steady_state_window_s[0] < processed.steady_state_window_s[0]

    short = processed.refit(analysis_hours=15)
    assert short.steady_state_window_s[1] <= 15 * 3600
    assert short.settings.analysis_hours == 15


def test_noise_margin_fraction_setting(processed):
    half = processed.refit(noise_margin_fraction=0.5)
    assert half.noise.end_s == pytest.approx(0.5 * half.noise.onset_s)


def test_manual_noise_end():
    processed = process_run(_permeation_run(), SAMPLE, noise_end_s=1200.0)
    assert processed.noise.end_s == 1200.0
    assert processed.noise.onset_from == "manual"


def test_timeseries_columns(processed):
    ts = processed.timeseries
    expected = {
        "timestamp",
        "time_s",
        "Baratron626D_1KT_voltage_V",
        "Baratron626D_1T_voltage_V",
        "upstream_torr",
        "upstream_err_torr",
        "downstream_torr",
        "downstream_err_torr",
        "temperature_K",
        "in_run",
        "time_since_init_s",
        "downstream_pa",
        "downstream_filtered_pa",
        "noise_recording",
        "usable",
        "fit_used",
    }
    assert expected <= set(ts.columns)
    np.testing.assert_allclose(
        ts["upstream_torr"], ts["Baratron626D_1KT_voltage_V"] * 100.0
    )
    # The filtered signal starts at ~0 at t_init.
    at_init = ts["time_since_init_s"].abs() < 30
    assert ts.loc[at_init, "downstream_filtered_pa"].abs().max() < 0.05
    assert ts["in_run"].all()


def test_result_dict_contents(processed):
    result = processed.result_dict()
    assert result["run_id"] == "synthetic_run"
    assert result["sample"] == {
        "substrate": "316L",
        "coating": "Al2O3",
        "thickness_m": 0.00088,
        "sample_id": "S-01",
        "coating_layers": [],
    }
    assert result["provenance"]["rig_version"] == "v2"
    assert result["provenance"]["settings"]["steady_state_start_taus"] == 3.0
    results = result["results"]
    assert results["initial_time_s"] == pytest.approx(T_STEP_S)
    assert results["noise_recording"]["onset_from"] == "10 min blocks"
    assert results["background"]["slope_pa_per_s"]["std_dev"] > 0
    assert results["steady_state"]["slope_pa_per_s"] == processed.steady_state.slope
    assert results["permeability"]["units"] == "H/(m·s·Pa^0.5)"
    assert results["time_lag"]["time_lag_s"] == processed.time_lag_s
    assert results["solubility"]["nominal"] > 0
    assert result["run_info"]["valve_times_s"]["v3_open_time"] == T_STEP_S
    json.dumps(result)  # serialisable


def test_write_creates_material_tree(processed, tmp_path):
    out_dir = processed.write(tmp_path)
    assert out_dir == tmp_path / "316L" / "Al2O3" / "synthetic_run"
    ts = pd.read_parquet(out_dir / TIMESERIES_FILENAME)
    assert ts["fit_used"].dtype == bool
    assert len(ts) == len(processed.timeseries)
    with open(out_dir / RESULT_FILENAME) as f:
        assert json.load(f)["sample"]["substrate"] == "316L"


def test_temperature_fallback_uses_setpoint_offset():
    # Old-rig run without a thermocouple: 500 °C setpoint, v1 offset −18 K.
    processed = process_run(
        _permeation_run(date="2025-10-06", thermocouple=False), SAMPLE
    )
    assert processed.rig is get_rig_config("v1")
    assert processed.sample_temperature_K == pytest.approx(755.15)
    assert processed.temperature_source == "furnace_setpoint_offset"


def test_window_ends_before_downstream_saturation():
    run = _permeation_run(saturate_at_s=15 * 3600)
    processed = process_run(run, SAMPLE)  # must not trip over the broken thermocouple
    ts = processed.timeseries
    first_saturated = int(np.argmax(run.gauge_voltages["Baratron626D_1T"] >= 10.0))
    assert ts["in_run"].iloc[:first_saturated].all()
    assert not ts["in_run"].iloc[first_saturated:].any()
    assert not ts["usable"].iloc[first_saturated:].any()
    # Temperature NaN once the reading breaks; the analysis T stays finite.
    assert ts["temperature_K"].iloc[first_saturated:].isna().all()
    assert np.isfinite(processed.sample_temperature_K)
    assert processed.steady_state_window_s[1] < 15 * 3600 - T_STEP_S


def test_process_run_defaults_to_metadata_sample():
    processed = process_run(_permeation_run())
    assert processed.sample.substrate == "316L steel"
    assert processed.sample.thickness_m == 0.00098


def test_process_run_without_sample_or_metadata_raises():
    run = _permeation_run()
    del run.metadata["run_info"]["sample_substrate"]
    with pytest.raises(ValueError, match="no sample description"):
        process_run(run)


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
