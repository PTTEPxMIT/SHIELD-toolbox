"""Tests for shield_toolbox.analysis.time_varying and the run figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from shield_toolbox import convert_run, get_rig_config, load_run
from shield_toolbox.analysis import (
    apparent_permeability_vs_time,
    smoothed_pressure_rise_pa_per_s,
)
from shield_toolbox.constants import N_A, TORR_TO_PA, R
from shield_toolbox.plotting import (
    plot_permeability,
    plot_run_overview,
    plot_temperature,
)
from shield_toolbox.processing import SampleInfo, process_run

RIG = get_rig_config("v1")


def test_smoothed_dpdt_recovers_linear_slope():
    time_s = np.linspace(0, 1000, 500)
    pressure_torr = 0.02 + 0.0008 * time_s
    dpdt = smoothed_pressure_rise_pa_per_s(time_s, pressure_torr)
    # Savitzky-Golay derivative of a straight line is exact everywhere.
    np.testing.assert_allclose(dpdt, 0.0008 * TORR_TO_PA, rtol=1e-9)


def test_smoothed_dpdt_gradient_fallback_for_short_traces():
    time_s = np.array([0.0, 1.0, 2.0, 3.0])
    pressure_torr = 0.1 + 0.01 * time_s
    dpdt = smoothed_pressure_rise_pa_per_s(time_s, pressure_torr)
    np.testing.assert_allclose(dpdt, 0.01 * TORR_TO_PA, rtol=1e-9)


def test_apparent_permeability_matches_hand_formula():
    time_s = np.linspace(0, 1000, 500)
    slope_torr_per_s = 0.0008
    downstream_torr = 0.02 + slope_torr_per_s * time_s
    phi = apparent_permeability_vs_time(
        time_s,
        downstream_torr,
        upstream_pressure_torr=400.0,
        temperature_K=500.0,
        sample_thickness_m=0.00088,
        rig=RIG,
    )
    # Phi = dPdt_Pa * V * N_A / (R * T * A) * e / sqrt(P_up_Pa)
    expected = (
        (slope_torr_per_s * TORR_TO_PA)
        * RIG.downstream_volume_m3.nominal_value
        * N_A
        / (R * 500.0 * RIG.sample_area_m2)
        * 0.00088
        / np.sqrt(400.0 * TORR_TO_PA)
    )
    np.testing.assert_allclose(phi, expected, rtol=1e-9)


@pytest.fixture
def processed(fixtures_dir, tmp_path):
    run_dir = convert_run(
        fixtures_dir / "pressure_only_run", tmp_path / "pressure_only_run"
    )
    return process_run(
        load_run(run_dir), SampleInfo(substrate="316L", thickness_m=0.00088)
    )


def test_timeseries_has_apparent_permeability(processed):
    phi = processed.timeseries["apparent_permeability"].to_numpy()
    in_run = processed.timeseries["in_run"].to_numpy()
    assert np.isfinite(phi[in_run]).all()


def test_temperature_plot_smoke(processed):
    ax = plot_temperature(processed)
    assert ax.get_ylabel() == "Sample temperature (K)"
    plt.close("all")


def test_permeability_plot_smoke(processed):
    ax = plot_permeability(processed)
    assert ax.get_yscale() == "log"
    plt.close("all")


def test_overview_is_two_by_two(processed):
    axes = plot_run_overview(processed)
    assert len(axes) == 4
    titles = [ax.get_title() for ax in axes]
    assert any("temperature" in t for t in titles)
    assert any("permeability" in t for t in titles)
    plt.close("all")
