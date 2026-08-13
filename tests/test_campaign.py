"""Tests for shield_toolbox.campaign — result aggregation and Arrhenius fits."""

import json
from pathlib import Path

import numpy as np
import pytest

from shield_toolbox.campaign import arrhenius, load_results
from shield_toolbox.constants import R
from shield_toolbox.plotting import plot_arrhenius

EA_TRUE = 60_000.0  # J/mol
PHI0_TRUE = 5.0e17


def _write_result(
    base: Path,
    run_id: str,
    temperature_K: float,
    permeability: float,
    substrate: str = "316L steel",
    coating: str = "none",
    diffusivity: float | None = 1e-10,
) -> None:
    solubility = None if diffusivity is None else permeability / diffusivity
    result = {
        "run_id": run_id,
        "sample": {
            "substrate": substrate,
            "coating": coating,
            "thickness_m": 0.00088,
            "coating_layers": [],
        },
        "run_info": {"furnace_setpoint": temperature_K - 273.15},
        "temperature": {
            "sample_temperature_K": temperature_K,
            "source": "thermocouple",
        },
        "results": {
            "upstream_pressure_torr": 740.0,
            "permeability": {
                "nominal": permeability,
                "std_dev": 0.1 * permeability,
                "units": "H/(m·s·Pa^0.5)",
            },
            "time_lag": {
                "time_lag_s": None if diffusivity is None else 1000.0,
                "permeation_start_s": 60.0,
                "permeation_start_source": "v3_open_time",
                "baseline_torr": 0.02,
            },
            "diffusivity": {"value": diffusivity, "units": "m^2/s"},
            "solubility": {
                "nominal": solubility,
                "std_dev": None if solubility is None else 0.1 * solubility,
                "units": "H/(m^3·Pa^0.5)",
            },
        },
    }
    run_dir = base / substrate.replace(" ", "_") / coating / run_id
    run_dir.mkdir(parents=True)
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f)


@pytest.fixture
def campaign_dir(tmp_path):
    """Four runs on a perfect Arrhenius line, plus one other-substrate run."""
    for index, temp in enumerate([600.0, 700.0, 800.0, 900.0]):
        phi = PHI0_TRUE * np.exp(-EA_TRUE / (R * temp))
        _write_result(tmp_path, f"run_{index}", temp, phi)
    _write_result(
        tmp_path, "run_other", 700.0, 1e12, substrate="carbon steel", diffusivity=None
    )
    return tmp_path


def test_load_results_table(campaign_dir):
    results = load_results(campaign_dir)
    assert len(results) == 5
    # Sorted by temperature; both substrates present.
    assert results["temperature_K"].is_monotonic_increasing
    assert set(results["substrate"]) == {"316L steel", "carbon steel"}
    # The run without a valid time lag has NaN diffusivity/solubility.
    other = results[results["run_id"] == "run_other"].iloc[0]
    assert np.isnan(other["diffusivity_m2_per_s"])
    assert np.isnan(other["solubility"])


def test_load_results_filters(campaign_dir):
    results = load_results(campaign_dir, substrate="316L steel", coating="none")
    assert len(results) == 4
    with pytest.raises(FileNotFoundError, match="substrate"):
        load_results(campaign_dir, substrate="unobtainium")


def test_arrhenius_recovers_activation_energy(campaign_dir):
    results = load_results(campaign_dir, substrate="316L steel")
    fit = arrhenius(results)
    assert fit.activation_energy_J_per_mol == pytest.approx(EA_TRUE, rel=1e-3)
    assert fit.pre_exponential == pytest.approx(PHI0_TRUE, rel=0.01)


def test_arrhenius_quantities_and_errors(campaign_dir):
    results = load_results(campaign_dir, substrate="316L steel")
    # Diffusivity is constant in the fixture → Ea ≈ 0.
    d_fit = arrhenius(results, quantity="diffusivity")
    assert d_fit.activation_energy_J_per_mol == pytest.approx(0.0, abs=1.0)
    # Solubility = Φ/D inherits Φ's activation energy.
    s_fit = arrhenius(results, quantity="solubility")
    assert s_fit.activation_energy_J_per_mol == pytest.approx(EA_TRUE, rel=1e-3)
    with pytest.raises(ValueError, match="Unknown quantity"):
        arrhenius(results, quantity="viscosity")


def test_arrhenius_needs_two_finite_rows(campaign_dir):
    results = load_results(campaign_dir, substrate="carbon steel")
    with pytest.raises(ValueError, match="at least two"):
        arrhenius(results, quantity="diffusivity")


def test_plot_arrhenius_draws_groups_and_fit(campaign_dir):
    results = load_results(campaign_dir)
    fit = arrhenius(load_results(campaign_dir, substrate="316L steel"))
    ax = plot_arrhenius(results, fit=fit)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("316L steel / none" in label for label in labels)
    assert any("$E_a$" in label for label in labels)
    assert ax.get_yscale() == "log"
