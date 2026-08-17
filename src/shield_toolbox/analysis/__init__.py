"""Pure physics functions: no file I/O, no plotting."""

from shield_toolbox.analysis.evacuation import EvacuationFit, fit_evacuation
from shield_toolbox.analysis.furnace import furnace_temperature_offset
from shield_toolbox.analysis.regime import (
    dissociation_coeff,
    fit_pressure_exponent,
    recombination_coeff,
)
from shield_toolbox.analysis.steady_state import (
    ArrheniusFit,
    DownstreamFit,
    UpstreamPlateau,
    fit_arrhenius,
    fit_downstream_rise,
    permeability_takaishi_sensui,
    stable_upstream_pressure,
)
from shield_toolbox.analysis.time_lag import (
    diffusivity_from_time_lag,
    downstream_baseline_torr,
    solubility_from_permeability,
    time_lag_from_fit,
)
from shield_toolbox.analysis.time_varying import (
    apparent_permeability_vs_time,
    smoothed_pressure_rise_pa_per_s,
)
from shield_toolbox.analysis.window import run_window_mask

__all__ = [
    "ArrheniusFit",
    "DownstreamFit",
    "EvacuationFit",
    "UpstreamPlateau",
    "apparent_permeability_vs_time",
    "diffusivity_from_time_lag",
    "dissociation_coeff",
    "downstream_baseline_torr",
    "fit_arrhenius",
    "fit_downstream_rise",
    "fit_evacuation",
    "fit_pressure_exponent",
    "furnace_temperature_offset",
    "permeability_takaishi_sensui",
    "recombination_coeff",
    "run_window_mask",
    "smoothed_pressure_rise_pa_per_s",
    "solubility_from_permeability",
    "stable_upstream_pressure",
    "time_lag_from_fit",
]
