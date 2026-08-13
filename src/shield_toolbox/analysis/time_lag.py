"""Time-lag extraction of diffusivity and solubility.

The classic permeation time-lag method (Daynes/Barrer): once permeation
reaches steady state, the downstream pressure rises linearly. Extrapolating
that steady-state line back to the pre-breakthrough baseline pressure gives
the time-axis intercept; the *time lag* τ is that intercept measured from the
moment upstream pressure was first applied to the sample (the loading valve
opening). For a membrane of thickness e with Fickian diffusion::

    τ = e² / (6·D)      →      D = e² / (6·τ)         [m²/s]
    S = Φ / D                                          [H/(m³·Pa^0.5)]

so one run yields permeability Φ (from the slope), diffusivity D (from the
intercept), and solubility S (their ratio).

Pure physics only: no file I/O, no plotting. ``process_run`` wires these into
the standard pipeline and stores the results in ``result.json``.
"""

from __future__ import annotations

import numpy as np
from uncertainties import UFloat

from shield_toolbox.analysis.steady_state import DownstreamFit


def time_lag_from_fit(
    fit: DownstreamFit,
    baseline_torr: float,
    permeation_start_s: float,
) -> float:
    """Time lag τ (s) from the steady-state downstream fit.

    The fitted line P(t) = slope·t + intercept (t in the run's time axis)
    crosses the baseline pressure at t₀ = (baseline − intercept)/slope; the
    time lag is t₀ minus the moment permeation started.

    Args:
        fit: Steady-state downstream rise fit
            (:func:`~shield_toolbox.analysis.steady_state.fit_downstream_rise`).
        baseline_torr: Downstream pressure before breakthrough, in Torr.
        permeation_start_s: When upstream pressure was applied to the sample
            (normally the loading-valve opening, e.g. ``v3_open_time``), on
            the same time axis as the fit.

    Returns:
        τ in seconds. Can be non-positive for degenerate inputs (e.g. a
        baseline above the fitted line at the start time) — validate with
        ``τ > 0`` before deriving a diffusivity.

    Raises:
        ValueError: If the fitted slope is not positive (no rising signal —
            the time-axis crossing is undefined).
    """
    if not fit.slope_torr_per_s > 0:
        raise ValueError(
            f"Downstream fit slope is {fit.slope_torr_per_s:.3e} Torr/s; "
            "the time-lag intercept needs a rising steady-state signal"
        )
    crossing_s = (baseline_torr - fit.intercept_torr) / fit.slope_torr_per_s
    return float(crossing_s - permeation_start_s)


def diffusivity_from_time_lag(time_lag_s: float, sample_thickness_m: float) -> float:
    """Diffusivity D = e²/(6·τ), in m²/s.

    Args:
        time_lag_s: Time lag τ in seconds; must be positive.
        sample_thickness_m: Sample thickness e in m.

    Raises:
        ValueError: If ``time_lag_s`` is not positive.
    """
    if not time_lag_s > 0:
        raise ValueError(f"Time lag must be positive, got {time_lag_s:.3g} s")
    return sample_thickness_m**2 / (6.0 * time_lag_s)


def solubility_from_permeability(
    permeability: UFloat | float, diffusivity_m2_per_s: float
) -> UFloat | float:
    """Solubility S = Φ/D, in H/(m³·Pa^0.5).

    Args:
        permeability: Permeability Φ in H/(m·s·Pa^0.5) (uncertainty
            propagates through if a ufloat).
        diffusivity_m2_per_s: Diffusivity D in m²/s; must be positive.

    Raises:
        ValueError: If ``diffusivity_m2_per_s`` is not positive.
    """
    if not diffusivity_m2_per_s > 0:
        raise ValueError(
            f"Diffusivity must be positive, got {diffusivity_m2_per_s:.3g} m²/s"
        )
    return permeability / diffusivity_m2_per_s


def downstream_baseline_torr(
    time_s: np.ndarray,
    downstream_torr: np.ndarray,
    permeation_start_s: float,
    window_s: float = 30.0,
) -> float:
    """Downstream baseline pressure: the median over a short window after
    permeation starts.

    The median over ``[permeation_start_s, permeation_start_s + window_s]``
    rejects single-sample spikes; before breakthrough (which takes at least
    the time lag) the downstream pressure is flat, so the window only needs
    to be short. Falls back to interpolating at ``permeation_start_s`` when
    no samples fall inside the window.

    Args:
        time_s: Time axis in seconds.
        downstream_torr: Downstream pressure in Torr, same length.
        permeation_start_s: Start of the baseline window.
        window_s: Window length in seconds.
    """
    time_arr = np.asarray(time_s, dtype=float)
    pressure_arr = np.asarray(downstream_torr, dtype=float)
    in_window = (time_arr >= permeation_start_s) & (
        time_arr <= permeation_start_s + window_s
    )
    if in_window.any():
        return float(np.median(pressure_arr[in_window]))
    return float(np.interp(permeation_start_s, time_arr, pressure_arr))
