"""Time-resolved (apparent) permeability from the downstream pressure rise.

Same convention as the permeation-barrier analysis
(``festim_pressure_rise.py`` / ``SHIELD_analysis_timevarying_perm.ipynb``):
the downstream dP/dt is smoothed with a Savitzky–Golay derivative filter,
converted to an instantaneous atom flux with the ideal-gas relation (no
thermal-transpiration correction — this is the *apparent* permeability), and
scaled by Sieverts' law::

    J(t)   = dP/dt · V / (R · T · A) · N_A        [H/(m²·s)]
    Phi(t) = J(t) · e / sqrt(P_up)                [H/(m·s·Pa^0.5)]
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

from shield_toolbox.config import RigConfig
from shield_toolbox.constants import N_A, TORR_TO_PA, R

SAVGOL_WINDOW = 155
"""Default Savitzky–Golay window (samples); forced odd, capped to the trace."""
SAVGOL_POLYORDER = 3


def smoothed_pressure_rise_pa_per_s(
    time_s: npt.ArrayLike,
    pressure_torr: npt.ArrayLike,
    window: int = SAVGOL_WINDOW,
    polyorder: int = SAVGOL_POLYORDER,
) -> np.ndarray:
    """Smoothed downstream dP/dt in Pa/s (Savitzky–Golay derivative).

    Falls back to a plain gradient for traces shorter than the window. The
    filter assumes a uniform sample spacing (the median dt is used), which
    holds for recorded runs.
    """
    time_arr = np.asarray(time_s, dtype=float)
    pressure_pa = np.asarray(pressure_torr, dtype=float) * TORR_TO_PA

    if len(time_arr) < max(window, polyorder + 2):
        return np.gradient(pressure_pa, time_arr)

    if window % 2 == 0:
        window += 1
    dt = float(np.median(np.diff(time_arr)))
    return savgol_filter(
        pressure_pa,
        window_length=window,
        polyorder=polyorder,
        deriv=1,
        delta=dt,
        mode="interp",
    )


def apparent_permeability_vs_time(
    time_s: npt.ArrayLike,
    downstream_torr: npt.ArrayLike,
    upstream_pressure_torr: float,
    temperature_K: float,
    sample_thickness_m: float,
    rig: RigConfig,
    window: int = SAVGOL_WINDOW,
) -> np.ndarray:
    """Instantaneous apparent permeability trace, H/(m·s·Pa^0.5).

    Args:
        time_s: Time axis in seconds.
        downstream_torr: Downstream pressure in Torr.
        upstream_pressure_torr: Stable upstream pressure in Torr.
        temperature_K: Sample temperature in K.
        sample_thickness_m: Sample thickness in m.
        rig: Rig configuration (downstream volume and sample area; nominal
            values are used — no uncertainty propagation per sample).
        window: Savitzky–Golay smoothing window in samples.

    Returns:
        Phi(t), same length as ``time_s``. Values can be negative where the
        smoothed dP/dt dips below zero (gauge noise before breakthrough);
        mask or log-plot as needed.
    """
    dpdt_pa = smoothed_pressure_rise_pa_per_s(time_s, downstream_torr, window=window)
    volume = rig.downstream_volume_m3.nominal_value
    flux = dpdt_pa * volume * N_A / (R * temperature_K * rig.sample_area_m2)
    p_up_pa = upstream_pressure_torr * TORR_TO_PA
    return flux * sample_thickness_m / np.sqrt(p_up_pa)
