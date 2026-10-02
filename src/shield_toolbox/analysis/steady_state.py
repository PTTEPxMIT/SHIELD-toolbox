"""Steady-state permeation analysis: upstream plateau, downstream rise fit,
Takaishi–Sensui permeability, and Arrhenius fitting.

The plateau and rise-fit algorithms are ported unchanged from
``shield_das.analysis`` (which the live DAS uses), so results stay comparable.
Rig constants come from :class:`~shield_toolbox.config.RigConfig` instead of
being hard-coded, and the Torr→Pa factor is 133.322 rather than the legacy
133.3 (<0.02 % effect). The Takaishi–Sensui permeability deliberately departs
from the legacy code, which dropped the heated downstream volume and used
mis-scaled constants (see :func:`permeability_takaishi_sensui`). Pure physics
only: no file I/O, no plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from uncertainties import UFloat

from shield_toolbox.config import RigConfig
from shield_toolbox.constants import N_A, TORR_TO_PA, TS_A_H2, TS_B_H2, TS_C_H2, R


@dataclass(frozen=True)
class UpstreamPlateau:
    """Stable upstream pressure after the fill transient."""

    average_torr: float
    start_index: int
    """Index of the first sample of the stable region."""


def stable_upstream_pressure(
    time_s: npt.ArrayLike,
    pressure_torr: npt.ArrayLike,
    window: int = 5,
    slope_threshold_torr_per_s: float = 1e-3,
) -> UpstreamPlateau:
    """Detect when the upstream pressure stabilises after the fill and return
    the average stable pressure.

    Slopes are smoothed with a moving average; the stable region starts at the
    first sample where the absolute smoothed slope falls below the threshold,
    at least 5 s after the start. Falls back to the second half of the data if
    no stable region is found.

    Args:
        time_s: Time axis in seconds.
        pressure_torr: Upstream pressure in Torr.
        window: Moving-average window (samples) for slope smoothing.
        slope_threshold_torr_per_s: Maximum |slope| considered stable.
    """
    time_arr = np.asarray(time_s, dtype=float)
    pressure_arr = np.asarray(pressure_torr, dtype=float)

    slopes = np.gradient(pressure_arr, time_arr)
    # Cap the kernel at the trace length: np.convolve(mode="same") returns
    # max(len, window) values, which breaks masking for very short traces.
    effective_window = min(window, len(pressure_arr))
    kernel = np.ones(effective_window) / effective_window
    smoothed = np.convolve(slopes, kernel, mode="same")

    is_stable = np.abs(smoothed) < slope_threshold_torr_per_s
    after_settling = time_arr > time_arr.min() + 5
    stability_mask = is_stable & after_settling

    if stability_mask.any():
        start = int(np.argmax(stability_mask))
    else:
        start = len(time_arr) // 2

    return UpstreamPlateau(
        average_torr=float(np.mean(pressure_arr[start:])),
        start_index=start,
    )


@dataclass(frozen=True)
class DownstreamFit:
    """Weighted linear fit to the downstream pressure rise."""

    slope_torr_per_s: float
    intercept_torr: float
    used: np.ndarray
    """Boolean mask of the input samples that entered the fit."""

    def evaluate(self, time_s: npt.ArrayLike) -> np.ndarray:
        """The fitted line P(t) = slope·t + intercept, in Torr."""
        return (
            self.slope_torr_per_s * np.asarray(time_s, dtype=float)
            + self.intercept_torr
        )


def fit_downstream_rise(
    time_s: npt.ArrayLike,
    pressure_torr: npt.ArrayLike,
    min_reliable_torr: float = 0.05,
    max_reliable_torr: float = 0.95,
) -> DownstreamFit:
    """Fit the downstream pressure rise; the slope is the permeation signal.

    Samples outside the gauge's reliable range (0.05–0.95 Torr for the 1 Torr
    Baratron) are excluded. Exponential weights from exp(-1) to exp(0)
    emphasise later, more stable samples.

    Args:
        time_s: Time axis in seconds.
        pressure_torr: Downstream pressure in Torr.
        min_reliable_torr: Lower edge of the reliable gauge range.
        max_reliable_torr: Upper edge of the reliable gauge range.

    Raises:
        ValueError: If fewer than two samples are inside the reliable range.
    """
    time_arr = np.asarray(time_s, dtype=float)
    pressure_arr = np.asarray(pressure_torr, dtype=float)

    used = (pressure_arr >= min_reliable_torr) & (pressure_arr <= max_reliable_torr)
    if used.sum() < 2:
        raise ValueError(
            f"Only {int(used.sum())} downstream samples inside the reliable "
            f"range [{min_reliable_torr}, {max_reliable_torr}] Torr — cannot fit"
        )

    weights = np.exp(np.linspace(-1, 0, int(used.sum())))
    slope, intercept = np.polyfit(time_arr[used], pressure_arr[used], 1, w=weights)
    return DownstreamFit(
        slope_torr_per_s=float(slope),
        intercept_torr=float(intercept),
        used=used,
    )


def takaishi_sensui_ratio(
    pressure_torr: float,
    tube_diameter_m: float,
    hot_temperature_K: float,
    cold_temperature_K: float,
) -> tuple[float, float]:
    """Thermal-transpiration pressure ratio between a hot and a cold volume.

    Takaishi & Sensui (1963), eqn (6) with the H2 constants of their Table 1::

        p_hot / p_cold = (A·X² + B·X + C·√X + √(T_hot/T_cold)) / (A·X² + B·X + C·√X + 1)
        X = 2·p_cold·d / (T_hot + T_cold)     [Torr·mm/K]

    The ratio tends to √(T_hot/T_cold) in the free-molecular limit (X → 0) and
    to 1 in the continuum limit (X → ∞).

    Args:
        pressure_torr: Pressure measured on the cold side, Torr.
        tube_diameter_m: Diameter of the tube joining the two volumes, m.
        hot_temperature_K: Hot-side temperature, K.
        cold_temperature_K: Cold-side (gauge) temperature, K.

    Returns:
        ``(ratio, p·d(ratio)/dp)``. The second value is the logarithmic
        pressure derivative that the time derivative of the hot-side pressure
        needs.
    """
    x = (
        2.0
        * pressure_torr
        * (tube_diameter_m * 1e3)
        / (hot_temperature_K + cold_temperature_K)
    )
    q = TS_A_H2 * x**2 + TS_B_H2 * x + TS_C_H2 * np.sqrt(x)
    sqrt_t = np.sqrt(hot_temperature_K / cold_temperature_K)
    ratio = (q + sqrt_t) / (q + 1.0)
    # p·dq/dp = x·dq/dx, since x is proportional to p
    x_dq_dx = 2.0 * TS_A_H2 * x**2 + TS_B_H2 * x + 0.5 * TS_C_H2 * np.sqrt(x)
    p_dratio_dp = (1.0 - sqrt_t) / (q + 1.0) ** 2 * x_dq_dx
    return float(ratio), float(p_dratio_dp)


def permeability_takaishi_sensui(
    slope_torr_per_s: float,
    temperature_K: float,
    sample_thickness_m: float,
    downstream_pressure_torr: float,
    upstream_pressure_torr: float,
    rig: RigConfig,
) -> UFloat:
    """Permeability from the downstream rise slope, with the Takaishi–Sensui
    thermal-transpiration correction and uncertainty propagation.

    The downstream volume is split into a hot section at the sample
    temperature (fraction ``rig.v1_v2_split_ratio``) and a cold section at
    ambient, where the gauge reads p. The hot-side pressure is
    p_hot = f(p)·p, with f from :func:`takaishi_sensui_ratio`. Counting the
    moles in both sections gives the molar flow::

        dn/dt = dp/dt / R · [V_cold/T_amb + V_hot/T_sample · (f + p·df/dp)]

    Sieverts' law then gives permeability = flux · thickness / sqrt(upstream
    pressure).

    This replaces the legacy formula, which divided the hot-section term by the
    ratio's numerator alone. At the rigs' downstream pressures that numerator
    is large, so the legacy code dropped the hot section almost entirely. Its
    constants were also mis-scaled: ``10e-5`` and ``10e-2`` (1e-4 and 0.1)
    were meant as 1e-5 and 1e-2, and X was missing the 2/(T1+T2) factor.

    Args:
        slope_torr_per_s: Downstream pressure rise rate in Torr/s.
        temperature_K: Sample temperature in K.
        sample_thickness_m: Sample thickness in m.
        downstream_pressure_torr: Downstream pressure the correction is
            evaluated at (the final one), in Torr.
        upstream_pressure_torr: Stable upstream pressure in Torr.
        rig: Rig configuration providing downstream volume (with
            uncertainty), hot-side fraction, sample area, ambient temperature,
            and the connecting-tube diameter (the sample fitting diameter, as
            in the legacy analysis).

    Returns:
        Permeability with propagated uncertainty, H/(m·s·Pa^0.5).
    """
    volume = rig.downstream_volume_m3
    hot_fraction = rig.v1_v2_split_ratio
    t_ambient = rig.ambient_temperature_K

    ratio, p_dratio_dp = takaishi_sensui_ratio(
        downstream_pressure_torr, rig.sample_diameter_m, temperature_K, t_ambient
    )
    effective_volume_over_t = volume * (
        1 - hot_fraction
    ) / t_ambient + volume * hot_fraction / temperature_K * (ratio + p_dratio_dp)
    molar_flow_rate = slope_torr_per_s * TORR_TO_PA * effective_volume_over_t / R

    hydrogen_flux = molar_flow_rate / rig.sample_area_m2 * N_A  # H/(m²·s)
    return (
        hydrogen_flux
        * sample_thickness_m
        / (upstream_pressure_torr * TORR_TO_PA) ** 0.5
    )


@dataclass(frozen=True)
class ArrheniusFit:
    """Weighted Arrhenius fit, log10(perm) = slope·(1000/T) + intercept."""

    slope: float
    intercept: float
    fit_x_inverse_kK: np.ndarray
    """1000/T values spanning the input range (100 points)."""
    fit_y: np.ndarray
    """Fitted permeability at ``fit_x_inverse_kK``."""

    @property
    def activation_energy_J_per_mol(self) -> float:
        """Ea from the slope: log10(P) = -Ea/(R·ln10) · (1/T) + const."""
        return -self.slope * 1000.0 * np.log(10.0) * R

    @property
    def pre_exponential(self) -> float:
        """P0 in Perm = P0·exp(-Ea/(R·T))."""
        return float(10.0**self.intercept)


def fit_arrhenius(
    temperatures_K: npt.ArrayLike,
    permeabilities: list[UFloat] | list[float],
) -> ArrheniusFit:
    """Weighted least-squares Arrhenius fit to permeability vs temperature.

    The fit runs in log10 space against 1000/T; uncertainties are propagated
    through the log transform and used as inverse weights (points with no
    valid uncertainty get unit weight).

    Args:
        temperatures_K: Temperatures in K.
        permeabilities: Permeabilities (ufloats or floats), H/(m·s·Pa^0.5).
    """
    temps = np.asarray(temperatures_K, dtype=float)

    if hasattr(permeabilities[0], "n"):
        nominal = np.array([p.n for p in permeabilities])
        std = np.array([p.s for p in permeabilities])
    else:
        nominal = np.asarray(permeabilities, dtype=float)
        std = np.zeros_like(nominal)

    log10_perm = np.log10(nominal)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_std = std / (nominal * np.log(10))
        valid = (log10_std > 0) & np.isfinite(log10_std)
        weights = np.where(valid, 1.0 / np.where(valid, log10_std, 1.0), 1.0)

    inv_t = 1000.0 / temps
    slope, intercept = np.polyfit(inv_t, log10_perm, 1, w=weights)

    fit_x = np.linspace(inv_t.min(), inv_t.max(), 100)
    fit_y = 10.0 ** (slope * fit_x + intercept)
    return ArrheniusFit(
        slope=float(slope),
        intercept=float(intercept),
        fit_x_inverse_kK=fit_x,
        fit_y=fit_y,
    )
