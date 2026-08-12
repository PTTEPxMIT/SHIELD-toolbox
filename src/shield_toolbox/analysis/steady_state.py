"""Steady-state permeation analysis: upstream plateau, downstream rise fit,
Takaishi–Sensui permeability, and Arrhenius fitting.

Algorithms are ported unchanged from ``shield_das.analysis`` (which the live
DAS uses) so results stay comparable; the only differences are that rig
constants come from :class:`~shield_toolbox.config.RigConfig` instead of being
hard-coded, and the Torr→Pa factor is 133.322 rather than the legacy 133.3
(<0.02 % effect). Pure physics only: no file I/O, no plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from uncertainties import UFloat

from shield_toolbox.config import RigConfig
from shield_toolbox.constants import N_A, TORR_TO_PA, TS_A, TS_B, TS_C, R


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

    Based on Takaishi & Sensui (1963), Trans. Faraday Soc. 59, 2503–2514.
    The downstream volume is split into a heated section at the sample
    temperature and an ambient section; tube-conductance corrections are
    applied to the heated contribution. Sieverts' law gives
    permeability = flux · thickness / sqrt(upstream pressure).

    Args:
        slope_torr_per_s: Downstream pressure rise rate in Torr/s.
        temperature_K: Sample temperature in K.
        sample_thickness_m: Sample thickness in m.
        downstream_pressure_torr: Final downstream pressure in Torr.
        upstream_pressure_torr: Stable upstream pressure in Torr.
        rig: Rig configuration providing downstream volume (with
            uncertainty), heated/ambient volume split, sample area, ambient
            temperature, and the connecting-tube diameter (the sample fitting
            diameter, as in the legacy analysis).

    Returns:
        Permeability with propagated uncertainty, H/(m·s·Pa^0.5).
    """
    volume = rig.downstream_volume_m3
    split = rig.v1_v2_split_ratio
    volume_heated = volume * split
    volume_ambient = volume * (1 - split)
    t_ambient = rig.ambient_temperature_K
    d = rig.sample_diameter_m  # connecting-tube diameter, as in legacy

    dp_dt_pa = slope_torr_per_s * TORR_TO_PA
    p_down_pa = downstream_pressure_torr * TORR_TO_PA
    p_up_pa = upstream_pressure_torr * TORR_TO_PA

    numerator_2 = (
        TS_C * (d * p_down_pa) ** 0.5
        + (t_ambient / temperature_K) ** 0.5
        + TS_A * d**2 * p_down_pa**2
        + TS_B * d * p_down_pa
    )
    denominator_3 = (
        TS_C * (d * p_down_pa) ** 0.5
        + TS_A * d**2 * p_down_pa**2
        + TS_B * d * p_down_pa
        + 1
    )
    numerator_1 = (
        TS_B * d * dp_dt_pa
        + (TS_C * d * dp_dt_pa) / (2 * (d * p_down_pa) ** 0.5)
        + 2 * TS_A * d**2 * p_down_pa * dp_dt_pa
    )

    molar_flow_rate = (
        (volume_ambient * dp_dt_pa) / (R * t_ambient)
        + (volume_heated * dp_dt_pa) / (R * temperature_K * numerator_2)
        + (volume_heated * p_down_pa * numerator_1)
        / (R * temperature_K * numerator_2 * denominator_3)
        - (volume_heated * p_down_pa * numerator_1)
        / (R * temperature_K * numerator_2**2)
    )

    hydrogen_flux = molar_flow_rate / rig.sample_area_m2 * N_A  # H/(m²·s)
    return hydrogen_flux * sample_thickness_m / p_up_pa**0.5


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
