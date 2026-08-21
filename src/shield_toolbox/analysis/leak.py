"""Background leak-rate measurement from a leak-test run.

A leak test is a short recorded run taken with the sample installed and
sealed, the upstream side unpressurized, and the downstream volume isolated
from the pump at a setpoint inside the 1 Torr Baratron's range. Any downstream
pressure rise is then background — seal leakage plus outgassing, not
permeation — and its rate is the offset subtracted from the downstream rise
of subsequent permeation runs on the same sample (``process_run(...,
leak=...)``).

Unlike the permeation rise fit, the leak fit is a plain unweighted straight
line over the isolated window: a constant background rate has no "later
samples are more settled" structure to weight for, and the trace may sit
below the permeation fit's 0.05 Torr reliability floor.

Pure physics only: no file I/O, no plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from uncertainties import UFloat

from shield_toolbox.config import RigConfig
from shield_toolbox.constants import TORR_TO_PA, R


@dataclass(frozen=True)
class LeakRateFit:
    """Linear fit P(t) = rate·t + intercept to an isolated downstream trace."""

    rate_torr_per_s: float
    """The background leak rate — the quantity used as a permeation offset."""
    intercept_torr: float
    mean_pressure_torr: float
    """Mean downstream pressure over the fitted window (the leak was measured
    at this pressure; the offset is most valid near it)."""
    r_squared: float
    """Coefficient of determination of the fit; near 1 for a clean linear
    rise, near 0 when the trace is flat noise (leak-free)."""
    used: np.ndarray
    """Boolean mask of the input samples that entered the fit."""

    def evaluate(self, time_s: npt.ArrayLike) -> np.ndarray:
        """The fitted line P(t) = rate·t + intercept, in Torr."""
        return (
            self.rate_torr_per_s * np.asarray(time_s, dtype=float) + self.intercept_torr
        )


def fit_leak_rate(
    time_s: npt.ArrayLike,
    pressure_torr: npt.ArrayLike,
    start_s: float = 0.0,
    min_torr: float | None = None,
    max_torr: float | None = None,
) -> LeakRateFit:
    """Fit the background leak rate to an isolated downstream pressure trace.

    Args:
        time_s: Time axis in seconds.
        pressure_torr: Downstream pressure in Torr, same length.
        start_s: Start of the measurement window (normally the
            ``downstream_isolated_time`` event); earlier samples — pump-down
            and setpoint settling — are excluded.
        min_torr: Optional lower pressure bound on fitted samples.
        max_torr: Optional upper pressure bound on fitted samples.

    Raises:
        ValueError: If fewer than two samples fall inside the window.
    """
    time_arr = np.asarray(time_s, dtype=float)
    pressure_arr = np.asarray(pressure_torr, dtype=float)

    used = time_arr >= start_s
    if min_torr is not None:
        used &= pressure_arr >= min_torr
    if max_torr is not None:
        used &= pressure_arr <= max_torr
    if used.sum() < 2:
        raise ValueError(
            f"Only {int(used.sum())} samples in the leak measurement window "
            f"(t >= {start_s:.1f} s) — cannot fit a leak rate"
        )

    slope, intercept = np.polyfit(time_arr[used], pressure_arr[used], 1)
    residuals = pressure_arr[used] - (slope * time_arr[used] + intercept)
    total = pressure_arr[used] - pressure_arr[used].mean()
    total_ss = float(np.sum(total**2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / total_ss if total_ss > 0 else 0.0

    return LeakRateFit(
        rate_torr_per_s=float(slope),
        intercept_torr=float(intercept),
        mean_pressure_torr=float(np.mean(pressure_arr[used])),
        r_squared=r_squared,
        used=used,
    )


def leak_molar_rate_mol_per_s(
    rate_torr_per_s: float,
    rig: RigConfig,
    temperature_K: float | None = None,
) -> UFloat | float:
    """The leak rate as a molar gas-accumulation rate, n̊ = (dP/dt)·V/(R·T).

    Informational — the permeation correction itself is applied in Torr/s.
    Uncertainty in the rig's downstream volume propagates through.

    Args:
        rate_torr_per_s: Fitted leak rate in Torr/s.
        rig: Rig configuration providing the downstream volume.
        temperature_K: Gas temperature in K; defaults to the rig's ambient
            temperature (a leak test is normally run with the furnace cold).
    """
    if temperature_K is None:
        temperature_K = rig.ambient_temperature_K
    return rate_torr_per_s * TORR_TO_PA * rig.downstream_volume_m3 / (R * temperature_K)
