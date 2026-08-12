"""Pump-down (evacuation) modelling: fit a pressure-decay trace and predict
the time to reach a target pressure.

For a volume V pumped at speed q the ideal pump-down is
``p(t) = p₀·exp(−t·q/V)``; real traces settle to a base pressure, so the
model fitted here is::

    p(t) = A·exp(−B·(t + C)) + D

with amplitude A, rate B = q/V, time offset C, and base (asymptotic)
pressure D. Ported from the legacy ``EvacuationPrediction.ipynb``; the
volume-ratio extrapolation now scales the *rate* (B → B/ratio, i.e. the
time constant V/q grows linearly with volume, matching the pump equation)
rather than the amplitude as the notebook did.

Times are in seconds throughout, matching the rest of the toolbox — convert
minute-based data before fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class EvacuationFit:
    """Fitted pump-down model p(t) = A·exp(−B·(t + C)) + D (Torr, seconds)."""

    amplitude_torr: float
    """A — the decaying amplitude above the base pressure."""
    rate_per_s: float
    """B — the decay rate q/V; 1/B is the pump-down time constant."""
    time_offset_s: float
    """C — time offset of the fitted decay."""
    base_pressure_torr: float
    """D — the asymptotic (ultimate) pressure."""

    def pressure_at(self, time_s: npt.ArrayLike) -> np.ndarray:
        """Modelled pressure (Torr) at ``time_s`` seconds."""
        time_arr = np.asarray(time_s, dtype=float)
        return (
            self.amplitude_torr
            * np.exp(-self.rate_per_s * (time_arr + self.time_offset_s))
            + self.base_pressure_torr
        )

    def time_to_reach(self, target_torr: float) -> float:
        """Seconds until the pressure first reaches ``target_torr``.

        Raises:
            ValueError: If the target is at or below the fitted base
                pressure (never reached), or above the starting pressure.
        """
        above_base = target_torr - self.base_pressure_torr
        if above_base <= 0:
            raise ValueError(
                f"Target {target_torr:.3g} Torr is at or below the fitted base "
                f"pressure {self.base_pressure_torr:.3g} Torr — never reached"
            )
        if above_base > self.amplitude_torr:
            raise ValueError(
                f"Target {target_torr:.3g} Torr is above the fitted starting "
                "pressure — already below it at t = 0"
            )
        return float(
            -np.log(above_base / self.amplitude_torr) / self.rate_per_s
            - self.time_offset_s
        )

    def for_volume_ratio(self, volume_ratio: float) -> EvacuationFit:
        """The predicted fit for a system ``volume_ratio`` times larger.

        The pump-down time constant V/q scales linearly with volume, so the
        rate B scales as 1/ratio (and the time offset with it). Pump speed,
        starting pressure, and base pressure are assumed unchanged.
        """
        if not volume_ratio > 0:
            raise ValueError(f"volume_ratio must be positive, got {volume_ratio}")
        return replace(
            self,
            rate_per_s=self.rate_per_s / volume_ratio,
            time_offset_s=self.time_offset_s * volume_ratio,
        )


def fit_evacuation(
    time_s: npt.ArrayLike,
    pressure_torr: npt.ArrayLike,
    skip_samples: int = 0,
) -> EvacuationFit:
    """Fit the pump-down model to a measured pressure-decay trace.

    The fit is weighted by 1/pressure² (as in the legacy analysis), so the
    low-pressure tail — the part that matters for predicting long pump-downs
    — dominates over the initial high-pressure transient.

    Args:
        time_s: Time axis in seconds.
        pressure_torr: Measured pressure in Torr, same length.
        skip_samples: Leading samples to drop (roughing-phase artefacts;
            the legacy notebook dropped the first 10).

    Raises:
        ValueError: If fewer than 5 samples remain after skipping.
        RuntimeError: If the fit does not converge.
    """
    time_arr = np.asarray(time_s, dtype=float)[skip_samples:]
    pressure_arr = np.asarray(pressure_torr, dtype=float)[skip_samples:]
    if len(pressure_arr) < 5:
        raise ValueError(
            f"Need at least 5 samples to fit, got {len(pressure_arr)} "
            f"after skipping {skip_samples}"
        )

    span_s = max(time_arr[-1] - time_arr[0], 1.0)
    initial_guess = [
        max(pressure_arr[0] - pressure_arr[-1], 1e-12),  # A
        3.0 / span_s,  # B — settle within roughly the trace span
        0.0,  # C
        max(pressure_arr[-1], 1e-12),  # D
    ]
    parameters, _ = curve_fit(
        lambda t, a, b, c, d: a * np.exp(-b * (t + c)) + d,
        time_arr,
        pressure_arr,
        p0=initial_guess,
        sigma=pressure_arr**2,
        absolute_sigma=False,
        maxfev=1_000_000,
    )
    return EvacuationFit(
        amplitude_torr=float(parameters[0]),
        rate_per_s=float(parameters[1]),
        time_offset_s=float(parameters[2]),
        base_pressure_torr=float(parameters[3]),
    )
