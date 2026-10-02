"""Legacy time-lag method for runs on the original (v1) rig.

The method of ``SHIELD_analysis.ipynb`` (``ShieldRunsAnalysis``), which
produced the original-rig results:

- **Rise slope.** Keep the downstream samples inside the gauge's reliable
  band (0.05–0.95 Torr) and fit an unweighted straight line to the last 25 %
  of them.
- **Time lag.** τ is where that line crosses P = 0, measured from the first
  sample of the recording (no initial-time detection, no baseline, no
  background subtraction).
- **Upstream pressure and temperature.** Means over the last 75 % of the
  recording.

Pure physics only: no file I/O, no plotting. ``process_legacy_run`` wires
these into a stored result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

LEGACY_MIN_TORR = 0.05
"""Lower edge of the downstream band entering the fit."""
LEGACY_MAX_TORR = 0.95
"""Upper edge of the downstream band entering the fit."""
LEGACY_TAIL_FRACTION = 0.25
"""The fit uses this final fraction of the in-band samples."""
LEGACY_MEAN_FRACTION = 0.75
"""Upstream pressure and temperature are averaged over this final fraction
of the recording."""


@dataclass(frozen=True)
class TailAsymptoteFit:
    """Straight line ``P = slope·t + intercept`` through the end of the rise."""

    slope_torr_per_s: float
    intercept_torr: float
    used: np.ndarray
    """Boolean mask of the samples that entered the fit."""

    @property
    def time_lag_s(self) -> float:
        """Where the line crosses P = 0, on the fit's time axis."""
        return -self.intercept_torr / self.slope_torr_per_s

    def evaluate(self, time_s: npt.ArrayLike) -> np.ndarray:
        """The fitted line at the given times, in Torr."""
        return self.slope_torr_per_s * np.asarray(time_s, dtype=float) + (
            self.intercept_torr
        )


def fit_tail_asymptote(
    time_s: npt.ArrayLike,
    downstream_torr: npt.ArrayLike,
    min_torr: float = LEGACY_MIN_TORR,
    max_torr: float = LEGACY_MAX_TORR,
    tail_fraction: float = LEGACY_TAIL_FRACTION,
) -> TailAsymptoteFit:
    """Unweighted linear fit to the last ``tail_fraction`` of the in-band
    downstream samples.

    Args:
        time_s: Time axis in seconds (the legacy method uses time since the
            first sample).
        downstream_torr: Downstream pressure in Torr.
        min_torr: Lower edge of the band.
        max_torr: Upper edge of the band.
        tail_fraction: Final fraction of the in-band samples that is fitted.
    """
    time_arr = np.asarray(time_s, dtype=float)
    pressure = np.asarray(downstream_torr, dtype=float)
    in_band = np.nonzero((pressure >= min_torr) & (pressure <= max_torr))[0]
    tail = in_band[int(len(in_band) * (1 - tail_fraction)) :]
    slope, intercept = np.polyfit(time_arr[tail], pressure[tail], 1)
    used = np.zeros(len(pressure), dtype=bool)
    used[tail] = True
    return TailAsymptoteFit(
        slope_torr_per_s=float(slope), intercept_torr=float(intercept), used=used
    )


def tail_mean(values: npt.ArrayLike, fraction: float = LEGACY_MEAN_FRACTION) -> float:
    """Mean over the final ``fraction`` of the samples (NaNs ignored)."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr[len(arr) - int(fraction * len(arr)) :]))
