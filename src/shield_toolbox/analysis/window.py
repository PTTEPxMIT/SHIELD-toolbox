"""Run-window detection from upstream gauge saturation.

The upstream Baratron 626D (1000 Torr full scale) saturates at a raw reading
of ~10.12 V. While the upstream side is filled above full scale the reading
sits at the cap and is not a real pressure. The analysable part of the run is
everything after the reading comes off the cap (drops below 10 V), through to
the final recorded time. Runs that never saturate are analysable end-to-end.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

SATURATION_THRESHOLD_V = 10.0
"""Raw upstream voltage above which the gauge is considered saturated (the
hard cap is ~10.12 V; 10 V leaves margin for noise)."""


def run_window_mask(
    upstream_voltage_v: npt.ArrayLike,
    saturation_threshold_v: float = SATURATION_THRESHOLD_V,
) -> np.ndarray:
    """Boolean mask of the samples that belong to the analysable run.

    Args:
        upstream_voltage_v: Raw upstream gauge voltage trace in volts.
        saturation_threshold_v: Readings at or above this are treated as
            saturated.

    Returns:
        Boolean array, True from the first sample after the last saturated
        reading through the end of the trace. All True if the gauge never
        saturated.
    """
    voltage = np.asarray(upstream_voltage_v, dtype=float)
    saturated = voltage >= saturation_threshold_v
    mask = np.zeros(voltage.shape, dtype=bool)
    if not saturated.any():
        mask[:] = True
        return mask
    last_saturated = int(np.nonzero(saturated)[0][-1])
    mask[last_saturated + 1 :] = True
    return mask
