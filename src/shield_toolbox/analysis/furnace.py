"""Sample-vs-furnace temperature offset.

The sample thermocouple sits inside the sample holder, the furnace
controller's sensor at the heating element — at steady state the sample runs
cooler. This module quantifies that offset from a furnace log and a
simultaneous sample-thermocouple trace; it is the empirical source of the
``furnace_setpoint_offset_K`` fallback in :class:`~shield_toolbox.config.RigConfig`
(−18 K on rig v1), used for old runs recorded without a sample thermocouple.

Ported from the legacy ``PlotFurnaceData.ipynb`` ΔT calculation, with the
sign made explicit: negative means the sample is cooler than the furnace.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def furnace_temperature_offset(
    sample_temperature_c: npt.ArrayLike,
    furnace_final_temperature_c: float,
    settled_fraction: float = 0.75,
) -> float:
    """Sample-minus-furnace temperature offset at steady state, in K.

    The sample temperature is averaged over the last ``settled_fraction`` of
    its trace (skipping the heat-up transient at the start, as in the legacy
    analysis) and compared to the furnace controller's final measured
    temperature.

    Args:
        sample_temperature_c: Sample thermocouple trace in °C, covering the
            settling to steady state.
        furnace_final_temperature_c: The furnace controller's settled
            process value in °C — e.g. ``furnace_temperature_C.iloc[-1]`` of
            a :func:`~shield_toolbox.io.load_furnace_log` frame.
        settled_fraction: Trailing fraction of the sample trace to average.

    Returns:
        Signed offset (sample − furnace), K: negative when the sample runs
        cooler than the furnace, the normal case.

    Raises:
        ValueError: If the sample trace is empty or ``settled_fraction`` is
            not in (0, 1].
    """
    if not 0 < settled_fraction <= 1:
        raise ValueError(f"settled_fraction must be in (0, 1], got {settled_fraction}")
    temperature = np.asarray(sample_temperature_c, dtype=float)
    if temperature.size == 0:
        raise ValueError("Sample temperature trace is empty")
    settled = temperature[int(len(temperature) * (1 - settled_fraction)) :]
    return float(np.mean(settled) - furnace_final_temperature_c)
