"""Furnace-controller log plots.

Same contract as the other plot modules: accepts/returns an ``Axes``, never
calls ``plt.show()`` or ``savefig``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from shield_toolbox.analysis import furnace_temperature_offset


def plot_furnace_log(
    furnace: pd.DataFrame,
    sample_time_s: np.ndarray | None = None,
    sample_temperature_c: np.ndarray | None = None,
    annotate_offset: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Furnace measured temperature and working setpoint vs time.

    Optionally overlays a simultaneous sample-thermocouple trace (on the
    same clock as the furnace log — trim/align first) and annotates the
    steady-state sample-vs-furnace offset ΔT from
    :func:`~shield_toolbox.analysis.furnace_temperature_offset`.

    Args:
        furnace: Frame from :func:`shield_toolbox.io.load_furnace_log`.
        sample_time_s: Sample-thermocouple time axis, seconds on the furnace
            log's clock.
        sample_temperature_c: Sample-thermocouple temperature in °C.
        annotate_offset: Annotate ΔT when a sample trace is given.
        ax: Axes to draw on; created if not given.
    """
    if ax is None:
        _, ax = plt.subplots()
    minutes = furnace["time_s"].to_numpy() / 60.0
    ax.plot(
        minutes,
        furnace["furnace_temperature_C"],
        color="tab:blue",
        lw=1,
        label="furnace",
    )
    ax.plot(
        minutes,
        furnace["setpoint_C"],
        color="tab:blue",
        ls="--",
        lw=1,
        label="setpoint",
    )

    if sample_time_s is not None and sample_temperature_c is not None:
        ax.plot(
            np.asarray(sample_time_s, dtype=float) / 60.0,
            sample_temperature_c,
            color="tab:red",
            lw=1,
            label="sample TC",
        )
        if annotate_offset:
            offset = furnace_temperature_offset(
                sample_temperature_c,
                float(furnace["furnace_temperature_C"].iloc[-1]),
            )
            ax.annotate(
                f"$\\Delta T$ = {offset:+.1f} K",
                xy=(0.98, 0.06),
                xycoords="axes fraction",
                ha="right",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"},
            )

    ax.set_xlabel("Time (minutes since log start)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    return ax
