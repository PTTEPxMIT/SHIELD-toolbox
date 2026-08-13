"""Cross-run Arrhenius plot for a processed campaign.

Same contract as the run plots: accepts/returns an ``Axes``, never calls
``plt.show()`` or ``savefig``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from shield_toolbox.analysis import ArrheniusFit
from shield_toolbox.campaign import ARRHENIUS_QUANTITIES

_AXIS_LABELS = {
    "permeability": "Permeability (H/(m·s·Pa$^{0.5}$))",
    "diffusivity": "Diffusivity (m$^2$/s)",
    "solubility": "Solubility (H/(m$^3$·Pa$^{0.5}$))",
}
_MARKERS = "osD^v<>P*"


def plot_arrhenius(
    results: pd.DataFrame,
    fit: ArrheniusFit | None = None,
    quantity: str = "permeability",
    ax: Axes | None = None,
) -> Axes:
    """Arrhenius plot: one extracted property vs 1000/T, log scale.

    Runs are grouped into one marker series per (substrate, coating); error
    bars are drawn where the table carries uncertainties. Rows without a
    finite value of the quantity (e.g. no valid time lag) are skipped. A
    secondary top axis shows the temperature in °C.

    Args:
        results: Table from :func:`shield_toolbox.campaign.load_results`.
        fit: Optional fit from :func:`shield_toolbox.campaign.arrhenius`,
            drawn as a dashed line labeled with the activation energy.
        quantity: ``"permeability"``, ``"diffusivity"``, or ``"solubility"``.
        ax: Axes to draw on; created if not given.
    """
    if quantity not in ARRHENIUS_QUANTITIES:
        raise ValueError(
            f"Unknown quantity {quantity!r}; expected one of {ARRHENIUS_QUANTITIES}"
        )
    if ax is None:
        _, ax = plt.subplots()
    column = "diffusivity_m2_per_s" if quantity == "diffusivity" else quantity
    err_column = f"{quantity}_err"

    subset = results[results[column].notna() & results["temperature_K"].notna()]
    groups = subset.groupby(["substrate", "coating"], dropna=False, sort=True)
    for index, ((substrate, coating), group) in enumerate(groups):
        inv_t = 1000.0 / group["temperature_K"].to_numpy(dtype=float)
        values = group[column].to_numpy(dtype=float)
        errs = None
        if err_column in group.columns and group[err_column].notna().all():
            errs = group[err_column].to_numpy(dtype=float)
        ax.errorbar(
            inv_t,
            values,
            yerr=errs,
            fmt=_MARKERS[index % len(_MARKERS)],
            ms=6,
            capsize=3,
            ls="none",
            label=f"{substrate} / {coating}",
        )

    if fit is not None:
        ea_kj = fit.activation_energy_J_per_mol / 1000.0
        ax.plot(
            fit.fit_x_inverse_kK,
            fit.fit_y,
            color="tab:red",
            ls="--",
            lw=1.5,
            label=f"fit: $E_a$ = {ea_kj:.1f} kJ/mol",
        )

    ax.set_yscale("log")
    ax.set_xlabel("1000/T (1/K)")
    ax.set_ylabel(_AXIS_LABELS[quantity])
    ax.legend()

    celsius_axis = ax.secondary_xaxis(
        "top",
        functions=(
            lambda inv: 1000.0 / np.maximum(inv, 1e-9) - 273.15,
            lambda c: 1000.0 / np.maximum(c + 273.15, 1e-9),
        ),
    )
    celsius_axis.set_xlabel("Temperature (°C)")
    return ax
