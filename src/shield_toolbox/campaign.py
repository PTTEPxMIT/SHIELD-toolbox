"""Campaign-level analysis: aggregate processed runs and fit Arrhenius trends.

``process_run(...).write(base_dir)`` stores each run as
``<base_dir>/<substrate>/<coating>/<run_id>/result.json``. This module walks
that tree back into one table (:func:`load_results`) and fits the temperature
dependence of any extracted property (:func:`arrhenius`)::

    results = load_results("processed_runs", substrate="316L steel")
    fit = arrhenius(results)                      # permeability vs 1/T
    fit.activation_energy_J_per_mol, fit.pre_exponential

Plot with :func:`shield_toolbox.plotting.plot_arrhenius`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from uncertainties import ufloat

from shield_toolbox.analysis import ArrheniusFit, fit_arrhenius
from shield_toolbox.processing import RESULT_FILENAME

ARRHENIUS_QUANTITIES = ("permeability", "diffusivity", "solubility")


def load_results(
    base_dir: str | Path,
    substrate: str | None = None,
    coating: str | None = None,
) -> pd.DataFrame:
    """Collect stored ``result.json`` files into one row-per-run table.

    Args:
        base_dir: Base directory that :meth:`ProcessedRun.write` wrote into.
        substrate: Keep only runs of this substrate (exact match).
        coating: Keep only runs with this coating (exact match).

    Returns:
        One row per processed run, sorted by temperature: ``run_id``,
        ``substrate``, ``coating``, ``thickness_m``, ``temperature_K``,
        ``temperature_source``, ``upstream_torr``, ``permeability`` /
        ``permeability_err``, ``time_lag_s``, ``diffusivity_m2_per_s``,
        ``solubility`` / ``solubility_err`` (NaN where a run has no valid
        time lag), ``furnace_setpoint``, and ``result_path``.

    Raises:
        FileNotFoundError: If no ``result.json`` is found under ``base_dir``
            (after filtering).
    """
    rows = []
    for result_path in sorted(Path(base_dir).rglob(RESULT_FILENAME)):
        with open(result_path) as f:
            result = json.load(f)
        sample = result.get("sample", {})
        if substrate is not None and sample.get("substrate") != substrate:
            continue
        if coating is not None and sample.get("coating") != coating:
            continue
        results = result.get("results", {})
        rows.append(
            {
                "run_id": result.get("run_id"),
                "substrate": sample.get("substrate"),
                "coating": sample.get("coating"),
                "thickness_m": sample.get("thickness_m"),
                "temperature_K": result.get("temperature", {}).get(
                    "sample_temperature_K"
                ),
                "temperature_source": result.get("temperature", {}).get("source"),
                "upstream_torr": results.get("upstream_pressure_torr"),
                "permeability": results.get("permeability", {}).get("nominal"),
                "permeability_err": results.get("permeability", {}).get("std_dev"),
                "time_lag_s": results.get("time_lag", {}).get("time_lag_s"),
                "diffusivity_m2_per_s": results.get("diffusivity", {}).get("value"),
                "solubility": results.get("solubility", {}).get("nominal"),
                "solubility_err": results.get("solubility", {}).get("std_dev"),
                "furnace_setpoint": result.get("run_info", {}).get("furnace_setpoint"),
                "result_path": str(result_path),
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No {RESULT_FILENAME} under {base_dir}"
            + (f" for substrate={substrate!r}" if substrate else "")
            + (f" coating={coating!r}" if coating else "")
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("temperature_K", ignore_index=True)


def arrhenius(results: pd.DataFrame, quantity: str = "permeability") -> ArrheniusFit:
    """Weighted Arrhenius fit of one extracted property across a results table.

    Args:
        results: Table from :func:`load_results`. Mixing substrates or
            coatings in one fit is physically meaningless — filter first.
        quantity: ``"permeability"``, ``"diffusivity"``, or ``"solubility"``.

    Returns:
        The fit; ``activation_energy_J_per_mol`` and ``pre_exponential``
        give the Arrhenius parameters.

    Raises:
        ValueError: For an unknown quantity, or fewer than two runs with a
            finite value of it (rows lacking a valid time lag are dropped
            for diffusivity/solubility).
    """
    if quantity not in ARRHENIUS_QUANTITIES:
        raise ValueError(
            f"Unknown quantity {quantity!r}; expected one of {ARRHENIUS_QUANTITIES}"
        )
    column = "diffusivity_m2_per_s" if quantity == "diffusivity" else quantity
    err_column = f"{quantity}_err"

    subset = results[results[column].notna() & results["temperature_K"].notna()]
    if len(subset) < 2:
        raise ValueError(
            f"Need at least two runs with a finite {quantity}; "
            f"got {len(subset)} of {len(results)}"
        )

    values: list = subset[column].tolist()
    if err_column in subset.columns:
        errs = subset[err_column]
        if errs.notna().all() and (errs > 0).all():
            values = [ufloat(v, e) for v, e in zip(values, errs)]
    return fit_arrhenius(subset["temperature_K"].to_numpy(), values)
