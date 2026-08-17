"""Permeation-regime analysis: pressure-exponent fit and surface-limited
property extraction.

Steady-state permeation through a metal membrane is bottlenecked either by
bulk diffusion or by the surface reactions (dissociative adsorption upstream,
recombinative desorption downstream), and the two regimes scale differently
with upstream pressure::

    diffusion-limited:  J = Φ·√P / e     (flux ∝ P^0.5, Sieverts scaling)
    surface-limited:    J = K_d·P / 2    (flux ∝ P^1)

Repeating runs at one temperature over a range of upstream pressures and
fitting the log-log slope of flux vs pressure identifies the regime
(:func:`fit_pressure_exponent`). The standard Φ / D / S extraction
(``process_run``) is physically meaningful only for diffusion-limited runs;
surface-limited runs instead yield the effective dissociation coefficient
(:func:`dissociation_coeff`) and, via detailed balance, the recombination
coefficient (:func:`recombination_coeff`).

Pure physics only: no file I/O, no plotting.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from uncertainties import UFloat, ufloat


def fit_pressure_exponent(pressure: npt.ArrayLike, flux: npt.ArrayLike) -> UFloat:
    """Pressure exponent n in J ∝ Pⁿ, from a log-log linear fit.

    n ≈ 0.5 means diffusion-limited transport, n ≈ 1 surface-limited;
    values in between indicate mixed transport, where neither limiting
    extraction is valid. The slope is unit-invariant, so any consistent
    pressure and flux units work — but the runs must share one sample and
    one temperature, or the exponent is meaningless.

    Args:
        pressure: Upstream pressures, all positive.
        flux: Steady-state permeation fluxes at those pressures, all
            positive, same length.

    Returns:
        The exponent with its standard error from the fit covariance.

    Raises:
        ValueError: For fewer than three points (no error estimate) or any
            non-positive pressure or flux.
    """
    pressure_arr = np.asarray(pressure, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    if len(pressure_arr) < 3:
        raise ValueError(
            f"Need at least three (pressure, flux) points to fit an "
            f"exponent with an error estimate; got {len(pressure_arr)}"
        )
    if (pressure_arr <= 0).any() or (flux_arr <= 0).any():
        raise ValueError("Pressures and fluxes must all be positive")

    coeffs, cov = np.polyfit(np.log10(pressure_arr), np.log10(flux_arr), 1, cov=True)
    return ufloat(coeffs[0], np.sqrt(cov[0, 0]))


def dissociation_coeff(flux: UFloat | float, pressure_pa: float) -> UFloat | float:
    """Effective dissociation coefficient K_d = 2·J/P, in H/(m²·s·Pa).

    Valid only in the surface-limited regime — verify flux ∝ P first with
    :func:`fit_pressure_exponent`. Assumes identical upstream and downstream
    surfaces and near-vacuum downstream, so half the dissociated atoms
    permeate (J = K_d·P/2). The result is an *effective* coefficient for
    this sample's surface condition: asymmetric surface states shift it
    within a factor of ~2, and any residual diffusion resistance makes it
    an underestimate.

    Args:
        flux: Steady-state atomic flux J in H/(m²·s) (uncertainty
            propagates through if a ufloat).
        pressure_pa: Upstream pressure in Pa; must be positive.

    Raises:
        ValueError: If ``pressure_pa`` is not positive.
    """
    if not pressure_pa > 0:
        raise ValueError(f"Pressure must be positive, got {pressure_pa:.3g} Pa")
    return 2.0 * flux / pressure_pa


def recombination_coeff(
    dissociation: UFloat | float, solubility: UFloat | float
) -> UFloat | float:
    """Recombination coefficient via detailed balance, K_r = K_d/K_s²,
    in m⁴/(H·s).

    Detailed balance (K_d = K_r·K_s²) preserves Sieverts' equilibrium: a
    modified surface changes the kinetics, not the bulk thermodynamics.
    The solubility cannot come from surface-limited data — take it from
    diffusion-limited runs (S = Φ/D) or the literature. It enters squared,
    so it usually dominates the uncertainty. Values follow the atomic-flux
    convention (J = K_d·P); check factor-of-2 conventions before comparing
    with published coefficients.

    Args:
        dissociation: K_d in H/(m²·s·Pa), from :func:`dissociation_coeff`.
        solubility: Sieverts solubility K_s in H/(m³·Pa^0.5); must be
            positive (uncertainty propagates through if a ufloat).

    Raises:
        ValueError: If ``solubility`` is not positive.
    """
    nominal = getattr(solubility, "nominal_value", solubility)
    if not nominal > 0:
        raise ValueError(f"Solubility must be positive, got {nominal:.3g}")
    return dissociation / solubility**2
