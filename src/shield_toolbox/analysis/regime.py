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

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from uncertainties import UFloat, ufloat

DIFFUSION_LIMIT_EXPONENT = 0.63
"""Exponents at or below this are conventionally treated as
diffusion-limited (n = 0.5 plus a commonly assumed margin)."""

SURFACE_LIMIT_EXPONENT = 0.87
"""Exponents at or above this are treated as surface-limited (n = 1 minus
the same margin)."""


def classify_exponent(
    exponent: UFloat | float,
    diffusion_limit: float = DIFFUSION_LIMIT_EXPONENT,
    surface_limit: float = SURFACE_LIMIT_EXPONENT,
) -> str:
    """Regime label for a pressure exponent: ``"diffusion"`` (n ≤ 0.63),
    ``"surface"`` (n ≥ 0.87), or ``"mixed"`` in between.

    The margins around the ideal n = 0.5 and n = 1 are the commonly assumed
    ones — n ≤ 0.63 is routinely taken as effectively diffusion-limited, and
    the surface bound mirrors it symmetrically.
    """
    nominal = getattr(exponent, "nominal_value", exponent)
    if nominal <= diffusion_limit:
        return "diffusion"
    if nominal >= surface_limit:
        return "surface"
    return "mixed"


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


@dataclass(frozen=True)
class PowerLawFit:
    """One fitted power law J = prefactor · P^exponent over a pressure range."""

    exponent: UFloat
    prefactor: UFloat
    """In the units the data came in: flux = prefactor · pressure^exponent."""
    regime: str
    """:func:`classify_exponent` of the exponent."""

    def evaluate(self, pressure: npt.ArrayLike) -> np.ndarray:
        """The fitted flux at ``pressure`` (nominal values), for plotting."""
        pressure_arr = np.asarray(pressure, dtype=float)
        return self.prefactor.nominal_value * pressure_arr**self.exponent.nominal_value


@dataclass(frozen=True)
class RegimeFit:
    """Result of :func:`fit_regime_transition`."""

    fits: tuple[PowerLawFit, ...]
    """One power law if a single regime describes the sweep, two (below
    then above the transition) if a regime change is detected."""
    transition: float | None
    """Crossover pressure in the input pressure units — where the two
    power laws intersect — or None when a single law fits."""


def fit_regime_transition(pressure: npt.ArrayLike, flux: npt.ArrayLike) -> RegimeFit:
    """Fit a pressure sweep and locate the regime transition, if there is one.

    Fits a single power law, then every two-segment split (each side keeps
    at least three points, sorted by pressure) and takes the best split by
    least squares. The split is only accepted over the single law when it
    is statistically justified: a better Bayesian information criterion
    AND exponents differing by more than twice their combined standard
    error. Sweeps with fewer than six points can never split.

    As with :func:`fit_pressure_exponent`, the data must come from one
    sample at one temperature; any consistent units work, and the
    transition is reported in the input pressure units.

    Args:
        pressure: Upstream pressures, all positive.
        flux: Steady-state permeation fluxes at those pressures, all
            positive, same length.

    Raises:
        ValueError: For fewer than three points or any non-positive
            pressure or flux.
    """
    pressure_arr = np.asarray(pressure, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    if len(pressure_arr) < 3:
        raise ValueError(
            f"Need at least three (pressure, flux) points; got {len(pressure_arr)}"
        )
    if (pressure_arr <= 0).any() or (flux_arr <= 0).any():
        raise ValueError("Pressures and fluxes must all be positive")

    order = np.argsort(pressure_arr)
    log_p = np.log10(pressure_arr[order])
    log_j = np.log10(flux_arr[order])
    n_points = len(log_p)

    single, sse_single = _fit_power_law(log_p, log_j)
    if n_points < 6:
        return RegimeFit(fits=(single,), transition=None)

    best_sse = np.inf
    best: tuple[int, PowerLawFit, PowerLawFit] | None = None
    for split in range(3, n_points - 2):
        low, sse_low = _fit_power_law(log_p[:split], log_j[:split])
        high, sse_high = _fit_power_law(log_p[split:], log_j[split:])
        if sse_low + sse_high < best_sse:
            best_sse = sse_low + sse_high
            best = (split, low, high)

    split, low, high = best
    # BIC with k = 2 for one line, k = 5 for two lines plus the breakpoint.
    with np.errstate(divide="ignore"):
        bic_single = n_points * np.log(sse_single / n_points) + 2 * np.log(n_points)
        bic_split = n_points * np.log(best_sse / n_points) + 5 * np.log(n_points)
    exponent_gap = abs(low.exponent.nominal_value - high.exponent.nominal_value)
    distinguishable = exponent_gap > 2 * np.hypot(low.exponent.s, high.exponent.s)
    if not (bic_split < bic_single and distinguishable):
        return RegimeFit(fits=(single,), transition=None)

    # Crossover where the two laws intersect; if the intersection falls
    # outside the gap between the flanking data points (possible with noisy
    # near-parallel segments), fall back to that gap's geometric mean.
    transition = (high.prefactor.nominal_value / low.prefactor.nominal_value) ** (
        1.0 / (low.exponent.nominal_value - high.exponent.nominal_value)
    )
    gap = (10.0 ** log_p[split - 1], 10.0 ** log_p[split])
    if not gap[0] <= transition <= gap[1]:
        transition = float(np.sqrt(gap[0] * gap[1]))
    return RegimeFit(fits=(low, high), transition=float(transition))


def _fit_power_law(log_p: np.ndarray, log_j: np.ndarray) -> tuple[PowerLawFit, float]:
    """Weighted-free linear fit in log space → (PowerLawFit, sum sq. residuals)."""
    coeffs, cov = np.polyfit(log_p, log_j, 1, cov=True)
    residuals = log_j - np.polyval(coeffs, log_p)
    exponent = ufloat(coeffs[0], np.sqrt(cov[0, 0]))
    prefactor = 10.0 ** ufloat(coeffs[1], np.sqrt(cov[1, 1]))
    fit = PowerLawFit(
        exponent=exponent, prefactor=prefactor, regime=classify_exponent(exponent)
    )
    return fit, float(np.sum(residuals**2))


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
