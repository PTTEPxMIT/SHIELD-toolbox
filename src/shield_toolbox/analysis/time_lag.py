"""Background-subtracted time-lag method.

The classic permeation time-lag method (Daynes/Barrer), applied after the
downstream background has been removed. The procedure, in four steps:

1. **Noise recording.** Before hydrogen has crossed the sample, the sealed
   downstream volume sees only the background: seal leakage plus outgassing.
   That flat stretch after the upstream step is each run's noise recording.
   It starts ``NOISE_START_S`` after ``t_init`` (skipping the small jump the
   valve opening puts on the downstream gauge) and ends a fraction
   ``NOISE_MARGIN_FRACTION`` of the pre-rise time (``t_init`` to the detected
   onset of the downstream rise, :func:`rise_onset`) before the onset.
2. **Initial time.** ``t_init`` is the moment the upstream pressure steps up:
   the first sample above half its plateau (:func:`initial_time`).
3. **Background fit.** A straight line ``P_bg = a + b·(t − t_init)`` is fitted
   to the noise recording (:func:`fit_background`).
4. **Time lag on filtered data.** Subtracting the background line gives the
   filtered signal, which is 0 at ``t_init``. The steady-state line
   ``S∞·(t − t_init − τ_L)`` is fitted from ``start_taus·τ_L`` to the end of
   usable data, iterating τ_L and the window start until they agree
   (:func:`fit_steady_state`). By 3 τ_L the flux is within 1.5 % of steady
   state.

For a membrane of thickness e with Fickian diffusion::

    τ_L = e² / (6·D)      →      D = e² / (6·τ_L)        [m²/s]
    S = Φ / D                                            [H/(m³·Pa^0.5)]

Pressures are in Pa and times in seconds since ``t_init`` unless stated.
Pure physics only: no file I/O, no plotting. ``process_run`` wires these into
the standard pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from uncertainties import UFloat, ufloat

PRESSURISED_TORR = 10.0
"""Upstream readings above this count as pressurised (plateau and dropout
detection)."""
UPSTREAM_ZERO_WINDOW_S = 60.0
"""The upstream pre-start bias is the median over this long before the step."""
NOISE_START_S = 60.0
"""The noise recording starts this long after ``t_init``, skipping the jump
the valve opening puts on the downstream gauge."""
NOISE_MARGIN_FRACTION = 0.25
"""The noise recording ends this fraction of the pre-rise time (``t_init``
to the detected onset) before the onset, i.e. at (1 − fraction)·onset."""
ONSET_SIGMA = 5.0
"""A block counts as the rise onset when its mean sits more than this many
standard errors above the background line fitted before it."""
ONSET_BLOCK_S = 600.0
"""Block length of the onset search."""
FINE_ONSET_BLOCK_S = 60.0
"""Block length of the second onset search, for rises that start within
minutes (high temperature)."""
SS_START_TAUS = 3.0
"""The steady-state window starts at this many time lags after ``t_init``."""


def initial_time(time_s: npt.ArrayLike, upstream_torr: npt.ArrayLike) -> float:
    """``t_init``: the first time the upstream passes half its plateau.

    The plateau is the median of the pressurised samples (above
    ``PRESSURISED_TORR``).

    Args:
        time_s: Time axis in seconds.
        upstream_torr: Upstream pressure in Torr.

    Raises:
        ValueError: If the upstream is never pressurised.
    """
    time_arr = np.asarray(time_s, dtype=float)
    upstream = np.asarray(upstream_torr, dtype=float)
    pressurised = upstream > PRESSURISED_TORR
    if not pressurised.any():
        raise ValueError(
            f"Upstream never rises above {PRESSURISED_TORR} Torr — no pressure step"
        )
    plateau = np.median(upstream[pressurised])
    return float(time_arr[np.nonzero(upstream > 0.5 * plateau)[0][0]])


def upstream_zero(
    time_since_init_s: npt.ArrayLike,
    upstream_torr: npt.ArrayLike,
    window_s: float = UPSTREAM_ZERO_WINDOW_S,
) -> float | None:
    """Pre-start bias of the upstream gauge: its reading before the step.

    The median of the samples in the last ``window_s`` before ``t_init``
    that are still below ``PRESSURISED_TORR`` (so the step itself, and any
    re-zeroing earlier in the recording, are left out). Subtracted from the
    measured upstream pressure, as the background level is from the
    downstream.

    Returns:
        The bias in Torr, or None when the recording has no samples before
        the step.
    """
    t_rel = np.asarray(time_since_init_s, dtype=float)
    upstream = np.asarray(upstream_torr, dtype=float)
    before = (t_rel < 0) & (upstream < PRESSURISED_TORR)
    if not before.any():
        return None
    last = before & (t_rel >= t_rel[before][-1] - window_s)
    return float(np.median(upstream[last]))


def rise_onset(
    time_since_init_s: npt.ArrayLike,
    pressure: npt.ArrayLike,
    start_s: float = NOISE_START_S,
    block_s: float = ONSET_BLOCK_S,
    n_sigma: float = ONSET_SIGMA,
    max_s: float | None = None,
) -> tuple[float | None, float]:
    """Onset of the downstream rise, and the noise level before it.

    Starting at ``start_s``, the downstream is averaged in ``block_s`` blocks.
    A straight line is fitted to all the data before each block and extended
    across it. The onset is the start of the first block whose mean sits more
    than ``n_sigma`` standard errors above that line. The first three blocks
    (at most 30 min) only seed the background line.

    Args:
        time_since_init_s: Time since ``t_init``, s.
        pressure: Downstream pressure (any unit; Pa in ``process_run``).
        start_s: Start of the search, s after ``t_init``.
        block_s: Block length, s.
        n_sigma: Detection threshold in standard errors of the block mean.
        max_s: End of the search, s after ``t_init``; default the end of
            the data.

    Returns:
        ``(onset_s, noise_sd)``: the onset in s after ``t_init`` (None if no
        block departs from the background), and the standard deviation of
        the seed stretch about its linear fit (same unit as ``pressure``).
    """
    t_rel = np.asarray(time_since_init_s, dtype=float)
    p = np.asarray(pressure, dtype=float)
    first = (t_rel >= start_s) & (t_rel < start_s + min(1800, 3 * block_s))
    noise_sd = np.std(
        p[first] - np.polyval(np.polyfit(t_rel[first], p[first], 1), t_rel[first])
    )
    max_s = t_rel[-1] if max_s is None else max_s
    for lo in np.arange(start_s, max_s, block_s)[3:]:
        before = (t_rel >= start_s) & (t_rel < lo)
        block = (t_rel >= lo) & (t_rel < lo + block_s)
        if not block.any():
            break
        b, a = np.polyfit(t_rel[before], p[before], 1)
        resid = p[block].mean() - (a + b * t_rel[block].mean())
        if resid > n_sigma * noise_sd / np.sqrt(block.sum()):
            return float(lo), float(noise_sd)
    return None, float(noise_sd)


@dataclass(frozen=True)
class NoiseRecording:
    """The stretch of downstream signal used as the background (step 1)."""

    start_s: float
    """Start, s after ``t_init``."""
    end_s: float
    """End, s after ``t_init``."""
    onset_s: float | None
    """Detected onset of the downstream rise, s after ``t_init`` (None if
    none was detected)."""
    onset_from: str
    """``"10 min blocks"``, ``"1 min blocks"``, ``"none detected"`` or
    ``"manual"``."""
    noise_sd: float
    """Noise level before the onset (unit of the pressure searched)."""
    used: np.ndarray
    """Boolean mask of the samples inside the noise recording."""


def find_noise_recording(
    time_since_init_s: npt.ArrayLike,
    pressure: npt.ArrayLike,
    start_s: float = NOISE_START_S,
    margin_fraction: float = NOISE_MARGIN_FRACTION,
    n_sigma: float = ONSET_SIGMA,
    end_s: float | None = None,
) -> NoiseRecording:
    """Find the noise recording: from ``start_s`` to
    ``(1 − margin_fraction)·onset``.

    The early permeation flux builds gradually, so the recording stops a
    fraction of the pre-rise time before the detected onset. The onset comes
    from 10 min blocks (:func:`rise_onset`). At high temperature the rise can
    start within minutes, before the 10 min blocks can resolve it, so the
    search is repeated with 1 min blocks; if that finds the rise inside the
    10-min-based noise window, the 1 min onset is used instead. If no onset
    is detected at all, the noise recording is the 30 min seed stretch the
    onset search starts from.

    Args:
        time_since_init_s: Time since ``t_init``, s.
        pressure: Downstream pressure (Pa in ``process_run``).
        start_s: Start of the noise recording, s after ``t_init``.
        margin_fraction: Fraction of the pre-rise time (``t_init`` to the
            onset) left between the end of the noise recording and the
            onset.
        n_sigma: Onset detection threshold (standard errors).
        end_s: Manual end of the noise recording, s after ``t_init``;
            overrides the detected one (the onset is still reported).
    """
    t_rel = np.asarray(time_since_init_s, dtype=float)
    p = np.asarray(pressure, dtype=float)
    onset_s, noise_sd = rise_onset(t_rel, p, start_s=start_s, n_sigma=n_sigma)
    onset_from = "10 min blocks"
    fine_onset_s, fine_sd = rise_onset(
        t_rel, p, start_s=start_s, block_s=FINE_ONSET_BLOCK_S, n_sigma=n_sigma
    )
    if fine_onset_s is not None and (
        onset_s is None or fine_onset_s < (1 - margin_fraction) * onset_s
    ):
        # rise already under way inside the 10-min-based window
        onset_s, noise_sd, onset_from = fine_onset_s, fine_sd, "1 min blocks"
    if end_s is not None:
        onset_from = "manual"
    elif onset_s is None:
        onset_from = "none detected"
        end_s = start_s + 1800.0
    else:
        end_s = (1 - margin_fraction) * onset_s
    return NoiseRecording(
        start_s=float(start_s),
        end_s=float(end_s),
        onset_s=onset_s,
        onset_from=onset_from,
        noise_sd=float(noise_sd),
        used=(t_rel >= start_s) & (t_rel <= end_s),
    )


@dataclass(frozen=True)
class BackgroundFit:
    """Background line ``P_bg = a + b·(t − t_init)`` (step 3)."""

    level: UFloat
    """``a``: the background extended back to ``t_init``."""
    slope: UFloat
    """``b``: the background rate (pressure unit per s)."""

    def evaluate(self, time_since_init_s: npt.ArrayLike) -> np.ndarray:
        """Nominal background at the given times since ``t_init``."""
        t_rel = np.asarray(time_since_init_s, dtype=float)
        return self.level.nominal_value + self.slope.nominal_value * t_rel


def fit_background(
    time_since_init_s: npt.ArrayLike,
    pressure: npt.ArrayLike,
    used: npt.ArrayLike,
) -> BackgroundFit:
    """Straight-line fit to the noise recording.

    Args:
        time_since_init_s: Time since ``t_init``, s.
        pressure: Downstream pressure (Pa in ``process_run``).
        used: Boolean mask of the noise recording.
    """
    mask = np.asarray(used, dtype=bool)
    x = np.asarray(time_since_init_s, dtype=float)[mask]
    y = np.asarray(pressure, dtype=float)[mask]
    (b, a), cov = np.polyfit(x, y, 1, cov=True)
    return BackgroundFit(
        level=ufloat(a, np.sqrt(cov[1, 1])), slope=ufloat(b, np.sqrt(cov[0, 0]))
    )


@dataclass(frozen=True)
class SteadyStateFit:
    """Steady-state line ``S∞·(t − t_init − τ_L)`` on the filtered signal."""

    slope: float
    """``S∞``, filtered pressure unit per s."""
    time_lag_s: float
    """``τ_L``: the line's zero crossing, s after ``t_init``."""
    used: np.ndarray
    """Boolean mask of the samples inside the steady-state window."""
    covariance: np.ndarray
    """2×2 covariance of (slope, intercept) from the fit."""
    converged: bool = True
    """False if the τ_L iteration stopped before τ_L settled (it ran out of
    iterations, or the next window start lay past the data)."""

    def evaluate(self, time_since_init_s: npt.ArrayLike) -> np.ndarray:
        """The steady-state line at the given times since ``t_init``."""
        t_rel = np.asarray(time_since_init_s, dtype=float)
        return self.slope * (t_rel - self.time_lag_s)


def fit_steady_state(
    time_since_init_s: npt.ArrayLike,
    filtered: npt.ArrayLike,
    usable: npt.ArrayLike,
    start_taus: float = SS_START_TAUS,
    end_s: float | None = None,
    n_iter: int = 20,
    start_s: float | None = None,
) -> SteadyStateFit:
    """Fit the steady-state line from ``start_taus·τ_L`` to ``end_s``.

    τ_L and the window start depend on each other, so they are iterated from
    a first guess of a quarter of the span until τ_L moves by less than 1 s
    (at most ``n_iter`` times; τ_L is floored at 60 s between iterations).
    If the next window start would leave fewer than four usable samples, the
    iteration stops and keeps the last fit (``converged=False``); if even the
    first guess does, it starts from the last half of the span instead. With
    ``start_s`` the window start is fixed and there is no iteration.

    Args:
        time_since_init_s: Time since ``t_init``, s.
        filtered: Background-subtracted downstream pressure.
        usable: Boolean mask of the samples that may enter the fit.
        start_taus: Window start in units of τ_L.
        end_s: Window end, s after ``t_init``; defaults to the last usable
            sample.
        n_iter: Maximum number of iterations.
        start_s: Fixed window start, s after ``t_init``; overrides
            ``start_taus``.
    """
    t_rel = np.asarray(time_since_init_s, dtype=float)
    p = np.asarray(filtered, dtype=float)
    ok = np.asarray(usable, dtype=bool)
    end_s = t_rel[ok][-1] if end_s is None else end_s

    def fit(window):
        (slope, intercept), cov = np.polyfit(t_rel[window], p[window], 1, cov=True)
        return slope, -intercept / slope, cov

    if start_s is not None:
        window = ok & (t_rel >= start_s) & (t_rel <= end_s)
        slope, tau, cov = fit(window)
        return SteadyStateFit(
            slope=float(slope), time_lag_s=float(tau), used=window, covariance=cov
        )

    tau = 0.25 * end_s  # first guess
    window = ok & (t_rel >= start_taus * tau) & (t_rel <= end_s)
    if window.sum() < 4:
        # first guess already past the data: start from the last half instead
        tau = 0.5 * end_s / start_taus
        window = ok & (t_rel >= start_taus * tau) & (t_rel <= end_s)
    slope, new_tau, cov = fit(window)
    converged = False
    for _ in range(n_iter - 1):
        if abs(new_tau - tau) < 1.0:
            converged = True
            break
        tau = max(new_tau, 60.0)
        next_window = ok & (t_rel >= start_taus * tau) & (t_rel <= end_s)
        if next_window.sum() < 4:
            break  # next start lies past the data: keep the last fit
        window = next_window
        slope, new_tau, cov = fit(window)
    else:
        converged = abs(new_tau - tau) < 1.0
    return SteadyStateFit(
        slope=float(slope),
        time_lag_s=float(new_tau),
        used=window,
        covariance=cov,
        converged=converged,
    )


def diffusivity_from_time_lag(time_lag_s: float, sample_thickness_m: float) -> float:
    """Diffusivity D = e²/(6·τ), in m²/s.

    Args:
        time_lag_s: Time lag τ in seconds; must be positive.
        sample_thickness_m: Sample thickness e in m.

    Raises:
        ValueError: If ``time_lag_s`` is not positive.
    """
    if not time_lag_s > 0:
        raise ValueError(f"Time lag must be positive, got {time_lag_s:.3g} s")
    return sample_thickness_m**2 / (6.0 * time_lag_s)


def solubility_from_permeability(
    permeability: UFloat | float, diffusivity_m2_per_s: float
) -> UFloat | float:
    """Solubility S = Φ/D, in H/(m³·Pa^0.5).

    Args:
        permeability: Permeability Φ in H/(m·s·Pa^0.5) (uncertainty
            propagates through if a ufloat).
        diffusivity_m2_per_s: Diffusivity D in m²/s; must be positive.

    Raises:
        ValueError: If ``diffusivity_m2_per_s`` is not positive.
    """
    if not diffusivity_m2_per_s > 0:
        raise ValueError(
            f"Diffusivity must be positive, got {diffusivity_m2_per_s:.3g} m²/s"
        )
    return permeability / diffusivity_m2_per_s
