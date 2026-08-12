"""Voltage-to-pressure and voltage-to-temperature calibrations for SHIELD
instruments.

Each calibration is a small frozen dataclass with a ``to_torr`` (pressure
gauges) or ``to_celsius`` / ``to_kelvin`` (thermocouples) method. Instances are
attached to gauges in :mod:`shield_toolbox.config`, so loaders and analysis
code never hard-code conversion formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

# Hydrogen correction polynomial for the WGM701, ascending powers of the
# indicated pressure. Valid for 7.6e-3 < P < 76 Torr (from the gauge manual).
_WGM701_H2_COEFFS = (
    0.3391937,
    0.8666103,
    0.1400703,
    0.0460218,
    0.0001714538,
    0.0002287221,
)


@runtime_checkable
class PressureCalibration(Protocol):
    """Anything that converts a gauge voltage (V) to pressure (Torr)."""

    def to_torr(self, voltage: npt.ArrayLike) -> np.ndarray: ...


@dataclass(frozen=True)
class Baratron626D:
    """MKS Baratron 626D capacitance manometer.

    The 0–10 V output spans 0 to ``full_scale_torr`` linearly. SHIELD uses a
    1000 Torr unit upstream and a 1 Torr unit downstream.
    """

    full_scale_torr: float

    def to_torr(self, voltage: npt.ArrayLike) -> np.ndarray:
        return np.asarray(voltage, dtype=float) * self.full_scale_torr / 10.0


@dataclass(frozen=True)
class WGM701:
    """Inficon WGM701 (Wasp) gauge with hydrogen gas correction.

    The manual's log-law gives the indicated (N2-equivalent) pressure; a
    hydrogen correction is then applied: a 5th-order polynomial above
    0.1 Torr indicated (valid 7.6e-3–76 Torr) and a constant ×2.4 factor
    below (valid 7.6e-7–7.6e-3 Torr).
    """

    def to_torr(self, voltage: npt.ArrayLike) -> np.ndarray:
        indicated = 10 ** ((np.asarray(voltage, dtype=float) - 5.5) / 0.5)
        corrected_high = sum(c * indicated**i for i, c in enumerate(_WGM701_H2_COEFFS))
        corrected_low = indicated * 2.4
        return np.where(indicated < 0.1, corrected_low, corrected_high)


@dataclass(frozen=True)
class CVM211:
    """InstruTech CVM211 Stinger gauge — calibration intentionally not
    implemented (its voltage is recorded by the DAS but has never been used
    in analysis)."""

    def to_torr(self, voltage: npt.ArrayLike) -> np.ndarray:
        raise NotImplementedError(
            "The CVM211 voltage-to-pressure calibration has not been "
            "implemented; use the Baratron or WGM701 readings instead."
        )


def _typek_coefficients(mv: float) -> tuple[float, ...]:
    """NIST ITS-90 inverse polynomial coefficients for a Type K thermocouple.

    Valid from -5.891 mV to 54.886 mV; three sub-ranges.
    """
    if mv < -5.892 or mv > 54.887:
        raise ValueError("Voltage out of valid Type K range (-5.891 to 54.886 mV).")
    if mv < 0:
        # -5.891 mV to 0 mV
        return (
            0.0e0,
            2.5173462e1,
            -1.1662878e0,
            -1.0833638e0,
            -8.977354e-1,
            -3.7342377e-1,
            -8.6632643e-2,
            -1.0450598e-2,
            -5.1920577e-4,
        )
    elif mv < 20.644:
        # 0 mV to 20.644 mV
        return (
            0.0e0,
            2.508355e1,
            7.860106e-2,
            -2.503131e-1,
            8.31527e-2,
            -1.228034e-2,
            9.804036e-4,
            -4.41303e-5,
            1.057734e-6,
            -1.052755e-8,
        )
    else:
        # 20.644 mV to 54.886 mV
        return (
            -1.318058e2,
            4.830222e1,
            -1.646031e0,
            5.464731e-2,
            -9.650715e-4,
            8.802193e-6,
            -3.11081e-8,
        )


def _typek_mv_to_celsius(mv: float) -> float:
    coeffs = _typek_coefficients(mv)
    return sum(a * mv**i for i, a in enumerate(coeffs))


@dataclass(frozen=True)
class TypeKThermocouple:
    """Type K thermocouple, NIST ITS-90 inverse polynomials (mV → °C)."""

    def to_celsius(self, millivolts: npt.ArrayLike) -> np.ndarray | float:
        arr = np.asarray(millivolts, dtype=float)
        out = np.array([_typek_mv_to_celsius(mv) for mv in np.atleast_1d(arr)])
        return float(out[0]) if arr.ndim == 0 else out

    def to_kelvin(self, millivolts: npt.ArrayLike) -> np.ndarray | float:
        return self.to_celsius(millivolts) + 273.15
