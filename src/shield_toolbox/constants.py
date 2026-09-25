"""Universal physical constants and unit conversions.

Nothing rig-specific belongs here — rig hardware constants live in
:mod:`shield_toolbox.config`.
"""

R = 8.314
"""Molar gas constant, J/(mol·K)."""

K_B_EV = 8.617333262e-5
"""Boltzmann constant, eV/K."""

N_A = 6.022e23
"""Avogadro constant, 1/mol."""

TORR_TO_PA = 133.322
"""Pascals per Torr."""

ZERO_CELSIUS_K = 273.15
"""0 °C in kelvin."""

PA_TO_TORR = 1.0 / TORR_TO_PA
"""Torr per Pascal."""

# Takaishi–Sensui thermal-transpiration constants for H2 (Takaishi & Sensui
# 1963, Trans. Faraday Soc. 59, 2503, Table 1). They go with the reduced variable
# X = 2·p·d/(T1 + T2), with p in Torr (mmHg), d in mm and T in K.
TS_A_H2 = 1.24e5
"""K²·Torr⁻²·mm⁻²."""

TS_B_H2 = 8.00e2
"""K·Torr⁻¹·mm⁻¹."""

TS_C_H2 = 10.6
"""K^½·Torr^-½·mm^-½."""
