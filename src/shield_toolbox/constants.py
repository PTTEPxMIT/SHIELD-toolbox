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

PA_TO_TORR = 1.0 / TORR_TO_PA
"""Torr per Pascal."""

# Takaishi–Sensui thermal-transpiration coefficients, SI units (pressure in Pa,
# tube diameter in m). Expressions kept verbatim from the original analysis
# (note 10e-5 = 1e-4 and 10e-2 = 1e-1, i.e. NOT 1e-5 / 1e-2).
TS_A = 1.24 * 56.3 / 10e-5
TS_B = 8 * 7.7 / 10e-2
TS_C = 10.6 * 2.73
