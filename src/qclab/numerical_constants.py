"""
This module contains numerical constants.

Values for physical constants are taken from 2022 CODATA recommended values.
Reference publication:
Mohr et al. Rev. Mod. Phys. 2025, 97 (2), 025002. https://doi.org/10.1103/RevModPhys.97.025002.
"""

import numpy as np

# Numerical threshold for near zero values.
SMALL = 1e-10

# Numerical threshold for nonadiabatic coupling gauge fixing.
# The misalignment is allowed to be 1e-3 of the magnitude of the coupling.
GAUGE_FIX_THRESHOLD = 1e-3

# Finite difference step size.
FINITE_DIFFERENCE_DELTA = 1e-6

# Speed of light [m/s].
C_M_PER_S = 299792458

# Planck constant [J*s].
H_J_S = 6.62607015e-34

# Conversion from Joules to inverse centimeters.
# J_TO_INVCM = 1 / (100[cm/m] * c[m/s] * h[J*s])
# A[J] * J_TO_INVCM = A[INVCM]
J_TO_INVCM = 1 / (100 * C_M_PER_S * H_J_S)

# Boltzmann constant [J/K].
KB_J_PER_K = 1.380649e-23

# Boltzmann constant [invcm/K]
KB_INVCM_PER_K = KB_J_PER_K * J_TO_INVCM

# Reduced Planck constant [J*s].
HBAR_J_S = H_J_S / (2 * np.pi)

# Thermal energy [invcm] at a reference temperature of 300 [K].
KBT_300K_INVCM = KB_INVCM_PER_K * 300

# Conversion between inverse centimeters to reference energy.
# A [INVCM] * INVCM_TO_300K = A [300K]
INVCM_TO_300K = 1 / KBT_300K_INVCM