"""
This module contains numerical constants.
"""

# Numerical threshold for near zero values.
SMALL = 1e-10

# Numerical threshold for nonadiabatic coupling gauge fixing.
# The misalignment is allowed to be 1e-3 of the magnitude of the coupling along each coordinate.
GAUGE_FIX_THRESHOLD = 1e-3

# Conversion factor from wavenumbers to kBT at 300 K.
INVCM_TO_300K = 1 / 208.521

# Finite difference step size.
FINITE_DIFFERENCE_DELTA = 1e-6
