"""
This is a Model class for the Fenna-Matthews-Olson (FMO) complex.
"""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients


class FMOComplex(Model):
    """
    A model representing a Fenna-Mathes-Olson (FMO) complex.
    All unitless quantities in this model are taken to be in units of kBT at 298.15K.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "temp": 1,
            "mass": 1,
            "lambda": np.sqrt(0.05),
            "w": 117 * 0.00509506,
        }
        super().__init__(self.default_constants, constants)
        self.dh_qc_dzc_inds = None
        self.dh_qc_dzc_mels = None
        self.dh_qc_dzc_shape = None

    def initialize_constants_model(self):
        """
        Initialize the model-specific constants.
        """
        mass = self.constants.get("mass", self.default_constants.get("mass"))
        self.constants.w = self.constants.get(
            "w", self.default_constants.get("w")
        ) * np.ones(8)
        self.constants.num_classical_coordinates = 8
        self.constants.num_quantum_states = 8
        self.constants.classical_coordinate_weight = self.constants.w
        self.constants.classical_coordinate_mass = mass * np.ones(8)

    def initialize_constants_h_c(self):
        """
        Initialize the constants for the classical Hamiltonian.
        """
        self.constants.harmonic_oscillator_frequency = self.constants.get("w")

    def initialize_constants_h_qc(self):
        """
        Initialize the constants for the quantum-classical coupling Hamiltonian.
        """
        lam = self.constants.get("lambda", self.default_constants.get("lambda"))
        w = self.constants.w
        self.constants.diagonal_linear_coupling = np.diag(w * lam)

    def initialize_constants_h_q(self):
        """
        Initialize the constants for the quantum Hamiltonian.
        """

    def h_q(self, constants, parameters, **kwargs):
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = len(parameters.seed)
        if hasattr(self, "h_q_mat"):
            if self.h_q_mat is not None:
                if len(self.h_q_mat) == batch_size:
                    return self.h_q_mat
        # these are in wavenumbers
        matrix_elements = np.array(
            [
                [12505.0, 94.8, 5.5, -5.9, 7.1, -15.1, -12.2, 39.5],
                [94.8, 12425.0, 29.8, 7.6, 1.6, 13.1, 5.7, 7.9],
                [5.5, 29.8, 12195.0, -58.9, -1.2, -9.3, 3.4, 1.4],
                [-5.9, 7.6, -58.9, 12375.0, -64.1, -17.4, -62.3, -1.6],
                [7.1, 1.6, -1.2, -64.1, 12600.0, 89.5, -4.6, 4.4],
                [-15.1, 13.1, -9.3, -17.4, 89.5, 12515.0, 35.1, -9.1],
                [-12.2, 5.7, 3.4, -62.3, -4.6, 35.1, 12465.0, -11.1],
                [39.5, 7.9, 1.4, -1.6, 4.4, -9.1, -11.1, 12700.0],
            ],
            dtype=complex,
        )
        # To convert wavenumbers to units of kBT at T=298.15K we
        # multiply by each value by 0.00509506 = (1/8065.544)[eV/cm^-1] / 0.0243342[eV]
        # where 0.0243342[eV] is the value of kBT at 298.15K
        # note that all other constants in this model must also be assumed to be
        # in units of kBT at 298.15K.
        matrix_elements *= 0.00509506

        # to reduce numerical errors we can offset the diagonal elements by their minimum value
        matrix_elements = matrix_elements - np.min(
            np.diag(matrix_elements)
        ) * np.identity(8)
        # Finally we broadcast the array to the desired shape
        out = matrix_elements[np.newaxis, :, :] + np.zeros(
            (batch_size, 1, 1), dtype=complex
        )
        return out

    init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical
    hop_function = ingredients.harmonic_oscillator_hop_function
    h_c = ingredients.harmonic_oscillator_h_c
    h_qc = ingredients.diagonal_linear_h_qc
    dh_qc_dzc = ingredients.diagonal_linear_dh_qc_dzc
    dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc
    linear_h_qc = True
    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_c,
        initialize_constants_h_qc,
        initialize_constants_h_q,
    ]
