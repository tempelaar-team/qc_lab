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
            "kBT": 1,
            "boson_mass": 1,
            "l_reorg": 35 * 0.00509506,  # reorganization energy
            "W": 106.14 * 0.00509506,  # characteristic frequency
            "A": 200,
        }
        super().__init__(self.default_constants, constants)
        self.dh_qc_dzc_inds = None
        self.dh_qc_dzc_mels = None
        self.dh_qc_dzc_shape = None

    def initialize_constants_model(self):
        """
        Initialize the model-specific constants.
        """
        self.constants.num_quantum_states = 7
        num_bosons = self.constants.get("A", self.default_constants.get("A"))
        char_freq = self.constants.get("W", self.default_constants.get("W"))
        boson_mass = self.constants.get(
            "boson_mass", self.default_constants.get("boson_mass")
        )

        self.constants.w = (
            char_freq
            * np.tan(((np.arange(num_bosons) + 1) - 0.5) * np.pi / (2 * num_bosons))[
                np.newaxis, :
            ]
            * np.ones((self.constants.num_quantum_states, num_bosons))
        ).flatten()
        self.constants.num_classical_coordinates = (
            self.constants.num_quantum_states
            * self.constants.get("A", self.default_constants.get("A"))
        )
        self.constants.classical_coordinate_weight = self.constants.w
        self.constants.classical_coordinate_mass = boson_mass * np.ones(
            self.constants.num_classical_coordinates
        )

    def initialize_constants_h_c(self):
        """
        Initialize the constants for the classical Hamiltonian.
        """
        self.constants.harmonic_oscillator_frequency = self.constants.w

    def initialize_constants_h_qc(self):
        """
        Initialize the constants for the quantum-classical coupling Hamiltonian.
        """
        num_bosons = self.constants.get("A", self.default_constants.get("A"))
        l_reorg = self.constants.get("l_reorg", self.default_constants.get("l_reorg"))
        m = self.constants.classical_coordinate_mass
        h = self.constants.classical_coordinate_weight
        w = self.constants.w
        self.constants.diagonal_linear_coupling = np.zeros(
            (
                self.constants.num_quantum_states,
                self.constants.num_classical_coordinates,
            )
        )
        for n in range(self.constants.num_quantum_states):
            self.constants.diagonal_linear_coupling[
                n, n * num_bosons : (n + 1) * num_bosons
            ] = (w * np.sqrt(2 * l_reorg / num_bosons) * (1 / np.sqrt(2 * m * h)))[
                n * num_bosons : (n + 1) * num_bosons
            ]

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
                [12410, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
                [-87.7, 12530, 30.8, 8.2, 0.7, 11.8, 4.3],
                [5.5, 30.8, 12210.0, -53.5, -2.2, -9.6, 6.0],
                [-5.9, 8.2, -53.5, 12320, -70.7, -17.0, -63.3],
                [6.7, 0.7, -2.2, -70.7, 12480, 81.1, -1.3],
                [-13.7, 11.8, -9.6, -17.0, 81.1, 12630, 39.7],
                [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 12440],
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
        ) * np.identity(self.constants.num_quantum_states)
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
