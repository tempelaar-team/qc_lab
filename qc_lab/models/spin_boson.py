"""
This file contains the spin-boson Model class.
"""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients


class SpinBoson(Model):
    """
    Spin-Boson model class for the simulation framework.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "temp": 1,
            "V": 0.5,
            "E": 0.5,
            "A": 100,
            "W": 0.1,
            "l_reorg": 0.02 / 4,
            "boson_mass": 1,
        }
        super().__init__(self.default_constants, constants)
        self.dh_qc_dzc_inds = None
        self.dh_qc_dzc_mels = None
        self.dh_qc_dzc_shape = None

    def initialize_constants_model(self):
        """
        Initialize the model-specific constants.
        """
        num_bosons = self.constants.get("A", self.default_constants.get("A"))
        char_freq = self.constants.get("W", self.default_constants.get("W"))
        boson_mass = self.constants.get(
            "boson_mass", self.default_constants.get("boson_mass")
        )
        self.constants.w = char_freq * np.tan(
            ((np.arange(num_bosons) + 1) - 0.5) * np.pi / (2 * num_bosons)
        )
        self.constants.num_classical_coordinates = num_bosons
        self.constants.num_quantum_states = 2
        self.constants.classical_coordinate_weight = self.constants.w
        self.constants.classical_coordinate_mass = boson_mass * np.ones(num_bosons)

    def initialize_constants_h_c(self):
        """
        Initialize the constants for the classical Hamiltonian.
        """
        w = self.constants.get("w", self.default_constants.get("w"))
        self.constants.harmonic_oscillator_frequency = w

    def initialize_constants_h_qc(self):
        """
        Initialize the constants for the quantum-classical coupling Hamiltonian.
        """
        num_bosons = self.constants.get("A", self.default_constants.get("A"))
        w = self.constants.get("w", self.default_constants.get("w"))
        l_reorg = self.constants.get("l_reorg", self.default_constants.get("l_reorg"))
        self.constants.spin_boson_coupling = w * np.sqrt(2 * l_reorg / num_bosons)

    def initialize_constants_h_q(self):
        """
        Initialize the constants for the quantum Hamiltonian.
        """
        self.constants.two_level_system_a = self.constants.get(
            "E", self.default_constants.get("E")
        )
        self.constants.two_level_system_b = -self.constants.get(
            "E", self.default_constants.get("E")
        )
        self.constants.two_level_system_c = self.constants.get(
            "V", self.default_constants.get("V")
        )
        self.constants.two_level_system_d = 0


    init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical
    hop_function = ingredients.harmonic_oscillator_hop_function
    h_c = ingredients.harmonic_oscillator_h_c
    h_q = ingredients.two_level_system_h_q
    h_qc = ingredients.spin_boson_h_qc
    dh_qc_dzc = ingredients.spin_boson_dh_qc_dzc
    dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc
    linear_h_qc = True
    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_c,
        initialize_constants_h_qc,
        initialize_constants_h_q,
    ]
