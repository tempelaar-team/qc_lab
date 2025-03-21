"""
This file contains the Holstein lattice Model class.
"""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients


class HolsteinLattice(Model):
    """
    A model representing a nearest-neighbor tight-binding model
    with Holstein-type electron-phonon coupling with a
    single optical mode.
    """

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}
        self.default_parameters = {
            "temp": 1,
            "g": 0.5,
            "w": 0.5,
            "N": 10,
            "j": 1,
            "phonon_mass": 1,
            "periodic_boundary": True,
        }
        super().__init__(self.default_parameters, parameters)
        self.dh_qc_dzc_inds = None
        self.dh_qc_dzc_mels = None
        self.dh_qc_dzc_shape = None

    def initialize_constants_model(self):
        num_sites = self.constants.get("N", self.default_parameters.get("N"))
        w = self.constants.get("w", self.default_parameters.get("w"))
        phonon_mass = self.constants.get(
            "phonon_mass", self.default_parameters.get("phonon_mass")
        )
        self.constants.num_quantum_states = num_sites
        self.constants.num_classical_coordinates = num_sites
        self.constants.classical_coordinate_weight = w * np.ones(num_sites)
        self.constants.classical_coordinate_mass = phonon_mass * np.ones(num_sites)

    def initialize_constants_h_q(self):
        j = self.constants.get("j", self.default_parameters.get("j"))
        periodic_boundary = self.constants.get(
            "periodic_boundary", self.default_parameters.get("periodic_boundary")
        )
        self.constants.nearest_neighbor_lattice_hopping_energy = j
        self.constants.nearest_neighbor_lattice_periodic_boundary = periodic_boundary

    def initialize_constants_h_qc(self):
        num_sites = self.constants.get("N", self.default_parameters.get("N"))
        w = self.constants.get("w", self.default_parameters.get("w"))
        g = self.constants.get("g", self.default_parameters.get("g"))
        self.constants.holstein_coupling_oscillator_frequency = w * np.ones(num_sites)
        self.constants.holstein_coupling_dimensionless_coupling = g * np.ones(num_sites)

    def initialize_constants_h_c(self):
        num_sites = self.constants.get("N", self.default_parameters.get("N"))
        w = self.constants.get("w", self.default_parameters.get("w"))
        self.constants.harmonic_oscillator_frequency = w * np.ones(num_sites)

    h_q = ingredients.nearest_neighbor_lattice_h_q
    dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc
    h_c = ingredients.harmonic_oscillator_h_c
    h_qc = ingredients.holstein_coupling_h_qc
    dh_qc_dzc = ingredients.holstein_coupling_dh_qc_dzc
    init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical
    hop_function = ingredients.harmonic_oscillator_hop_function
    linear_h_qc = True
    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_c,
        initialize_constants_h_qc,
        initialize_constants_h_q,
    ]
