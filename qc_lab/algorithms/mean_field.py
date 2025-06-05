"""
This file contains the MF algorithm class.
"""

from qc_lab.algorithm import Algorithm
from qc_lab import tasks


class MeanField(Algorithm):
    """
    Mean-field dynamics algorithm class.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {}
        super().__init__(self.default_settings, settings)

    def _assign_to_parameters(self, sim, parameters, state):
        return tasks.assign_to_parameters(
            self, sim, parameters, state, name="seed", val=state.seed
        )

    def _initialize_z(self, sim, parameters, state):
        return tasks.initialize_z(self, sim, parameters, state, seed=state.seed)

    def _update_h_quantum(self, sim, parameters, state):
        return tasks.update_h_quantum(self, sim, parameters, state, z=state.z)

    initialization_recipe = {
        "0": tasks.assign_norm_factor_mf,
        "1": _assign_to_parameters,
        "2": _initialize_z,
        "3": _update_h_quantum,
    }

    def _update_z_rk4(self, sim, parameters, state):
        return tasks.update_z_rk4(
            self,
            sim,
            parameters,
            state,
            z=state.z,
            output_name="z",
            wf=state.wf_db,
            use_gauge_field_force=False,
        )

    update_recipe = {
        "0": _update_h_quantum,
        "1": _update_z_rk4,
        "2": tasks.update_wf_db_rk4,
    }

    def _update_quantum_energy(self, sim, parameters, state):
        return tasks.update_quantum_energy(self, sim, parameters, state, wf=state.wf_db)

    def _update_classical_energy(self, sim, parameters, state):
        return tasks.update_classical_energy(self, sim, parameters, state, z=state.z)

    output_recipe = {
        "0": tasks.update_dm_db_mf,
        "1": _update_quantum_energy,
        "2": _update_classical_energy,
    }
    output_variables = [
        "dm_db",
        "classical_energy",
        "quantum_energy",
    ]
