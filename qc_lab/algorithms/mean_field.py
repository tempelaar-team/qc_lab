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

    initialization_recipe = [
        tasks.assign_norm_factor_mf,
        lambda sim, parameters, state: tasks.assign_to_parameters(
            sim, parameters, state, name="seed", val=state.seed
        ),
        lambda sim, parameters, state: tasks.initialize_z(
            sim=sim, parameters=parameters, state=state, seed=state.seed
        ),
        lambda sim, parameters, state: tasks.update_h_quantum(
            sim=sim, parameters=parameters, state=state, z=state.z
        ),
    ]
    update_recipe = [
        lambda sim, parameters, state: tasks.update_h_quantum(
            sim=sim, parameters=parameters, state=state, z=state.z
        ),
        lambda sim, parameters, state: tasks.update_z_rk4(
            sim=sim,
            parameters=parameters,
            state=state,
            z=state.z,
            output_name="z",
            wf=state.wf_db,
        ),
        tasks.update_wf_db_rk4,
    ]
    output_recipe = [
        tasks.update_dm_db_mf,
        lambda sim, parameters, state: tasks.update_quantum_energy(
            sim=sim, parameters=parameters, state=state, wf=state.wf_db
        ),
        lambda sim, parameters, state: tasks.update_classical_energy(
            sim=sim, parameters=parameters, state=state, z=state.z
        ),
    ]
    output_variables = [
        "dm_db",
        "classical_energy",
        "quantum_energy",
    ]
