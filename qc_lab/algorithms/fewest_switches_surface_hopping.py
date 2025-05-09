"""
This file contains the FSSH algorithm class.
"""

import warnings
import numpy as np
from qc_lab.algorithm import Algorithm
from qc_lab import tasks


class FewestSwitchesSurfaceHopping(Algorithm):
    """
    Fewest switches surface hopping algorithm class.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {
            "fssh_deterministic": False,
            "gauge_fixing": 0,
        }
        super().__init__(self.default_settings, settings)

    def update_algorithm_settings(self):
        pass

    initialization_recipe = [
        tasks.assign_norm_factor_fssh,
        tasks.initialize_branch_seeds,
        lambda sim, parameters, state: tasks.initialize_z(
            sim=sim, parameters=parameters, state=state, seed=state.seed
        ),
        lambda sim, parameters, state: tasks.update_h_quantum(
            sim=sim,
            parameters=parameters,
            state=state,
            z=state.z,
        ),
        lambda sim, parameters, state: tasks.diagonalize_matrix(
            sim=sim,
            parameters=parameters,
            state=state,
            matrix=state.h_quantum,
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        lambda sim, parameters, state: tasks.gauge_fix_eigs(
            sim=sim,
            parameters=parameters,
            state=state,
            eigvals=state.eigvals,
            eigvecs=state.eigvecs,
            eigvecs_previous=state.eigvecs,
            output_eigvecs_name="eigvecs",
            z=state.z,
            gauge_fixing=2,
        ),
        lambda sim, parameters, state: tasks.assign_to_state(
            sim=sim,
            parameters=parameters,
            state=state,
            val=state.eigvecs,
            name="eigvecs_previous",
        ),
        lambda sim, parameters, state: tasks.assign_to_state(
            sim=sim,
            parameters=parameters,
            state=state,
            val=state.eigvals,
            name="eigvals_previous",
        ),
        lambda sim, parameters, state: tasks.basis_transform_vec(
            sim=sim,
            parameters=parameters,
            state=state,
            input_vec=state.wf_db,
            basis=np.einsum("...ij->...ji", state.eigvecs).conj(),
            output_name="wf_adb",
        ),
        tasks.initialize_random_values_fssh,
        tasks.initialize_active_surface,
        tasks.initialize_dm_adb_0_fssh,
        tasks.update_act_surf_wf,
        lambda sim, parameters, state: tasks.update_quantum_energy(
            sim=sim, parameters=parameters, state=state, wf=state.act_surf_wf
        ),
    ]
    update_recipe = [
        lambda sim, parameters, state: tasks.assign_to_state(
            sim=sim,
            parameters=parameters,
            state=state,
            val=state.eigvecs,
            name="eigvecs_previous",
        ),
        lambda sim, parameters, state: tasks.assign_to_state(
            sim=sim,
            parameters=parameters,
            state=state,
            val=state.eigvals,
            name="eigvals_previous",
        ),
        lambda sim, parameters, state: tasks.update_z_rk4(
            sim=sim,
            parameters=parameters,
            state=state,
            wf=state.act_surf_wf,
            z=state.z,
            output_name="z",
            use_gauge_field_force=True,
        ),
        lambda sim, parameters, state: tasks.update_wf_db_eigs(
            sim=sim,
            parameters=parameters,
            state=state,
            wf_db=state.wf_db,
            eigvals=state.eigvals,
            eigvecs=state.eigvecs,
            adb_name="wf_adb",
            output_name="wf_db",
        ),
        lambda sim, parameters, state: tasks.update_h_quantum(
            sim=sim,
            parameters=parameters,
            state=state,
            z=state.z,
        ),
        lambda sim, parameters, state: tasks.diagonalize_matrix(
            sim=sim,
            parameters=parameters,
            state=state,
            matrix=state.h_quantum,
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        lambda sim, parameters, state: tasks.gauge_fix_eigs(
            sim=sim,
            parameters=parameters,
            state=state,
            eigvals=state.eigvals,
            eigvecs=state.eigvecs,
            eigvecs_previous=state.eigvecs_previous,
            output_eigvecs_name="eigvecs",
            z=state.z,
            gauge_fixing=sim.algorithm.settings.gauge_fixing,
        ),
        tasks.update_active_surface_fssh,
        tasks.update_act_surf_wf,
    ]
    output_recipe = [
        tasks.update_dm_db_fssh,
        lambda sim, parameters, state: tasks.update_quantum_energy_fssh(
            sim=sim,
            parameters=parameters,
            state=state,
            wf=state.act_surf_wf,
        ),
        lambda sim, parameters, state: tasks.update_classical_energy_fssh(
            sim=sim,
            parameters=parameters,
            state=state,
            z=state.z,
        ),
    ]
    output_variables = [
        "quantum_energy",
        "classical_energy",
        "dm_db",
    ]
