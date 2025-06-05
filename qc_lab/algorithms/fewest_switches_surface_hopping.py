"""
This file contains the FSSH algorithm class.
"""

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

    def _initialize_z(self, sim, parameters, state):
        return tasks.initialize_z(self, sim, parameters, state, seed=state.seed)

    def _update_h_quantum(self, sim, parameters, state):
        return tasks.update_h_quantum(self, sim, parameters, state, z=state.z)

    def _diagonalize_matrix(self, sim, parameters, state):
        return tasks.diagonalize_matrix(
            self,
            sim,
            parameters,
            state,
            matrix=state.h_quantum,
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        )

    def _gauge_fix_eigs_init(self, sim, parameters, state):
        return tasks.gauge_fix_eigs(
            self,
            sim,
            parameters,
            state,
            eigvals=state.eigvals,
            eigvecs=state.eigvecs,
            eigvecs_previous=state.eigvecs,
            output_eigvecs_name="eigvecs",
            z=state.z,
            gauge_fixing=2,
        )

    def _assign_eigvecs_to_state(self, sim, parameters, state):
        return tasks.assign_to_state(
            self,
            sim,
            parameters,
            state,
            val=state.eigvecs,
            name="eigvecs_previous",
        )

    def _assign_eigvals_to_state(self, sim, parameters, state):
        return tasks.assign_to_state(
            self,
            sim,
            parameters,
            state,
            val=state.eigvecs,
            name="eigvals_previous",
        )

    def _basis_transform_vec(self, sim, parameters, state):
        return tasks.basis_transform_vec(
            self,
            sim,
            parameters,
            state,
            input_vec=state.wf_db,
            basis=np.einsum("...ij->...ji", state.eigvecs).conj(),
            output_name="wf_adb",
        )

    def _update_quantum_energy(self, sim, parameters, state):
        return tasks.update_quantum_energy(
            self,
            sim,
            parameters,
            state,
            wf=state.act_surf_wf,
        )

    initialization_recipe = [
        tasks.assign_norm_factor_fssh,
        tasks.initialize_branch_seeds,
        _initialize_z,
        _update_h_quantum,
        _diagonalize_matrix,
        _gauge_fix_eigs_init,
        _assign_eigvecs_to_state,
        _assign_eigvals_to_state,
        _basis_transform_vec,
        tasks.initialize_random_values_fssh,
        tasks.initialize_active_surface,
        tasks.initialize_dm_adb_0_fssh,
        tasks.update_act_surf_wf,
        _update_quantum_energy,
    ]

    def _update_z_rk4(self, sim, parameters, state):
        return tasks.update_z_rk4(
            self,
            sim,
            parameters,
            state,
            z=state.z,
            output_name="z",
            wf=state.act_surf_wf,
            use_gauge_field_force=True,
        )

    def _update_wf_db_eigs(self, sim, parameters, state):
        return tasks.update_wf_db_eigs(
            self,
            sim,
            parameters,
            state,
            wf_db=state.wf_db,
            eigvals=state.eigvals,
            eigvecs=state.eigvecs,
            adb_name="wf_adb",
            output_name="wf_db",
        )

    def _gauge_fix_eigs_update(self, sim, parameters, state):
        return tasks.gauge_fix_eigs(
            self,
            sim,
            parameters,
            state,
            eigvals=state.eigvals,
            eigvecs=state.eigvecs,
            eigvecs_previous=state.eigvecs_previous,
            output_eigvecs_name="eigvecs",
            z=state.z,
            gauge_fixing=sim.algorithm.settings.gauge_fixing,
        )

    update_recipe = [
        _assign_eigvecs_to_state,
        _assign_eigvals_to_state,
        _update_z_rk4,
        _update_wf_db_eigs,
        _update_h_quantum,
        _diagonalize_matrix,
        _gauge_fix_eigs_update,
        tasks.update_active_surface_fssh,
        tasks.update_act_surf_wf,
    ]

    def _update_quantum_energy_fssh(self, sim, parameters, state):
        return tasks.update_quantum_energy_fssh(
            self,
            sim,
            parameters,
            state,
            wf=state.act_surf_wf,
        )

    def _update_classical_energy_fssh(self, sim, parameters, state):
        return tasks.update_classical_energy_fssh(
            self, sim, parameters, state, z=state.z
        )

    def _update_classical_energy_fssh(self, sim, parameters, state):
        return tasks.update_classical_energy_fssh(
            self, sim, parameters, state, z=state.z
        )

    output_recipe = [
        tasks.update_dm_db_fssh,
        _update_quantum_energy_fssh,
        _update_classical_energy_fssh,
    ]
    output_variables = [
        "quantum_energy",
        "classical_energy",
        "dm_db",
    ]
