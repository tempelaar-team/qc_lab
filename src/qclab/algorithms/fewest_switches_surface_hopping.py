"""
This module contains the FewestSwitchesSurfaceHopping algorithm class.
"""

from functools import partial
from qclab.algorithm import Algorithm
from qclab import tasks


class FewestSwitchesSurfaceHopping(Algorithm):
    """
    Fewest switches surface hopping algorithm class.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {
            "fssh_deterministic": False,
            "gauge_fixing": "sign_overlap",
            "use_gauge_field_force": False,
        }
        super().__init__(self.default_settings, settings)

    initialization_recipe = [
        tasks.initialize_variable_objects,
        tasks.initialize_norm_factor,
        tasks.initialize_branch_seeds,
        tasks.initialize_z,
        tasks.update_h_q_tot,
        partial(
            tasks.diagonalize_matrix,
            matrix_name="h_q_tot",
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        partial(
            tasks.update_eigvecs_gauge,
            gauge_fixing="phase_der_couple",
            eigvecs_previous_name="eigvecs",
        ),
        partial(tasks.copy_in_state, copy_name="eigvecs_previous", orig_name="eigvecs"),
        partial(
            tasks.update_vector_basis,
            input_vec_name="wf_db",
            basis_name="eigvecs",
            output_vec_name="wf_adb",
            adb_to_db=False,
        ),
        tasks.initialize_random_values_fssh,
        tasks.initialize_active_surface,
        tasks.initialize_dm_adb_0_fssh,
        tasks.update_act_surf_wf,
    ]

    update_recipe = [
        partial(tasks.copy_in_state, copy_name="eigvecs_previous", orig_name="eigvecs"),
        # Begin RK4 integration steps.
        tasks.update_classical_force,
        partial(
            tasks.update_quantum_classical_force,
            wf_db_name="act_surf_wf",
            wf_changed=True,
        ),
        tasks.update_z_rk4_k123,
        partial(tasks.update_classical_force, z_name="z_1"),
        partial(
            tasks.update_quantum_classical_force,
            wf_db_name="act_surf_wf",
            z_name="z_1",
            wf_changed=False,
        ),
        partial(tasks.update_z_rk4_k123, z_name="z", z_k_name="z_2", k_name="z_rk4_k2"),
        partial(tasks.update_classical_force, z_name="z_2"),
        partial(
            tasks.update_quantum_classical_force,
            wf_db_name="act_surf_wf",
            z_name="z_2",
            wf_changed=False,
        ),
        partial(
            tasks.update_z_rk4_k123,
            z_name="z",
            z_k_name="z_3",
            k_name="z_rk4_k3",
            dt_factor=1.0,
        ),
        partial(tasks.update_classical_force, z_name="z_3"),
        partial(
            tasks.update_quantum_classical_force,
            wf_db_name="act_surf_wf",
            z_name="z_3",
            wf_changed=False,
        ),
        tasks.update_z_rk4_k4,
        # End RK4 integration steps.
        tasks.update_wf_db_propagator,
        tasks.update_h_q_tot,
        partial(
            tasks.diagonalize_matrix,
            matrix_name="h_q_tot",
            eigvals_name="eigvals",
            eigvecs_name="eigvecs",
        ),
        tasks.update_eigvecs_gauge,
        partial(
            tasks.update_vector_basis,
            input_vec_name="wf_db",
            basis_name="eigvecs",
            output_vec_name="wf_adb",
            adb_to_db=False,
        ),
        tasks.update_hop_prob_fssh,
        tasks.update_hop_inds_fssh,
        tasks.update_hop_vals_fssh,
        tasks.update_z_hop,
        tasks.update_act_surf_hop,
        tasks.update_act_surf_wf,
    ]

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_fssh,
        tasks.update_quantum_energy_act_surf,
        tasks.update_classical_energy_fssh,
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_quantum_energy,
        tasks.collect_classical_energy,
    ]
