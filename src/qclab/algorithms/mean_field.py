"""
This module contains the MeanField algorithm class.
"""

from functools import partial
from qclab.algorithm import Algorithm
from qclab import tasks


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
        tasks.initialize_variable_objects,
        tasks.initialize_norm_factor,
        tasks.initialize_z,
        tasks.update_h_q_tot,
    ]

    update_recipe = [
        # Begin RK4 integration steps.
        partial(tasks.update_classical_force, z_name="z"),
        tasks.update_quantum_classical_force,
        tasks.update_z_rk4_k123,
        partial(tasks.update_classical_force, z_name="z_1"),
        partial(
            tasks.update_quantum_classical_force,
            z_name="z_1",
            wf_changed=False,
        ),
        partial(tasks.update_z_rk4_k123, z_name="z", z_k_name="z_2", k_name="z_rk4_k2"),
        partial(tasks.update_classical_force, z_name="z_2"),
        partial(
            tasks.update_quantum_classical_force,
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
            z_name="z_3",
            wf_changed=False,
        ),
        tasks.update_z_rk4_k4,
        # End RK4 integration steps.
        tasks.update_wf_db_rk4,
        tasks.update_h_q_tot,
    ]

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_wf,
        tasks.update_quantum_energy_wf,
        tasks.update_classical_energy,
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_classical_energy,
        tasks.collect_quantum_energy,
    ]
