"""
This file contains the dynamics core.
"""

import numpy as np
from tqdm import tqdm


def dynamics(sim, parameter_vector, state_vector, data):
    """
    Dynamics core for QC Lab.
    """
    # Execute initialization recipe.
    parameter_vector, state_vector = sim.algorithm.execute_initialization_recipe(
        sim, parameter_vector, state_vector
    )
    # Iterate over each time step.
    for sim.t_ind in tqdm(sim.settings.tdat_n):
        # Detect output timesteps.
        if np.mod(sim.t_ind, sim.settings.dt_output_n) == 0:
            # Calculate output variables.
            parameter_vector, state_vector = sim.algorithm.execute_output_recipe(
                sim, parameter_vector, state_vector
            )
            # Collect output variables into a dictionary.
            state_vector.collect_output_variables(sim.algorithm.output_variables)
            # Collect totals in output dictionary.
            data.add_to_output_total_arrays(sim, state_vector, sim.t_ind)
        # Execute update recipe.
        parameter_vector, state_vector = sim.algorithm.execute_update_recipe(
            sim, parameter_vector, state_vector
        )
    return data
