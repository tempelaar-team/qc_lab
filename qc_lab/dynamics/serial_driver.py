"""
This file contains the serial driver for the dynamics core.
"""

import warnings
import numpy as np
from qc_lab.data import Data
from qc_lab.vector import initialize_vector_objects
import qc_lab.dynamics as dynamics


def serial_driver(sim, seeds=None, data=None):
    """
    Serial driver for the dynamics core.
    """
    if data is None:
        data = Data()
    if seeds is None:
        if len(data.data_dict["seed"]) > 0:
            offset = np.max(data.data_dict["seed"]) + 1
        else:
            offset = 0
        seeds = offset + np.arange(sim.settings.num_trajs, dtype=int)
        num_trajs = sim.settings.num_trajs
    else:
        num_trajs = len(seeds)
        warnings.warn(
            "Setting sim.settings.num_trajs to the number of provided seeds.",
            UserWarning,
        )
        sim.settings.num_trajs = num_trajs
    # determine the number of simulations required to execute the total number of trajectories.
    num_sims = int(num_trajs / sim.settings.batch_size) + 1
    for n in range(num_sims):
        batch_seeds = seeds[
            n * sim.settings.batch_size : (n + 1) * sim.settings.batch_size
        ]
        if len(batch_seeds) == 0:
            break
        sim.settings.batch_size = len(batch_seeds)
        sim.initialize_timesteps()
        parameters, state = initialize_vector_objects(sim, batch_seeds)
        new_data = Data()
        new_data.data_dict["seed"] = state.seed
        new_data = dynamics.dynamics(sim, parameters, state, new_data)
        data.add_data(new_data)
    return data
