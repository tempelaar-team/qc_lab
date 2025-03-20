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
        if len(data.data_dic["seed"]) > 0:
            offset = np.max(data.data_dic["seed"]) + 1
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
    if sim.settings.num_trajs % sim.settings.batch_size != 0:
        # The reason we enforce this is because it is possible for a simulation to generate
        # intermediate quantities that are dependent on the batch size. To avoid an error
        # we require that all simulations are run with the same batch size.
        warnings.warn(
            "The number of trajectories is not divisible by the batch size.\n \
            Setting num_trajs to the lower multiple of batch_size.",
            UserWarning,
        )
        sim.settings.num_trajs = (
            int(sim.settings.num_trajs / sim.settings.batch_size)
            * sim.settings.batch_size
        )
        seeds = seeds[: sim.settings.num_trajs]

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
        new_data = dynamics.dynamics(sim, parameters, state, new_data)
        new_data.data_dic["seed"] = state.seed
        data.add_data(new_data)
    return data
