"""
This file contains the parallel driver for the dynamics simulation in QC Lab.
"""

import multiprocessing
import warnings
import copy
import numpy as np
import qc_lab.dynamics as dynamics
from qc_lab.data import Data
from qc_lab.vector import initialize_vector_objects


def parallel_driver_multiprocessing(sim, seeds=None, data=None, num_tasks=None):
    """
    Parallel driver for the dynamics core using the python library multiprocessing.
    """
    if seeds is None:
        offset = 0
        if data is not None:
            if len(data.data_dict["seed"]) > 0:
                offset = np.max(data.data_dict["seed"]) + 1
        else:
            data = Data()
        seeds = offset + np.arange(sim.settings.num_trajs, dtype=int)
        num_trajs = sim.settings.num_trajs
    else:
        num_trajs = len(seeds)
        warnings.warn(
            "Setting sim.settings.num_trajs to the number of provided seeds.",
            UserWarning,
        )
        sim.settings.num_trajs = num_trajs
    if num_tasks is None:
        size = multiprocessing.cpu_count()
    else:
        size = num_tasks
    if sim.settings.num_trajs % sim.settings.batch_size == 0:
        num_sims = sim.settings.num_trajs // sim.settings.batch_size
    else:
        num_sims = sim.settings.num_trajs // sim.settings.batch_size + 1
    batch_seeds_list = (
        np.zeros((num_sims * sim.settings.batch_size), dtype=int) + np.nan
    )
    batch_seeds_list[:num_trajs] = seeds
    batch_seeds_list = batch_seeds_list.reshape((num_sims, sim.settings.batch_size))
    sim.initialize_timesteps()
    input_data = [
        (
            copy.deepcopy(sim),
            *initialize_vector_objects(
                sim, batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)
            ),
            Data(),
        )
        for n in range(num_sims)
    ]
    for i in range(num_sims):
        input_data[i][0].settings.batch_size = len(input_data[i][2].seed)
    with multiprocessing.Pool(processes=size) as pool:
        results = pool.starmap(dynamics.dynamics, input_data)
    for result in results:
        data.add_data(result)
    data.data_dict["seed"] = seeds
    return data


def parallel_driver_multiprocessing_(sim, seeds=None, data=None, num_tasks=None):
    """
    Parallel driver for the dynamics core using the python library multiprocessing.
    """
    if seeds is None:
        offset = 0
        if data is not None:
            if len(data.data_dict["seed"]) > 0:
                offset = np.max(data.data_dict["seed"]) + 1
        else:
            data = Data()
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
    if num_tasks is None:
        size = multiprocessing.cpu_count()
    else:
        size = num_tasks
    num_sims = sim.settings.num_trajs // sim.settings.batch_size
    if num_sims % size != 0:
        warnings.warn(
            "The number of batches to run is not divisible by the number of processors.\n \
            Setting the number of batches to the lower multiple of size.\n"
            + " running "
            + str((num_sims // size) * size * sim.settings.batch_size)
            + " trajectories.",
            UserWarning,
        )
        num_sims = (num_sims // size) * size
        seeds = seeds[: num_sims * sim.settings.batch_size]
    print(
        "running ",
        num_sims * sim.settings.batch_size,
        "trajectories in batches of",
        sim.settings.batch_size,
        "on",
        size,
        "tasks.",
    )
    batch_seeds_list = seeds.reshape((num_sims, sim.settings.batch_size))
    sim.initialize_timesteps()
    input_data = [
        (sim, *initialize_vector_objects(sim, batch_seeds_list[n]), Data())
        for n in range(num_sims)
    ]
    with multiprocessing.Pool(processes=size) as pool:
        results = pool.starmap(dynamics.dynamics, input_data)
    for result in results:
        data.add_data(result)
    data.data_dict["seed"] = seeds
    return data
