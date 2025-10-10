"""
This module contains the parallel driver.
"""

import multiprocessing
import logging
import copy
import numpy as np
import qclab.dynamics as dynamics
from qclab.utils import get_log_output
from qclab import Variable, Data

logger = logging.getLogger(__name__)


def parallel_driver_multiprocessing(sim, seeds=None, data=None, num_tasks=None):
    """
    Parallel driver for the dynamics core using the python library multiprocessing.

    .. rubric:: Args
    sim: Simulation
        The simulation object containing the model, algorithm, initial state, and settings.
    seeds: ndarray, optional
        An array of integer seeds for the trajectories. If None, seeds will be
        generated automatically.
    data: Data, optional
        A Data object for collecting output data. If None, a new Data object
        will be created.
    num_tasks: int, optional
        The number of tasks to use for parallel processing. If None, the
        number of available tasks will be used.

    .. rubric:: Returns
    data: Data
        The updated Data object containing collected output data.
    """
    # First initialize the model constants.
    sim.model.initialize_constants()
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
        logger.warning(
            "Setting sim.settings.num_trajs to the number of provided seeds: %s",
            num_trajs,
        )
        sim.settings.num_trajs = num_trajs
    if num_tasks is None:
        size = multiprocessing.cpu_count()
    else:
        size = num_tasks
    logger.info("Using %s CPU cores for parallel processing.", size)
    # Determine the number of simulations required to execute the total number
    # of trajectories.
    if num_trajs % sim.settings.batch_size == 0:
        num_sims = num_trajs // sim.settings.batch_size
    else:
        num_sims = num_trajs // sim.settings.batch_size + 1
    logger.info(
        "Running %s simulations with %s seeds in each batch.",
        num_sims,
        sim.settings.batch_size,
    )
    batch_seeds_list = (
        np.zeros((num_sims * sim.settings.batch_size), dtype=int) + np.nan
    )
    batch_seeds_list[:num_trajs] = seeds
    batch_seeds_list = batch_seeds_list.reshape((num_sims, sim.settings.batch_size))
    sim.initialize_timesteps()
    input_data = [
        (
            copy.deepcopy(sim),
            Variable(
                {
                    "seed": batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(
                        int
                    )
                }
            ),
            Variable(),
            Data(batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)),
        )
        for n in range(num_sims)
    ]
    for i in range(num_sims):
        # Determine the batch size from the seeds in the state object.
        input_data[i][0].settings.batch_size = len(input_data[i][1].seed)
        logger.info(
            "Running simulation %s with seeds %s.", i + 1, input_data[i][1].seed
        )
    with multiprocessing.Pool(processes=size) as pool:
        results = pool.starmap(dynamics.run_dynamics, input_data)
    for result in results:
        data.add_data(result)
    # Attach collected log output.
    data.log = get_log_output()
    return data
