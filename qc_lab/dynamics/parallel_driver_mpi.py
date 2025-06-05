"""
This file contains the parallel MPI driver for the dynamics simulation in QC Lab.
"""

import warnings
import copy
import numpy as np
import qc_lab.dynamics as dynamics
from qc_lab.data import Data
from qc_lab.vector import initialize_vector_objects


def parallel_driver_mpi(sim, seeds=None, data=None, num_tasks=None):
    """
    Parallel driver for the dynamics core using the mpi4py library.
    """
    # first initialize the model constants
    sim.model.initialize_constants()
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError(
            "mpi4py is required for the parallel_driver_mpi driver"
        ) from None
    except Exception as e:
        raise RuntimeError(f"An error occurred when importing mpi4py: {e}") from None
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if num_tasks is None:
        size = comm.Get_size()
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
    # Split the simulations into chunks for each MPI process
    chunk_inds = np.linspace(0, num_sims, size + 1, dtype=int)
    start = chunk_inds[rank]
    end = chunk_inds[rank + 1]
    chunk_size = end - start
    # Create the input data for each local simulation
    sim.initialize_timesteps()
    local_input_data = [
        (
            copy.deepcopy(sim),
            *initialize_vector_objects(
                sim, batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)
            ),
            Data(),
        )
        for n in np.arange(num_sims)[start:end]
    ]
    # Set the batch size for each local simulation
    for i in range(chunk_size):
        local_input_data[i][0].settings.batch_size = len(local_input_data[i][2].seed)
    # Execute the local simulations
    local_results = [dynamics.dynamics(*x) for x in local_input_data]

    comm.Barrier()

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        final_results = [item for sublist in all_results for item in sublist]
        for result in final_results:
            data.add_data(result)
        data.data_dict["seed"] = seeds

    return data
