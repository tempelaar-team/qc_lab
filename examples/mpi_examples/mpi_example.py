"""
This is an example of how to use the parallel_driver_mpi function in QC Lab.

This script can either be executed by a scheduler like SLURM or can be called in
terminal by running

mpirun -n num_tasks python mpi_example.py

where num_tasks is the number of tasks you want to run in parallel.

In this example num_tasks is determined automatically by the mpi driver.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from qclab import Simulation
from qclab.models import SpinBoson
from qclab.algorithms import MeanField
from qclab.dynamics import parallel_driver_mpi

# instantiate a simulation
sim = Simulation()

# change settings to customize simulation
sim.settings.num_trajs = 400
sim.settings.batch_size = 100
sim.settings.tmax = 10
sim.settings.dt_update = 0.001

# instantiate a model
sim.model = SpinBoson()
# instantiate an algorithm
sim.algorithm = MeanField()
# define an initial diabatic wavefunction
sim.initial_state["wf_db"] = np.array([1, 0], dtype=complex)

data = parallel_driver_mpi(sim)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    print(data.log)
