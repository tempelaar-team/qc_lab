"""

This script can either be executed by a scheduler like SLURM or can be called in terminal by running

mpirun -n num_tasks python mpi_example.py

where num_tasks is the number of tasks you want to run in parallel.

In this example num_tasks is determined automatically by the mpi driver.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from qc_lab import Simulation
from qc_lab.models import SpinBoson
from qc_lab.algorithms import MeanField
from qc_lab.dynamics import parallel_driver_mpi

# instantiate a simulation
sim = Simulation()

# change settings to customize simulation
sim.settings.num_trajs = 200
sim.settings.batch_size = 50
sim.settings.tmax = 10
sim.settings.dt = 0.01


# instantiate a model
sim.model = SpinBoson({
    'V':0.5,
    'E':0.5,
    'A':100,
    'W':0.1,
    'l_reorg':0.005,
    'boson_mass':1.0,
    'kBT':1.0,
})

# instantiate an algorithm
sim.algorithm = MeanField()
# define an initial diabatic wavefunction
sim.state.wf_db = np.array([1, 0], dtype=complex)

data = parallel_driver_mpi(sim)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    print(data.data_dict["seed"])
    data.save_as_h5("./tests/mpi_example.h5")

