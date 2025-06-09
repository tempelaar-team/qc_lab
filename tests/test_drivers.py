""" 
This module tests the serial and multiprocessing drivers for a few simple cases.
"""
import pytest

def test_drivers_spinboson():
    import numpy as np
    import matplotlib.pyplot as plt
    from qc_lab import Simulation, Data # import simulation class
    from qc_lab.models import SpinBoson # import model class
    from qc_lab.algorithms import MeanField # import algorithm class
    from qc_lab.dynamics import serial_driver, parallel_driver_multiprocessing # import dynamics driver
    import subprocess
    import sys
    import os

    sim = Simulation()
    sim.settings.num_trajs = 200
    sim.settings.batch_size = 50
    sim.settings.tmax = 10
    sim.settings.dt = 0.01

    sim.model = SpinBoson({
        'V':0.5,
        
        'E':0.5,
        'A':100,
        'W':0.1,
        'l_reorg':0.005,
        'boson_mass':1.0,
        'kBT':1.0,

    })
    sim.algorithm = MeanField()
    sim.model.initialize_constants()
    sim.state.wf_db= np.zeros((sim.model.constants.num_quantum_states), dtype=complex)
    sim.state.wf_db[0] += 1.0
    print('Running serial driver...')
    data_serial = serial_driver(sim)
    print('Running parallel multiprocessing driver...')
    data_parallel_multiprocessing = parallel_driver_multiprocessing(sim)
    cmd = ["mpirun", "-n", "4", sys.executable, "./tests/test_mpi.py"]
    print('Running parallel mpi driver...')
    subprocess.run(cmd)
    if not(os.path.exists("./tests/mpi_example.h5")):
        raise FileNotFoundError("MPI output file not found. Ensure the MPI driver ran successfully.")
    else:
        data_parallel_mpi = Data().load_from_h5("./tests/mpi_example.h5")
        os.remove("./tests/mpi_example.h5")
    print('Comparing results...')
    for key, val in data_serial.data_dict.items():
        if isinstance(val, np.ndarray):
            assert np.allclose(val, data_parallel_multiprocessing.data_dict[key])
            assert np.allclose(val, data_parallel_mpi.data_dict[key])
    print('parallel and serial resultsd match!')
    return

if __name__ == "__main__":
    test_drivers_spinboson()
