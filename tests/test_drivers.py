""" 
This module tests the serial and multiprocessing drivers for a few simple cases.
"""
import pytest

def test_drivers_spinboson():
    import numpy as np
    import matplotlib.pyplot as plt
    from qc_lab import Simulation # import simulation class
    from qc_lab.models import SpinBoson # import model class
    from qc_lab.algorithms import MeanField # import algorithm class
    from qc_lab.dynamics import serial_driver, parallel_driver_multiprocessing # import dynamics driver

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

    sim.state.wf_db= np.zeros((sim.model.constants.num_quantum_states), dtype=complex)
    sim.state.wf_db[0] += 1.0

    data_serial = serial_driver(sim)
    data_parallel = parallel_driver_multiprocessing(sim)
    for key, val in data_serial.data_dict.items():
        if isinstance(val, np.ndarray):
            assert np.allclose(val, data_parallel.data_dict[key])
    return
