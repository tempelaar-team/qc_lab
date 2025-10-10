"""
This module tests the serial and multiprocessing drivers for a few simple cases.
"""

import pytest


def test_drivers_spinboson():
    import numpy as np
    from qclab import Simulation  # import simulation class
    from qclab.models import SpinBoson  # import model class
    from qclab.algorithms import MeanField  # import algorithm class
    from qclab.dynamics import (
        serial_driver,
        parallel_driver_multiprocessing,
    )  # import dynamics driver

    sim = Simulation()
    sim.settings.progress_bar = False
    sim.settings.num_trajs = 200
    sim.settings.batch_size = 50
    sim.settings.tmax = 10
    sim.settings.dt_update = 0.01

    sim.model = SpinBoson(
        {
            "V": 0.5,
            "E": 0.5,
            "A": 100,
            "W": 0.1,
            "l_reorg": 0.005,
            "boson_mass": 1.0,
            "kBT": 1.0,
        }
    )
    sim.algorithm = MeanField()
    sim.model.initialize_constants()
    sim.initial_state.wf_db = np.zeros(
        (sim.model.constants.num_quantum_states), dtype=complex
    )
    sim.initial_state.wf_db[0] += 1.0
    print("Running serial driver...")
    data_serial = serial_driver(sim)
    print("Running parallel multiprocessing driver...")
    data_parallel_multiprocessing = parallel_driver_multiprocessing(sim)
    print("Comparing results...")
    for key, val in data_serial.data_dict.items():
        if isinstance(val, np.ndarray):
            assert np.allclose(val, data_parallel_multiprocessing.data_dict[key])
    print("parallel and serial results match!")
    return


@pytest.mark.mpi
def test_drivers_spinboson_mpi():
    """
    This test runs the serial and parallel drivers for the SpinBoson model using MPI. It
    compares the results to ensure they match.

    This test requires MPI to be set up and run with a command like:
    mpirun -n 4 pytest -m mpi -s tests/test_drivers.py
    """
    import numpy as np
    from qclab import Simulation  # import simulation class
    from qclab.models import SpinBoson  # import model class
    from qclab.algorithms import MeanField  # import algorithm class
    from mpi4py import MPI  # import MPI for parallel processing
    from qclab.dynamics import (
        serial_driver,
        parallel_driver_mpi,
    )  # import dynamics driver

    sim = Simulation()
    sim.settings.progress_bar = False
    sim.settings.num_trajs = 200
    sim.settings.batch_size = 50
    sim.settings.tmax = 10
    sim.settings.dt_update = 0.01

    sim.model = SpinBoson(
        {
            "V": 0.5,
            "E": 0.5,
            "A": 100,
            "W": 0.1,
            "l_reorg": 0.005,
            "boson_mass": 1.0,
            "kBT": 1.0,
        }
    )
    sim.algorithm = MeanField()
    sim.model.initialize_constants()
    sim.initial_state.wf_db = np.zeros(
        (sim.model.constants.num_quantum_states), dtype=complex
    )
    sim.initial_state.wf_db[0] += 1.0

    data_parallel_mpi = parallel_driver_mpi(sim)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("finished running parallel driver on rank", rank)
    if rank == 0:
        print("Running serial driver... on rank", rank)
        data_serial = serial_driver(sim)
        print("Comparing results...")
        for key, val in data_serial.data_dict.items():
            if isinstance(val, np.ndarray):
                assert np.allclose(val, data_parallel_mpi.data_dict[key])
        print("parallel and serial results match!")
    return


def test_incommensurate_batch_size_serial():
    """
    This test checks that the drivers work correctly when the number of trajectories
    is not an integer multiple of the batch size.
    """
    import numpy as np
    from qclab import Simulation  # import simulation class
    from qclab.models import SpinBoson  # import model class
    from qclab.algorithms import MeanField  # import algorithm class
    from qclab.dynamics import serial_driver  # import dynamics driver

    sim = Simulation()
    sim.settings.progress_bar = False
    sim.settings.num_trajs = 200  # Not an integer multiple of batch_size
    sim.settings.batch_size = 50
    sim.settings.tmax = 10
    sim.settings.dt_update = 0.01

    sim.model = SpinBoson(
        {
            "V": 0.5,
            "E": 0.5,
            "A": 100,
            "W": 0.1,
            "l_reorg": 0.005,
            "boson_mass": 1.0,
            "kBT": 1.0,
        }
    )
    sim.algorithm = MeanField()
    sim.model.initialize_constants()
    sim.initial_state.wf_db = np.zeros(
        (sim.model.constants.num_quantum_states), dtype=complex
    )
    sim.initial_state.wf_db[0] += 1.0
    print("Running commensurate batch size serial driver...")
    data_serial = serial_driver(sim)
    sim.settings.batch_size = 30  # Change batch size to not divide num_trajs.
    print("Running incommensurate batch size serial driver...")
    data_serial_incomm = serial_driver(sim)
    print("Comparing results...")
    for key, val in data_serial.data_dict.items():
        if isinstance(val, np.ndarray):
            assert np.allclose(val, data_serial_incomm.data_dict[key])
            # assert np.allclose(val, data_parallel_mpi.data_dict[key])
    print("results match!")
    return


def test_incommensurate_batch_size_multiprocessing():
    """
    This test checks that the drivers work correctly when the number of trajectories
    is not an integer multiple of the batch size.
    """
    import numpy as np
    from qclab import Simulation  # import simulation class
    from qclab.models import SpinBoson  # import model class
    from qclab.algorithms import MeanField  # import algorithm class
    from qclab.dynamics import (
        parallel_driver_multiprocessing,
    )  # import dynamics driver

    sim = Simulation()
    sim.settings.progress_bar = False
    sim.settings.num_trajs = 200  # Not an integer multiple of batch_size
    sim.settings.batch_size = 50
    sim.settings.tmax = 10
    sim.settings.dt_update = 0.01

    sim.model = SpinBoson(
        {
            "V": 0.5,
            "E": 0.5,
            "A": 100,
            "W": 0.1,
            "l_reorg": 0.005,
            "boson_mass": 1.0,
            "kBT": 1.0,
        }
    )
    sim.algorithm = MeanField()
    sim.model.initialize_constants()
    sim.initial_state.wf_db = np.zeros(
        (sim.model.constants.num_quantum_states), dtype=complex
    )
    sim.initial_state.wf_db[0] += 1.0
    print("Running commensurate batch size parallel driver...")
    data_parallel = parallel_driver_multiprocessing(sim)
    sim.settings.batch_size = 30  # Change batch size to not divide num_trajs.
    print("Running incommensurate batch size parallel driver...")
    data_parallel_incomm = parallel_driver_multiprocessing(sim)
    print("Comparing results...")
    for key, val in data_parallel.data_dict.items():
        if isinstance(val, np.ndarray):
            assert np.allclose(val, data_parallel_incomm.data_dict[key])
            # assert np.allclose(val, data_parallel_mpi.data_dict[key])
    print("results match!")
    return


if __name__ == "__main__":
    test_drivers_spinboson()
    test_incommensurate_batch_size_serial()
    test_incommensurate_batch_size_multiprocessing()
