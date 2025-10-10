"""
This module contains tests of the data object methods.
"""

import pytest


def test_save_load_h5py():
    import numpy as np
    from qclab import Simulation, Data
    from qclab.models import SpinBoson
    from qclab.algorithms import MeanField
    from qclab.dynamics import serial_driver
    import os
    try:
        import h5py as _
    except ImportError:
        pytest.skip("h5py not available, skipping test.", allow_module_level=True)

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
    sim.initial_state.wf_db = np.zeros(
        (sim.model.constants.num_quantum_states), dtype=complex
    )
    sim.initial_state.wf_db[0] += 1.0
    data_serial = serial_driver(sim)
    data_serial.save("test_data.h5")
    loaded_data = Data().load("test_data.h5")
    print("Comparing results...")
    for key, val in data_serial.data_dict.items():
        if isinstance(val, np.ndarray):
            assert np.allclose(val, loaded_data.data_dict[key])
    print("calculated and loaded results match!")
    os.remove("test_data.h5")
    return

def test_save_load_no_h5py():
    import numpy as np
    from qclab import Simulation, Data
    from qclab.models import SpinBoson
    from qclab.algorithms import MeanField
    from qclab.dynamics import serial_driver
    import os

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
    sim.initial_state.wf_db = np.zeros(
        (sim.model.constants.num_quantum_states), dtype=complex
    )
    sim.initial_state.wf_db[0] += 1.0
    data_serial = serial_driver(sim)
    data_serial.save("test_data.npz", disable_h5py=True)
    loaded_data = Data().load("test_data.npz", disable_h5py=True)
    print("Comparing results...")
    for key, val in data_serial.data_dict.items():
        if isinstance(val, np.ndarray):
            assert np.allclose(val, loaded_data.data_dict[key])
    print("calculated and loaded results match!")
    os.remove("test_data.npz")
    return

def test_load_sum():
    import numpy as np
    from qclab import Simulation, Data
    from qclab.models import SpinBoson
    from qclab.algorithms import MeanField
    from qclab.dynamics import serial_driver
    import os
    try:
        import h5py as _
    except ImportError:
        pytest.skip("h5py not available, skipping test.", allow_module_level=True)

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
    sim.initial_state.wf_db = np.zeros(
        (sim.model.constants.num_quantum_states), dtype=complex
    )
    sim.initial_state.wf_db[0] += 1.0
    data_serial = serial_driver(sim)
    data_serial.save("test_data.h5")
    new_data = Data().load("test_data.h5")
    new_data = new_data.load("test_data.h5")
    assert np.all(new_data.data_dict["seed"] == np.concatenate((data_serial.data_dict["seed"], data_serial.data_dict["seed"])))
    print("Comparing results...")
    for key, val in data_serial.data_dict.items():
        if isinstance(val, np.ndarray):
            if key == "seed":
                continue
            assert np.allclose(val, new_data.data_dict[key])
    print("calculated and loaded results match!")
    os.remove("test_data.h5")


if __name__ == "__main__":
    test_save_load_h5py()
    test_save_load_no_h5py()
    test_load_sum()