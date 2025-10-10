"""
This module contains the settings necessary to reproduce the reference data.
"""


from qclab.numerical_constants import INVCM_TO_300K

model_sim_settings = {
    "SpinBoson": {
        "num_trajs": 100,
        "batch_size": 100,
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "HolsteinLattice": {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "HolsteinLatticeReciprocalSpace": {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "FMOComplex": {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "TullyProblemOne": {
        "num_trajs": 100,
        "batch_size": 100,
        "tmax": 5000,
        "dt_update": 0.5,
        "dt_collect": 10,
        "progress_bar": False,
    },
    "TullyProblemTwo": {
        "num_trajs": 100,
        "batch_size": 100,
        "tmax": 5000,
        "dt_update": 0.5,
        "dt_collect": 10,
        "progress_bar": False,
    },
    "TullyProblemThree": {
        "num_trajs": 100,
        "batch_size": 100,
        "tmax": 5000,
        "dt_update": 0.5,
        "dt_collect": 10,
        "progress_bar": False,
    },
}


model_settings = {
    "SpinBoson": {
        "kBT": 1.0,
        "V": 0.5,
        "E": 0.5,
        "A": 100,
        "W": 0.1,
        "l_reorg": 0.005,
        "boson_mass": 1.0,
    },
    "HolsteinLattice": {
        "kBT": 1.0,
        "g": 0.5,
        "w": 0.5,
        "N": 10,
        "J": 1.0,
        "phonon_mass": 1.0,
        "periodic": True,
    },
    "HolsteinLatticeReciprocalSpace": {
        "kBT": 1.0,
        "g": 0.5,
        "w": 0.5,
        "N": 10,
        "J": 1.0,
        "phonon_mass": 1.0,
        "periodic": True,
    },
    "FMOComplex": {
        "kBT": 1.0,
        "mass": 1.0,
        "l_reorg": 35.0 * INVCM_TO_300K,
        "w_c": 106.14 * INVCM_TO_300K,
        "N": 200,
    },
    "TullyProblemOne": {
        "init_momentum": 10.0,
        "init_position": -5.0,
        "mass": 2000.0,
        "A": 0.01,
        "B": 1.6,
        "C": 0.005,
        "D": 1.0,
    },
    "TullyProblemTwo": {
        "init_momentum": 10.0,
        "init_position": -5.0,
        "mass": 2000.0,
        "A": 0.1,
        "B": 0.28,
        "C": 0.015,
        "D": 0.06,
        "E_0": 0.05,
    },
    "TullyProblemThree": {
        "init_momentum": 10.0,
        "init_position": -5.0,
        "mass": 2000.0,
        "A": 0.0006,
        "B": 0.1,
        "C": 0.9,
    },
}