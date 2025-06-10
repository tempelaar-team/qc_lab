"""
This file contains the Simulation, State, and Data classes. It also contains additional
functions for initializing and handling these objects.
"""

import numpy as np
from qc_lab.constants import Constants
from qc_lab.vector import Vector


class Simulation:
    """
    The simulation object represents the entire simulation process.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {
            "tmax": 10,
            "dt": 0.001,
            "dt_output": 0.1,
            "num_trajs": 100,
            "batch_size": 25,
        }
        settings = {**self.default_settings, **settings}
        self.settings = Constants()
        for key, val in settings.items():
            setattr(self.settings, key, val)
        self.algorithm = None
        self.model = None
        self.state = Vector()

    def initialize_timesteps(self):
        """
        Initialize the timesteps for the simulation based on the parameters.

        First adjusts tmax to be the closest integer multiple of dt.
        Then adjusts dt_output to be the closest integer multiple of dt as well.
        Then adjusts tmax to be the closest point on the grid defined by dt_output.
        """
        tmax = self.settings.get("tmax", self.default_settings.get("tmax"))
        dt = self.settings.get("dt", self.default_settings.get("dt"))
        dt_output = self.settings.get(
            "dt_output", self.default_settings.get("dt_output")
        )

        tmax_n = np.round(tmax / dt).astype(int)
        dt_output_n = np.round(dt_output / dt).astype(int)
        tmax_n = np.round(tmax_n / dt_output_n).astype(int) * dt_output_n

        self.settings.tmax_n = tmax_n
        self.settings.dt_output_n = dt_output_n
        self.settings.tdat = np.arange(0, self.settings.tmax_n + 1, 1) * dt
        self.settings.tdat_n = np.arange(0, self.settings.tmax_n + 1, 1)
        self.settings.tdat_output = (
            np.arange(
                0,
                self.settings.tmax_n + self.settings.dt_output_n,
                self.settings.dt_output_n,
            )
            * dt
        )
        self.settings.tdat_output_n = np.arange(
            0,
            self.settings.tmax_n + self.settings.dt_output_n,
            self.settings.dt_output_n,
        )
