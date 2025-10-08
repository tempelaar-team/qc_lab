"""
This module contains the Data class.
"""

import logging
import numpy as np
from qc_lab.utils import DISABLE_H5PY

logger = logging.getLogger(__name__)


class Data:
    """
    Data class for handling the collection of data during a simulation.
    """

    def __init__(self, seeds=None):
        if seeds is None:
            seeds = np.array([], dtype=int)
        self.data_dict = {"seed": seeds, "norm_factor": 0}
        # Store log messages captured during a simulation run. This attribute is
        # populated by the drivers when they return the Data object.
        self.log = ""

    def add_output_to_data_dict(self, sim, state, t_ind):
        """
        Add data to the output dictionary ``data_dict``.

        Args
        ----
        sim: Simulation
            The simulation object containing settings and parameters.
        state: Variable
            The state object containing the current simulation state.
        t_ind: int
            The current time index in the simulation.
        """
        # Check if the norm_factor is zero. If it is, save it from the state object.
        if self.data_dict["norm_factor"] == 0:
            if not hasattr(state, "norm_factor"):
                logger.critical(
                    "The state object does not have a norm_factor attribute. "
                    "This is required to normalize the data."
                )
                raise ValueError(
                    "The state object does not have a norm_factor attribute."
                )
            logger.info(
                "Setting norm_factor to %s from state object.", state.norm_factor
            )
            self.data_dict["norm_factor"] = state.norm_factor
        for key, val in state.output_dict.items():
            if not key in self.data_dict:
                # If the key is not in the data_dict, initialize it with zeros.
                self.data_dict[key] = np.zeros(
                    (len(sim.settings.t_collect), *np.shape(val)[1:]), dtype=val.dtype
                )
                logger.info(
                    "Initializing data_dict[%s] with shape %s.",
                    key,
                    self.data_dict[key].shape,
                )
            # Store the data in the data_dict at the correct time index.
            self.data_dict[key][t_ind // sim.settings.dt_collect_n] = (
                np.sum(val, axis=0) / self.data_dict["norm_factor"]
            )

    def add_data(self, new_data):
        """
        Add data from ``new_data`` to the output dictionary ``data_dict``.

        Args
        ----
        new_data: Data
            A Data instance containing the new data to merge.
        """
        new_norm_factor = (
            new_data.data_dict["norm_factor"] + self.data_dict["norm_factor"]
        )
        for key, val in new_data.data_dict.items():
            if key == "seed":
                self.data_dict[key] = np.concatenate(
                    (self.data_dict[key], val.flatten()), axis=0
                )
            elif key != "norm_factor":
                if key in self.data_dict:
                    self.data_dict[key] = (
                        self.data_dict[key] * self.data_dict["norm_factor"]
                        + val * new_data.data_dict["norm_factor"]
                    ) / new_norm_factor
                else:
                    self.data_dict[key] = val
        self.data_dict["norm_factor"] = new_norm_factor
        # Append any log messages stored in new_data to this instance's log.
        if getattr(new_data, "log", ""):
            self.log += new_data.log

    def save(self, filename):
        """
        Save the data to disk with file name ``filename``.

        If h5py is available the data is stored as an HDF5 archive; otherwise
        each variable is saved using numpy.savez.

        Args
        ----
        filename : str
            The file name to save the data to.
        """
        if DISABLE_H5PY:
            np.savez(filename, log=self.log, **self.data_dict)
        else:
            with h5py.File(filename, "w") as h5file:
                self._recursive_save(h5file, "/", self.data_dict)
                h5file.attrs["log"] = self.log

    def load(self, filename):
        """
        Load a Data object from ``filename``.

        Args
        ----
        filename : str
            The file name to load the data from.

        Returns
        -------
        Data : Data
            The loaded Data object.
        """
        if DISABLE_H5PY:
            loaded = np.load(filename)
            self.data_dict = {key: loaded[key] for key in loaded.files if key != "log"}
            self.log = str(loaded.get("log", ""))
            return self
        with h5py.File(filename, "r") as h5file:
            self._recursive_load(h5file, "/", self.data_dict)
            self.log = h5file.attrs["log"]
        return self

    def _recursive_save(self, h5file, path, dict):
        """
        Recursively save dictionary contents to an HDF5 group.

        Args
        ----
        h5file : h5py.File
            The HDF5 file object to save data into.
        path : str
            The current path in the HDF5 file.
        dict : dict
            The dictionary to save.
        """
        for key, item in dict.items():
            if isinstance(item, dict):
                self._recursive_save(h5file, path + key + "/", item)
            else:
                try:
                    h5file.create_dataset(path + key, data=np.asarray(item))
                except TypeError as exc:
                    raise ValueError(
                        f"Cannot save {key} with unsupported type {type(item)}."
                    ) from exc

    def _recursive_load(self, h5file, path, dict):
        """
        Recursively load dictionary contents from an HDF5 group.

        Args
        ----
        h5file : h5py.File
            The HDF5 file object to load data from.
        path : str
            The current path in the HDF5 file.
        dict : dict
            The dictionary to populate with loaded data.
        """
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                dict[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                dict[key] = {}
                self._recursive_load(h5file, path + key + "/", dict[key])
