"""
This file defines the Vector class.
"""

import warnings
import numpy as np


class Vector:
    """
    The Vector object defines a vehicle in which quantities can be placed, retrieved, and collected.
    """

    def __init__(self):
        self.output_dict = {}
        self.seed = None

    def __getattr__(self, name):
        """
        If an attribute is not in the VectorObject it returns None rather than throwing an error.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        return None

    def collect_output_variables(self, output_variables):
        """
        Collect output variables for the state.

        Args:
            output_variables: List of output variable names.
        """
        for var in output_variables:
            self.output_dict[var] = getattr(self, var)

    def __getstate__(self):
        """Called when pickling the object."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def set_inplace(self, name, val):
        """
        Set an attribute in the Vector object in place.
        """
        setattr(self, name, val)
        return self


def initialize_vector_objects(sim, batch_seeds):
    """
    This takes the numpy arrays inside the state and parameter objects
    and creates a first index corresponding to the number of trajectories
    """
    state_vector = Vector()
    state_vector.seed = batch_seeds
    for name in sim.state.__dict__.keys():
        obj = getattr(sim.state, name)
        if isinstance(obj, np.ndarray) and name[0] != "_":
            init_shape = np.shape(obj)
            new_obj = (
                np.zeros((len(batch_seeds), *init_shape), dtype=obj.dtype)
                + obj[np.newaxis]
            )
            setattr(state_vector, name, new_obj)
    parameter_vector = Vector()
    for name in sim.model.parameters.__dict__.keys():
        obj = getattr(sim.model.parameters, name)
        if isinstance(obj, np.ndarray) and name[0] != "_":
            init_shape = np.shape(obj)
            new_obj = (
                np.zeros((len(batch_seeds), *init_shape), dtype=obj.dtype)
                + obj[np.newaxis]
            )
            setattr(parameter_vector, name, new_obj)
    return parameter_vector, state_vector
