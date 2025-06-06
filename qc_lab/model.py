"""
This file contains the Model class, which is the base class for Model objects in QC Lab.
"""

from qc_lab.constants import Constants
from qc_lab.vector import Vector
import copy


class Model:
    """
    Base class for models in the simulation framework.
    """

    def __init__(self, default_constants=None, constants=None):
        if constants is None:
            constants = {}
        internal_defaults = {}
        constants = {**internal_defaults, **default_constants, **constants}
        self.constants = Constants(self.initialize_constants)
        for key, val in constants.items():
            setattr(self.constants, key, val)
        self.constants._init_complete = True
        self.parameters = Vector()
        self.initialization_functions = copy.deepcopy(self.initialization_functions)
        self.ingredients = copy.deepcopy(self.ingredients)

    def initialize_constants(self):
        """
        Initialize the constants for the model and ingredients.
        """
        for func in self.initialization_functions:
            # Here, self is needed because the initialization functions are
            # defined in the subclass.
            func(self)

    initialization_functions = []
    ingredients = []
