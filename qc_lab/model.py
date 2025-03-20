"""
This file contains the Model class, which is the base class for Model objects in QC Lab.
"""

from qc_lab.constants import Constants
from qc_lab.vector import Vector


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
        self.initialize_constants()
        self.parameters = Vector()

    def initialize_constants(self):
        """
        Initialize the constants for the model and ingredients.
        """
        for func in self.initialization_functions:
            # Here, self is needed because the initialization functions are
            # defined in the subclass.
            func(self)

    def h_q(self, constants, parameters, **kwargs):
        """
        Quantum Hamiltonian function. This method should be overridden by subclasses.
        """
        del constants, parameters, kwargs

    def h_qc(self, constants, parameters, **kwargs):
        """
        Quantum-classical Hamiltonian function. This method should be overridden by subclasses.
        """
        del constants, parameters, kwargs

    def h_c(self, constants, parameters, **kwargs):
        """
        Classical Hamiltonian function. This method should be overridden by subclasses.
        """
        del constants, parameters, kwargs

    def initialize_constants_h_c(self):
        """
        Initialize the constants for the classical Hamiltonian.
        """

    def initialize_constants_h_qc(self):
        """
        Initialize the constants for the quantum-classical Hamiltonian.
        """

    def initialize_constants_h_q(self):
        """
        Initialize the constants for the quantum Hamiltonian.
        """

    def initialize_constants_model(self):
        """
        Initialize the constants for the model.
        """

    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_c,
        initialize_constants_h_qc,
        initialize_constants_h_q,
    ]
