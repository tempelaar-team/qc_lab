"""
This module contains the Model class.
"""

import copy
from qclab.constants import Constants


class Model:
    """
    Model class for defining model constants and ingredients.
    """

    def __init__(self, default_constants=None, constants=None):
        if constants is None:
            constants = {}
        if default_constants is None:
            default_constants = {}
        # Merge default constants with user-provided constants.
        constants = {**default_constants, **constants}
        # Construct a Constants object to hold constants.
        # Provide an initialization function to reinitialize
        # internal model constants upon changing any constant.
        self.constants = Constants(self.initialize_constants)
        # Put constants from the dictionary into the Constants object.
        for key, val in constants.items():
            setattr(self.constants, key, val)
        # Mark the constants as initialized.
        self.constants._init_complete = True
        # Copy the ingredients to ensure they are not shared across instances.
        self.ingredients = copy.deepcopy(self.ingredients)
        # Flags to indicate if the quantum Hamiltonian and quantum-classical
        # gradients need to be updated.
        self.update_h_q = True
        self.update_dh_qc_dzc = True
        self.initialize_constants()

    def get(self, ingredient_name):
        """
        Retrieve an ingredient by name.
        If the ingredient is not found or is None, returns (None, False).
        If the ingredient is found and not None, returns (ingredient, True).

        Args
        -----------
        ingredient_name : str
            Name of the ingredient to search for.

        Returns
        -----------
        tuple[callable | None, bool]: The ingredient function (or None if
            not found) and a flag indicating whether it exists.
        """
        for ingredient in self.ingredients[::-1]:
            if ingredient[0] == ingredient_name and not (ingredient[1] is None):
                return ingredient[1], True
            if ingredient[0] == ingredient_name and (ingredient[1] is None):
                return None, False
        return None, False

    def initialize_constants(self):
        """
        Initialize the constants for the model and ingredients.
        Calls ingredients with names starting with "_init_".
        """
        for ingredient in self.ingredients[::-1]:
            if ingredient[0].startswith("_init_") and ingredient[1] is not None:
                ingredient[1](self, None)
        return

    ingredients = []
