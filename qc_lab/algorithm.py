"""
This module contains the Algorithm class, which is the base class for Algorithm objects in QC Lab.
"""

from qc_lab.constants import Constants
import copy


class Algorithm:
    """
    Base class for algorithms in QC Lab.
    """

    def __init__(self, default_settings=None, settings=None):
        if settings is None:
            settings = {}
        if default_settings is None:
            default_settings = {}
        settings = {**default_settings, **settings}
        self.settings = Constants(self.update_algorithm_settings)
        for key, val in settings.items():
            setattr(self.settings, key, val)
        self.settings._init_complete = True
        self.update_algorithm_settings()
        # copy the recipes and output variables to ensure they are not shared across instances
        self.initialization_recipe = copy.copy(self.initialization_recipe)
        self.update_recipe = copy.copy(self.update_recipe)
        self.output_recipe = copy.copy(self.output_recipe)
        self.output_variables = copy.copy(self.output_variables)

    def update_algorithm_settings(self):
        """
        Update algorithm settings. This method should be overridden by subclasses.
        """

    initialization_recipe = []
    update_recipe = []
    output_recipe = []
    output_variables = []

    def execute_initialization_recipe(self, sim, parameter, state):
        """
        Executes the initialization recipe for the given simulation.
        """
        for _, func in enumerate(self.initialization_recipe):
            parameter, state = func(self, sim, parameter, state)
        return parameter, state

    def execute_update_recipe(self, sim, parameter, state):
        """
        Executes the update recipe for the given simulation.
        """
        for _, func in enumerate(self.update_recipe):
            parameter, state = func(self, sim, parameter, state)
        return parameter, state

    def execute_output_recipe(self, sim, parameter, state):
        """
        Executes the output recipe for the given simulation.
        """
        for _, func in enumerate(self.output_recipe):
            parameter, state = func(self, sim, parameter, state)
        return parameter, state
