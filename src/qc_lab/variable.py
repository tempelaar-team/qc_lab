"""
This module contains the Variable class.
"""

import logging

logger = logging.getLogger(__name__)


class Variable:
    """
    Variable class for storing time-dependent variables.
    """

    def __init__(self, input_dict=None):
        self.output_dict = {}
        if input_dict is not None:
            for key, val in input_dict.items():
                setattr(self, key, val)

    def __getattr__(self, name):
        """
        ``Variable.get`` method. Returns ``None`` for missing attributes.
        This is to avoid errors when checking for the presence of an attribute.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        return None

    def __getstate__(self):
        """
        Support pickling of the object. Important for parallelization.
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Restore state during unpickling. Important for parallelization.
        """
        self.__dict__.update(state)
