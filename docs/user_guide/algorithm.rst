.. _algorithm:

==========================
Algorithms
==========================

Algorithms in QC Lab define the sequence of operations that evolve the system defined by the model object (see :ref:`Models <model>`) in time. They are composed of three recipes which define initialization steps, update steps, and collect steps that together define the desired algorithm. Each recipe is a list of "tasks" (see :ref:`Tasks <task>`) which are executed in the order specified by the recipe list. Algorithms define the transient quantities of an algorithm in the state object, which is an instance of the Variable class (see :ref:`Variable Objects <variable_objects>`).


.. _algorithm_objects:
Algorithm Objects
-----------------------

Algorithm objects in QC Lab are instances of the ``qclab.Algorithm`` class. Each algorithm object is composed of three recipes: an initialization recipe ``algorithm.initialization_recipe``, an update recipe ``algorithm.update_recipe``, and a collect recipe ``algorithm.collect_recipe``. Like a model object, an algorithm object has an instance of the Constants class ``algorithm.settings`` which contains the settings specific to the algorithm. Unlike the model object, algorithm objects do not have internal constants and so there is no initialization method as there is for model objects (see :ref:`Models <model>`). Instead, the settings of the algorithm object are set directly by the user during or after instantiation of the algorithm object.

The empty Algorithm class is:


.. code-block:: python


    class Algorithm:
        """
        Algorithm class for defining and executing algorithm recipes.
        """

        def __init__(self, default_settings=None, settings=None):
            if settings is None:
                settings = {}
            if default_settings is None:
                default_settings = {}
            # Merge default settings with user-provided settings.
            settings = {**default_settings, **settings}
            # Construct a Constants object to hold settings.
            self.settings = Constants()
            # Put settings from the dictionary into the Constants object.
            for key, val in settings.items():
                setattr(self.settings, key, val)
            # Copy the recipes and output variables to ensure they are not shared
            # across instances.
            self.initialization_recipe = copy.deepcopy(self.initialization_recipe)
            self.update_recipe = copy.deepcopy(self.update_recipe)
            self.collect_recipe = copy.deepcopy(self.collect_recipe)

        initialization_recipe = []
        update_recipe = []
        collect_recipe = []

        def execute_recipe(self, sim, state, parameters, recipe):
            """
            Carry out the given recipe for the simulation by running
            each task in the recipe.
            """
            for func in recipe:
                state, parameters = func(sim, state, parameters)
            return state, parameters


After instantiating an algorithm object, the user can populate its recipes by assigning tasks to each recipe. For example, the mean-field algorithm can be defined from an empty Algorithm object as:

.. code-block:: python

    from qclab import Algorithm
    import qclab.tasks as tasks
    from functools import partial

    # Create an empty algorithm object.
    algorithm = Algorithm()
    # Populate the initialization recipe.
    algorithm.initialization_recipe = [
            tasks.initialize_variable_objects,
            tasks.initialize_norm_factor,
            tasks.initialize_z,
            tasks.update_h_quantum,
    ]
    # Populate the update recipe.
    algorithm.update_recipe = [
        # Begin RK4 integration steps.
        # RK4 steps excluded for brevity.
        # End RK4 integration steps.
        tasks.update_wf_db_rk4,
        tasks.update_h_q
    ]
    # Populate the collect recipe.
    algorithm.collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_mf,
        tasks.update_quantum_energy,
        tasks.update_classical_energy,
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_classical_energy,
        tasks.collect_quantum_energy,
    ]


Each recipe is executed by the method ``algorithm.execute_recipe``. The initialization recipe is executed once at the beginning of the simulation, the update recipe is executed at each time step of the simulation, and the collect recipe is executed once at the end of the simulation to gather and process results.


.. _variable_objects:
Variable Objects
--------------------------

The tasks that make up an algorithm operate on the attributes of the ``state`` and ``parameters`` objects that are passed to each task. These objects are instances of the ``qclab.Variable`` class. The variable class is a simple container for attributes that are created and updated by tasks during the simulation. Importantly, when accessing an attribute that does not exist, a variable class returns ``None`` instead of raising an error. This allows tasks to check for the existence of attributes in the variable objects and create them if they do not exist.

Additionally, the contents of a variable class can be passed to it when instantiating it:

.. code-block:: python

    from qclab import Variable

    # Create a variable object with some initial attributes.
    state = Variable({"wf_db": None, "z": None, "p": None})

but can also be created empty. 

By default, each variable object has an attribute ``output_dict`` which is an empty dictionary. This dictionary is used to gather results during the collect recipe of the algorithm. At the end of the simulation, the contents of this dictionary are returned as the output of the simulation. For example, if during the collect recipe a task adds an entry to the output dictionary as:

.. code-block:: python

    state.output_dict['dm_db'] = state.dm_db

then the output of the simulation will contain an entry with key ``'dm_db'`` and value equal to the contents of the attribute ``state.dm_db`` averaged over its batch dimension.

Importantly, the ``parameters`` object is not used for the generation of outputs in QC Lab so its ``output_dict`` attribute is not used. The ``parameters`` object is instead used to interface with the ingredients of a model (see :ref:`Models <model>` and :ref:`Ingredients <ingredient>`).


Mean Field Example
-------------------------------

As an example of a complete algorithm we include the source code for the mean-field algorithm below. This algorithm is defined in the ``qclab.algorithms.MeanField`` module and uses tasks from the ``qclab.tasks`` module to populate its recipes.

.. dropdown:: View full source
   :icon: code

   .. literalinclude:: ../../src/qclab/algorithms/mean_field.py
      :language: python
      :linenos:

.. _fssh_source:
Surface Hopping Example
-------------------------------

As an additional example of a complete algorithm we include the source code for the fewest-switches surface hopping algorithm below. This algorithm is defined in the ``qclab.algorithms.FewestSwitchesSurfaceHopping`` module and uses tasks from the ``qclab.tasks`` module to populate its recipes.

.. dropdown:: View full source
   :icon: code

   .. literalinclude:: ../../src/qclab/algorithms/fewest_switches_surface_hopping.py
      :language: python
      :linenos: