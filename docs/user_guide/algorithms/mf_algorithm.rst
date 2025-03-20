.. _mf-algorithm:
Mean-Field Dynamics 
~~~~~~~~~~~~~~~~~~~

The `MeanField` class implements the mean-field dynamics algorithm. This class is part of the `qclab` library and is used to simulate quantum systems using mean-field theory.

Class Definition
----------------

.. autoclass:: qclab.algorithms.mean_field.MeanField
    :members:
    :undoc-members:
    :show-inheritance:

Initialization
--------------

The `MeanField` class is initialized with a set of parameters. These parameters can be provided as a dictionary. If no parameters are provided, default parameters are used.

.. code-block:: python

    from qclab.algorithms import MeanField

    parameters = {
        'param1': value1,
        'param2': value2,
        # ... other parameters ...
    }

    mean_field = MeanField(parameters)

Recipes
-------

The `MeanField` class uses three main recipes for its operations:

1. **Initialization Recipe**: This recipe initializes the simulation state.
2. **Update Recipe**: This recipe updates the simulation state at each time step.
3. **Output Recipe**: This recipe computes the output variables from the simulation state.

Initialization Recipe
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    initialization_recipe = [
        lambda sim, state: tasks.initialize_z(sim=sim, state=state, seed=state.seed),
        lambda sim, state: tasks.update_h_quantum_vectorized(sim=sim, state=state, z=state.z),
    ]

Update Recipe
^^^^^^^^^^^^^

.. code-block:: python

    update_recipe = [
        lambda sim, state: tasks.update_h_quantum_vectorized(sim=sim, state=state, z=state.z),
        lambda sim, state: tasks.update_z_rk4_vectorized(sim=sim, state=state, z=state.z,
                                                               output_name='z', wf=state.wf_db,
                                                               update_quantum_classical_forces_bool=False),
        lambda sim, state: tasks.update_wf_db_rk4_vectorized(sim=sim, state=state),
    ]

Output Recipe
^^^^^^^^^^^^^

.. code-block:: python

    output_recipe = [
        lambda sim, state: tasks.update_dm_db_mf_vectorized(sim=sim, state=state),
        lambda sim, state: tasks.update_quantum_energy_mf_vectorized(sim=sim, state=state, wf=state.wf_db),
        lambda sim, state: tasks.update_classical_energy_vectorized(sim=sim, state=state, z=state.z),
    ]

Output Variables
----------------

The `MeanField` class computes the following output variables:

- `dm_db`: The density matrix database.
- `classical_energy`: The classical energy of the system.
- `quantum_energy`: The quantum energy of the system.

.. code-block:: python

    output_variables = [
        'dm_db',
        'classical_energy',
        'quantum_energy',
    ]
