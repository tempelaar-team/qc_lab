.. _fssh-algorithm:
Fewest-Switches Surface Hopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `FewestSwitchesSurfaceHopping` class implements the fewest-switches surface hopping algorithm. This class is part of the `qclab` library and is used to simulate quantum systems using surface hopping methods.

Class Definition
----------------

.. autoclass:: qclab.algorithms.fewest_switches_surface_hopping.FewestSwitchesSurfaceHopping
    :members:
    :undoc-members:
    :show-inheritance:

Initialization
--------------

The `FewestSwitchesSurfaceHopping` class is initialized with a set of parameters. These parameters can be provided as a dictionary. If no parameters are provided, default parameters are used.

.. code-block:: python

    from qclab.algorithms import FewestSwitchesSurfaceHopping

    parameters = {
        'fssh_deterministic': False,
        'num_branches': 2,
        'gauge_fixing': 2,
        # ... other parameters ...
    }

    fssh = FewestSwitchesSurfaceHopping(parameters)

Recipes
-------

The `FewestSwitchesSurfaceHopping` class uses three main recipes for its operations:

1. **Initialization Recipe**: This recipe initializes the simulation state.
2. **Update Recipe**: This recipe updates the simulation state at each time step.
3. **Output Recipe**: This recipe computes the output variables from the simulation state.

Initialization Recipe
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    initialization_recipe = [
        lambda sim, state: tasks.initialize_z(sim=sim, state=state, seed=state.seed),
        lambda sim, state: tasks.broadcast_var_to_branch_vectorized(sim=sim, state=state, val=state.z,
                                                                    name='z_branch'),
        lambda sim, state: tasks.broadcast_var_to_branch_vectorized(sim=sim, state=state, val=state.wf_db,
                                                                    name='wf_db_branch'),
        lambda sim, state: tasks.update_h_quantum_vectorized(sim=sim, state=state, z=state.z_branch),
        lambda sim, state: tasks.diagonalize_matrix_vectorized(sim=sim, state=state, matrix=state.h_quantum,
                                                               eigvals_name='eigvals', eigvecs_name='eigvecs'),
        lambda sim, state: tasks.gauge_fix_eigs_vectorized(sim=sim, state=state, eigvals=state.eigvals,
                                                           eigvecs=state.eigvecs, eigvecs_previous=state.eigvecs,
                                                           output_eigvecs_name='eigvecs', z=state.z_branch,
                                                           gauge_fixing=2),
        lambda sim, state: tasks.copy_value_vectorized(sim=sim, state=state, val=state.eigvecs,
                                                       name='eigvecs_previous'),
        lambda sim, state: tasks.copy_value_vectorized(sim=sim, state=state, val=state.eigvals,
                                                       name='eigvals_previous'),
        lambda sim, state: tasks.basis_transform_vec_vectorized(sim=sim, state=state, input_vec=state.wf_db_branch,
                                                                basis=np.einsum('...ij->...ji', state.eigvecs).conj(),
                                                                output_name='wf_adb_branch'),
        lambda sim, state: tasks.initialize_random_values_fssh(sim=sim, state=state),
        lambda sim, state: tasks.initialize_active_surface(sim=sim, state=state),
        lambda sim, state: tasks.initialize_dm_adb_0_fssh_vectorized(sim=sim, state=state),
        lambda sim, state: tasks.update_act_surf_wf_vectorized(sim=sim, state=state),
        lambda sim, state: tasks.update_quantum_energy_fssh_vectorized(sim=sim, state=state),
        lambda sim, state: tasks.initialize_timestep_index(sim=sim, state=state),
    ]

Update Recipe
^^^^^^^^^^^^^

.. code-block:: python

    update_recipe = [
        lambda sim, state: tasks.copy_value_vectorized(sim=sim, state=state, val=state.eigvecs,
                                                       name='eigvecs_previous'),
        lambda sim, state: tasks.copy_value_vectorized(sim=sim, state=state, val=state.eigvals,
                                                       name='eigvals_previous'),
        lambda sim, state: tasks.update_z_rk4_vectorized(sim=sim, state=state, wf=state.act_surf_wf,
                                                               z=state.z_branch,
                                                               output_name='z_branch',
                                                               update_quantum_classical_forces_bool=False),
        lambda sim, state: tasks.update_wf_db_eigs_vectorized(sim=sim, state=state, wf_db=state.wf_db_branch,
                                                              eigvals=state.eigvals, eigvecs=state.eigvecs,
                                                              adb_name='wf_adb_branch', output_name='wf_db_branch'),
        lambda sim, state: tasks.update_h_quantum_vectorized(sim=sim, state=state, z=state.z_branch),
        lambda sim, state: tasks.diagonalize_matrix_vectorized(sim=sim, state=state, matrix=state.h_quantum,
                                                               eigvals_name='eigvals', eigvecs_name='eigvecs'),
        lambda sim, state: tasks.gauge_fix_eigs_vectorized(sim=sim, state=state, eigvals=state.eigvals,
                                                           eigvecs=state.eigvecs,
                                                           eigvecs_previous=state.eigvecs_previous,
                                                           output_eigvecs_name='eigvecs', z=state.z_branch,
                                                           gauge_fixing=sim.algorithm.parameters.gauge_fixing),
        lambda sim, state: tasks.update_active_surface_fssh(sim=sim, state=state),
        lambda sim, state: tasks.update_act_surf_wf_vectorized(sim=sim, state=state),
        lambda sim, state: tasks.update_timestep_index(sim=sim, state=state),
    ]

Output Recipe
^^^^^^^^^^^^^

.. code-block:: python

    output_recipe = [
        lambda sim, state: tasks.update_dm_db_fssh_vectorized(sim=sim, state=state),
        lambda sim, state: tasks.update_quantum_energy_fssh_vectorized(sim=sim, state=state),
        lambda sim, state: tasks.update_classical_energy_fssh_vectorized(sim=sim, state=state,
                                                                         z=state.z_branch),
    ]

Output Variables
----------------

The `FewestSwitchesSurfaceHopping` class computes the following output variables:

- `quantum_energy`: The quantum energy of the system.
- `classical_energy`: The classical energy of the system.
- `dm_db`: The density matrix database.

.. code-block:: python

    output_variables = [
        'quantum_energy',
        'classical_energy',
        'dm_db',
    ]