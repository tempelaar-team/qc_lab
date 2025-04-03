.. _ingredients:

Ingredients
-----------

Ingredients are methods associated with model classes. A generic ingredient has the form:

.. code-block:: python

    def ingredient_name(model, constants, parameters, **kwargs):
        # Calculate var.
        var = None
        return var


For consistency we include all of the arguments even if the ingredient does not use them.
An ingredient that generates the quantum Hamiltonian for a two-level system might look like this:

.. code-block:: python

    def two_level_system_h_q(model, constants, parameters, **kwargs):
        """
        Quantum Hamiltonian for a two-level system.

        Required Constants:
            - two_level_system_a: Energy of the first level.
            - two_level_system_b: Energy of the second level.
            - two_level_system_c: Real part of the coupling between levels.
            - two_level_system_d: Imaginary part of the coupling between levels.

        Keyword Arguments:
            - batch_size: (Optional) Number of batches for vectorized computation.
        """
        del model
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = len(parameters.seed)
        h_q = np.zeros((batch_size, 2, 2), dtype=complex)
        h_q[:, 0, 0] = constants.two_level_system_a
        h_q[:, 1, 1] = constants.two_level_system_b
        h_q[:, 0, 1] = constants.two_level_system_c + 1j * constants.two_level_system_d
        h_q[:, 1, 0] = constants.two_level_system_c - 1j * constants.two_level_system_d
        return h_q


When incorporated directly into the model class (for instance when writing a model class from scratch) one should replace `model` with `self`. See the Model Development section of the User Guide
for a detailed example.

Below we list all of the ingredients available in the current version of QC Lab and group ingredients by the attribute of the model that they pertain to. 


Quantum Hamiltonian
^^^^^^^^^^^^^^^^^^^


.. automodule:: qc_lab.ingredients
    :members: two_level_system_h_q, nearest_neighbor_lattice_h_q


Quantum-Classical Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ingredients that generate quantum-classical interaction terms.

.. automodule:: qc_lab.ingredients
    :members: diagonal_linear_h_qc, diagonal_linear_dh_qc_dzc

Classical Hamiltonian
^^^^^^^^^^^^^^^^^^^^^

Ingredients that generate classical Hamiltonians. 

.. automodule:: qc_lab.ingredients
    :members: harmonic_oscillator_h_c, harmonic_oscillator_dh_c_dzc, harmonic_oscillator_hop 

Classical Initialization 
^^^^^^^^^^^^^^^^^^^^^^^^

Ingredients that initialize the complex-valued classical coordinates.

.. automodule:: qc_lab.ingredients
    :members: harmonic_oscillator_boltzmann_init_classical, harmonic_oscillator_wigner_init_classical

