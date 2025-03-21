.. _ingredients:

Ingredients
-----------

Ingredients are are methods associated with model classes. A generic ingredient has the form:

.. code-block:: python

    def ingredient_name(model, constants, parameters, **kwargs):
        return var


For consistency we include all of the arguments even if the ingredient does not use them.
An ingredient that generates the quantum-classical Hamiltonian for the spin-boson model might look like this:

.. code-block:: python

    def spin_boson_h_qc(model, constants, parameters, **kwargs):
        z = kwargs.get("z")
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
            assert len(z) == batch_size
        else:
            batch_size = len(z)
        g = constants.spin_boson_coupling
        m = constants.classical_coordinate_mass
        h = constants.classical_coordinate_weight
        h_qc = np.zeros((batch_size, 2, 2), dtype=complex)
        h_qc[:, 0, 0] = np.sum(
            g * np.sqrt(1 / (2 * m * h))[np.newaxis, :] * (z + np.conj(z)), axis=-1
        )
        h_qc[:, 1, 1] = -h_qc[:, 0, 0]
        return h_qc


When incorporated directly into the model class one should replace `model` with `self`. See the :ref:`model_class` section 
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
    :members: diagonal_linear_coupling_h_qc, diagonal_linear_coupling_dh_qc_dzc

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

