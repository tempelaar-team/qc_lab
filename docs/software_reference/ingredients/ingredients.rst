.. _ingredients:

===========
Ingredients
===========

Ingredients are methods associated with model classes. A generic ingredient has the form:

.. code-block:: python

    def ingredient_name(model, parameters, **kwargs):
        # Calculate var.
        var = None
        return var


For consistency we include all of the arguments even if the ingredient does not use them.
An ingredient that generates the quantum Hamiltonian for a two-level system might look like this:

.. code-block:: python

    def h_q_two_level(model, parameters, **kwargs):
        """
        Quantum Hamiltonian for a two-level system.

        Required constants:
            - two_level_00: Energy of the first level.
            - two_level_11: Energy of the second level.
            - two_level_01_re: Real part of the coupling between levels.
            - two_level_01_im: Imaginary part of the coupling between levels.

        Keyword Arguments:
            - batch_size: (Optional) Number of batches for vectorized computation.
        """
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = len(parameters.seed)
        h_q = np.zeros((batch_size, 2, 2), dtype=complex)
        h_q[:, 0, 0] = model.constants.two_level_00
        h_q[:, 1, 1] = model.constants.two_level_11
        h_q[:, 0, 1] = model.constants.two_level_sysmtem_c + 1j * model.constants.two_level_01_im
        h_q[:, 1, 0] = model.constants.two_level_01_re - 1j * model.constants.two_level_01_im
        return h_q


When incorporated directly into the model class (for instance when writing a model class from scratch) one should replace `model` with `self`. See the Model Development section of the User Guide for a detailed example.


Implementations available in QC Lab
===================================

The following ingredients are implemented in QC Lab and can be accessed through the `qc_lab.ingredients` module.



Quantum Hamiltonian (`h_q`)
---------------------------------

.. automodule:: qc_lab.ingredients
    :members: h_q_two_level, h_q_nearest_neighbor


Quantum-Classical Hamiltonian (`h_qc`)
--------------------------------------------

.. automodule:: qc_lab.ingredients
    :members: h_qc_diagonal_linear

Classical Hamiltonian (`h_c`)
-----------------------------------

.. automodule:: qc_lab.ingredients
    :members: h_c_harmonic, h_c_free

Classical Initialization (`init_classical`)
-------------------------------------------------

.. automodule:: qc_lab.ingredients
    :members: init_classical_boltzmann_harmonic, init_classical_wigner_harmonic, init_classical_wigner_coherent_state, init_classical_definite_position_momentum


Quantum-Classical Gradients (`dh_qc_dzc`)
-------------------------------------------------

.. automodule:: qc_lab.ingredients
    :members: dh_qc_dzc_diagonal_linear


Classical Gradients (`dh_c_dzc`)
---------------------------------

.. automodule:: qc_lab.ingredients
    :members: dh_c_dzc_harmonic, dh_c_dzc_free

FSSH Hop Function (`hop`)
--------------------------------------

.. automodule:: qc_lab.ingredients
    :members: hop_harmonic, hop_free

FSSH Rescaling Direction (`rescaling_direction_fssh`)
-------------------------------------------------

.. automodule:: qc_lab.ingredients
    :members: harmonic_oscillator_rescaling_direction_fssh