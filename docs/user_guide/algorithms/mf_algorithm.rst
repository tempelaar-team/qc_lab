.. _mf-algorithm:
Mean-Field Dynamics 
~~~~~~~~~~~~~~~~~~~

The `qclab.algorithms.MeanField` class implements the mean-field (Ehrenfest) dynamics algorithm according to `Tully 1998 <https://doi.org/10.1039/A801824C>`_.
For a more recent reference we suggest `Krotz et al. 2021 <https://doi.org/10.1063/5.0053177>`_.

Settings
--------

The mean-field algorithm has no default settings.

Initial State
-------------

The mean-field algorithm requires an initial diabatic wavefunction called `wf_db` which is a complex numpy array with dimension `sim.model.constants.num_quantum_states`.
For example:
.. code-block:: python

    sim.state.wf_db = np.array([1, 0], dtype=complex)


Output Variables
----------------

The following table lists the default output variables for the `MeanField` class.

.. list-table:: MeanField Output Variables
   :header-rows: 1

   * - Variable name
     - Description
   * - `classical_energy`
     - Energy in the classical subsystem
   * - `quantum_energy`
     - Energy in the quantum subsystem
   * - `dm_db`
     - Diabatic density matrix

Example
-------

The following example demonstrates how to run a mean-field simulation for a spin-boson model using all default settings.

.. code-block:: python

    import numpy as np
    from qc_lab import Simulation # import simulation class 
    from qc_lab.models import SpinBoson # import model class 
    from qc_lab.algorithms import MeanField # import algorithm class 
    from qc_lab.dynamics import serial_driver # import dynamics driver

    sim = Simulation()
    sim.model = SpinBoson()
    sim.algorithm = MeanField()
    sim.state.wf_db= np.array([1, 0], dtype=complex)
    data = serial_driver(sim)