.. _fssh-algorithm:

Fewest-Switches Surface Hopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `qc_lab.algorithms.FewestSwitchesSurfaceHopping` class implements Tully's Fewest-Switches Surface Hopping (FSSH) dynamics algorithm according to `Hammes-Schiffer 1994 <https://doi.org/10.1063/1.467455>`_.


Settings
--------


.. list-table:: `FewestSwitchesSurfaceHopping` settings
   :widths: 30 80 20
   :header-rows: 1

   * - Setting name (type)
     - Description
     - Default value
   * - `fssh_deterministic (bool)`
     - If `True` the algorithm uses a deterministic representation of the initial state by propagating all possible initial active surfaces. If `False`, it samples the initial active surface according to the adiabatic populations.
     - `False`
   * - `gauge_fixing (string)`
     - The type of gauge fixing to employ on the eigenvectors at each timestep. ("sign_overlap": adjust only the sign, "phase_overlap": adjust the sign and phase using the overlap with the previous timestep, "phase_der_couple": adjust the sign and phase by calculating the derivative couplings.)
     - 0

Initial State
-------------

The FSSH algorithm requires an initial diabatic wavefunction called `wf_db` which is a complex NumPy array with dimension `sim.model.constants.num_quantum_states`.
For example:


.. code-block:: python

    sim.state.wf_db = np.array([1, 0], dtype=complex)


Output Variables
----------------

The following table lists the default output variables for the `FewestSwitchesSurfaceHopping` class.

.. list-table:: `FewestSwitchesSurfaceHopping` Output Variables
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
    from qc_lab.algorithms import FewestSwitchesSurfaceHopping # import algorithm class 
    from qc_lab.dynamics import serial_driver # import dynamics driver

    sim = Simulation()
    sim.model = SpinBoson()
    sim.algorithm = FewestSwitchesSurfaceHopping()
    sim.state.wf_db= np.array([1, 0], dtype=complex)
    data = serial_driver(sim)



