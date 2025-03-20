.. _spin_boson_model:

Spin-Boson Model
~~~~~~~~~~~~~~~~

We employ the same Hamiltonian and naming conventions as in `Tempelaar & Reichman 2019 <https://doi.org/10.1063/1.5000843>`_. 

The quantum-classical Hamiltonian of the spin-boson model is:

.. math::
    
    \hat{H}_{\mathrm{q}} = \left(\begin{array}{cc} E & V \\ V & -E \end{array}\right)

.. math::

    \hat{H}_{\mathrm{q-c}} = \sigma_{z} \sum_{\alpha}^{A}  g_{\alpha}q_{\alpha}

.. math::

    H_{\mathrm{c}} = \sum_{\alpha}^{A} \frac{p_{\alpha}^{2}}{2m} + \frac{1}{2}m\omega_{\alpha}^{2}q_{\alpha}^{2}

where :math:`\sigma_{z}` is the Pauli matrix, :math:`E` is the diagonal energy, :math:`V` is the off-diagonal coupling, and :math:`A` is the number of bosons.

The couplings and frequencies are sampled from a Debye spectral density:

.. math::

    \omega_{\alpha} = \Omega\tan\left(\frac{\alpha - 1/2}{2A}\pi\right)

.. math::

    g_{\alpha} = \omega_{\alpha}\sqrt{\frac{2\lambda}{A}}

Where :math:`\Omega` is the characteristic frequency and :math:`\lambda` is the reorganization energy. 

The classical coordinates are sampled from a Boltzmann distribution:

.. math::

    P(z) \propto \exp\left(-\frac{H_{\mathrm{c}}(\boldsymbol{z})}{T}\right)

and by convention we assume that :math:`\hbar = 1`, :math:`k_{B} = 1`.

Constants
----------

The following table lists all of the constants required by the `SpinBosonModel` class:

.. list-table:: SpinBosonModel constants
   :header-rows: 1

   * - Parameter (symbol)
     - Description
     - Default Value
   * - `temp` :math:`(T)`
     - Temperature
     - 1
   * - `V` :math:`(V)`
     - Off-diagonal coupling
     - 0.5
   * - `E` :math:`(E)`
     - Diagonal energy
     - 0.5
   * - `A` :math:`(A)`
     - Number of bosons
     - 100
   * - `W` :math:`(\Omega)`
     - Characteristic frequency
     - 0.1
   * - `l_reorg` :math:`(\lambda)`
     - Reorganization energy
     - 0.005
   * - `boson_mass` :math:`(m)`
     - Mass of the bosons
     - 1


Example
-------

::

    from qc_lab.models import SpinBoson
    from qc_lab import Simulation
    from qc_lab.algorithms import MeanField
    from qc_lab.dynamics import serial_driver
    import numpy as np

    # instantiate a simulation
    sim = Simulation()

    # instantiate a model 
    sim.model = SpinBoson()

    # instantiate an algorithm 
    sim.algorithm = MeanField()

    # define an initial diabatic wavefunction 
    sim.state.wf_db = np.array([1, 0], dtype=complex)

    # run the simulation
    data = serial_driver(sim)