.. _fmo_model:

Fenna-Matthews-Olson Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Fenna-Matthews-Olson (FMO) complex is a pigment-protein complex found in green sulfur bacteria. We implement it in QC Lab as an 
8 site model with Holstein-type coupling to local vibrational modes with identical couplings as in `Tempelaar 2014 <https://doi.org/10.1021/jp510074q>`_. 
The quantum Hamiltonian matrix elements are taken from `Busch 2010 <https://doi.org/10.1021/jz101541b>`_.


.. math::
    
    \hat{H}_{\mathrm{q}} = \begin{pmatrix}
        12505.0 & 94.8 & 5.5 & -5.9 & 7.1 & -15.1 & -12.2 & 39.5 \\
        94.8 & 12425.0 & 29.8 & 7.6 & 1.6 & 13.1 & 5.7 & 7.9 \\
        5.5 & 29.8 & 12195.0 & -58.9 & -1.2 & -9.3 & 3.4 & 1.4 \\
        -5.9 & 7.6 & -58.9 & 12375.0 & -64.1 & -17.4 & -62.3 & -1.6 \\
        7.1 & 1.6 & -1.2 & -64.1 & 12600.0 & 89.5 & -4.6 & 4.4 \\
        -15.1 & 13.1 & -9.3 & -17.4 & 89.5 & 12515.0 & 35.1 & -9.1 \\
        -12.2 & 5.7 & 3.4 & -62.3 & -4.6 & 35.1 & 12465.0 & -11.1 \\
        39.5 & 7.9 & 1.4 & -1.6 & 4.4 & -9.1 & -11.1 & 12700.0
    \end{pmatrix}

where the matrix elements above are in units of wavenumbers. Note that the values below are in units of kBT at 298.15K, internally QC Lab 
also implements the quantum Hamiltonian in these units.

.. math::

    \hat{H}_{\mathrm{q-c}} = \sum_{i}\omega\lambda \frac{1}{\sqrt{2m\omega}}c^{\dagger}_{i}c_{i}q_{j}

.. math::

    H_{\mathrm{c}} = \sum_{i}^{N} \frac{p_{i}^{2}}{2m} + \frac{1}{2}m\omega^{2}q_{i}^{2}



Here, :math:`\lambda` is the square-root of the Huang-Rhys factor, :math:`\omega` is the vibrational frequency, and :math:`m` is the vibrational mass. 

The classical coordinates are sampled from a Boltzmann distribution:

.. math::

    P(q,p) \propto \exp\left(-\frac{H_{\mathrm{c}}}{T}\right)

and by convention we assume that :math:`\hbar = 1`, :math:`k_{B} = 1`.

Constants
----------

The following table lists all of the constants required by the `HolsteinLatticeModel` class:

.. list-table:: HolsteinLatticeModel constants
   :header-rows: 1

   * - Parameter (symbol)
     - Description
     - Default Value
   * - `temp` :math:`(T)`
     - Temperature
     - 1
   * - `lambda` :math:`(\lambda)`
     - Square-root of the Huang-Rhys factor
     - :math:`\sqrt{0.05}`
   * - `w` :math:`(\omega)`
     - Vibrational frequency
     - 117 :math:`\mathrm{cm}^{-1}`
   * - `mass` :math:`(m)`
     - Vibrational mass
     - 1

     
Example
-------

::

    from qc_lab.models import FMOComplex
    from qc_lab import Simulation
    from qc_lab.algorithms import MeanField
    from qc_lab.dynamics import serial_driver
    import numpy as np

    # instantiate a simulation
    sim = Simulation()

    # instantiate a model 
    sim.model = FMOComplex()

    # instantiate an algorithm 
    sim.algorithm = MeanField()

    # define an initial diabatic wavefunction 
    wf_db_0 = np.zeros((sim.model.constants.N), dtype=np.complex128)
    wf_db_0[5] = 1.0 + 0.0j
    sim.state.wf_db = wf_db_0

    # run the simulation
    data = serial_driver(sim)