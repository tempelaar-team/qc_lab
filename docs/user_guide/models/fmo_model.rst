.. _fmo_model:

Fenna-Matthews-Olson Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Fenna-Matthews-Olson (FMO) complex is a pigment-protein complex found in green sulfur bacteria. We implement it in QC Lab as an 
7 site model with Holstein-type coupling to local vibrational modes with couplings and frequencies sampled from a Debye spectral 
density according to `Mulvihill et. al 2021 <https://doi.org/10.1063/5.0051101>`_. 


.. math::
    
    \hat{H}_{\mathrm{q}} = \begin{pmatrix}
        12410 & -87.7 & 5.5 & -5.9 & 6.7 & -13.7 & -9.9 \\
        -87.7 & 12530 & 30.8 & 8.2 & 0.7 & 11.8 & 4.3 \\
        5.5 & 30.8 & 12210.0 & -53.5 & -2.2 & -9.6 & 6.0 \\
        -5.9 & 8.2 & -53.5 & 12320 & -70.7 & -17.0 & -63.3 \\
        6.7 & 0.7 & -2.2 & -70.7 & 12480 & 81.1 & -1.3 \\
        -13.7 & 11.8 & -9.6 & -17.0 & 81.1 & 12630 & 39.7 \\
        -9.9 & 4.3 & 6.0 & -63.3 & -1.3 & 39.7 & 12440
    \end{pmatrix}

where the matrix elements above are in units of wavenumbers. Note that the values below are in units of thermal quantum at 298.15K.

.. math::

    \hat{H}_{\mathrm{q-c}} = \sum_{i}\sum_{j}^{A}\omega_{j}g_{j}c^{\dagger}_{i}c_{i}q_{ij}

.. math::

    H_{\mathrm{c}} = \sum_{i}\sum_{j}^{A} \frac{p_{ij}^{2}}{2m} + \frac{1}{2}m\omega_{j}^{2}q_{ij}^{2}


The couplings and frequencies are sampled from a Debye spectral density:

.. math::

    \omega_{j} = \Omega\tan\left(\frac{j - 1/2}{2A}\pi\right)

.. math::

    g_{j} = \omega_{j}\sqrt{\frac{2\lambda}{A}}

Where :math:`\Omega` is the characteristic frequency and :math:`\lambda` is the reorganization energy. 

The classical coordinates are sampled from a Boltzmann distribution:

.. math::

    P(\boldsymbol{p},\boldsymbol{q}) \propto \exp\left(-\frac{H_{\mathrm{c}}(\boldsymbol{p},\boldsymbol{q})}{k_{\mathrm{B}}T}\right)

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
   * - `mass` :math:`(m)`
     - Vibrational mass
     - 1
   * - `A` :math:`(A)`
     - Number of bosons
     - 200
   * - `W` :math:`(\Omega)`
     - Characteristic frequency
     - 106.14 :math:`\mathrm{cm}^{-1}`
   * - `l_reorg` :math:`(\lambda)`
     - Reorganization energy
     - 35 :math:`\mathrm{cm}^{-1}`

     
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
    wf_db_0 = np.zeros((sim.model.constants.num_quantum_states), dtype=np.complex128)
    wf_db_0[5] = 1.0 + 0.0j
    sim.state.wf_db = wf_db_0

    # run the simulation
    data = serial_driver(sim)