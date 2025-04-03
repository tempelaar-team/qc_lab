.. _holstein_model:

Holstein Lattice Model
~~~~~~~~~~~~~~~~~~~~~~

The Holstein Lattice Model is a nearest-neighbor tight-binding model combined with an idealized optical phonon that interacts via a 
Holstein coupling. The current implementation accommodates a single electronic particle and is described in detail in `Krotz et al. 2021 <https://doi.org/10.1063/5.0053177>`_
. 

The quantum Hamiltonian of the Holstein model is a nearest-neighbor tight-binding model

.. math::
    
    \hat{H}_{\mathrm{q}} = -j\sum_{\langle i,j\rangle}^{N}\hat{c}^{\dagger}_{i}\hat{c}_{j}

where :math:`\langle i,j\rangle` denotes nearest-neighbor sites with or without periodic boundaries determined by the parameter `periodic_boundary=True`.

The quantum-classical Hamiltonian is the Holstein coupling with dimensionless electron-phonon coupling :math:`g` and phonon frequency :math:`\omega`

.. math::

    \hat{H}_{\mathrm{q-c}} = g\sqrt{2m\omega^{3}}\sum_{i}^{N} \hat{c}^{\dagger}_{i}\hat{c}_{i} q_{i}

and the classical Hamiltonian is the harmonic oscillator

.. math::

    H_{\mathrm{c}} = \sum_{i}^{N} \frac{p_{i}^{2}}{2m} + \frac{1}{2}m\omega^{2}q_{i}^{2}

with mass :math:`m`.

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
   * - `g` :math:`(g)`
     - Dimensionless electron-phonon coupling
     - 0.5
   * - `w` :math:`(\omega)`
     - Phonon frequency
     - 0.5
   * - `N` :math:`(N)`
     - Number of sites
     - 10
   * - `j` :math:`(j)`
     - Hopping energy
     - 1
   * - `phonon_mass` :math:`(m)`
     - Phonon mass
     - 1
   * - `periodic_boundary`
     - Periodic boundary condition
     - `True``

     
Example
-------

::

    from qc_lab.models import HolsteinLattice
    from qc_lab import Simulation
    from qc_lab.algorithms import MeanField
    from qc_lab.dynamics import serial_driver
    import numpy as np

    # instantiate a simulation
    sim = Simulation()

    # instantiate a model 
    sim.model = HolsteinLattice()

    # instantiate an algorithm 
    sim.algorithm = MeanField()

    # define an initial diabatic wavefunction 
    wf_db_0 = np.zeros((sim.model.constants.num_quantum_states), dtype=np.complex128)
    wf_db_0[0] = 1.0 + 0.0j
    sim.state.wf_db = wf_db_0

    # run the simulation
    data = serial_driver(sim)