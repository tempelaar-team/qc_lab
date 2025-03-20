.. _defaults:

Default Model Attributes
------------------------

For minimal models where only the Hamiltonian of the system is defined in the Model class, QC Lab employs numerical methods to carry out 
particular steps in the dynamics algorithms. This page descripes those default actions and also the constants that can be used to manipulate them. 
Because they are formally treated as model ingredients they  have the same ingredient format discussed in the model development guide. 

All of the constants below can be set by adjusting their value in the `model.constants` object, for example:

.. code-block:: python

    sim.model.constants.default_value = # fix the value of the constant 'default_value'


Initialization of classical coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. function:: sim.model.init_classical(model, constants, parameters, seed=seed)

    :param seed: List of seeds used to initialize random numbers. 
    :type seed: np.ndarray((batch_size), dtype=int)
    :returns: Initial complex classical coordinate. 
    :rtype: np.ndarray((batch_size, sim.model.constants.num_classical_coordinates), dtype=complex)

By default, QC Lab uses a Markov-chain Monte Carlo implementation of the Metropolis-Hastings algorithm to sample a Boltzmann distribution corresponding to 
`sim.model.h_c` at the temperature `sim.model.constants.temp` assuming `kB=1`. We encourage caution and further validation before using it on arbitrary classical 
potentials as fine-tuning of the algorithm parameters may be required to obtain reliable results.

The implementation utilizes with a single random walker in `sim.model.constants.num_classical_coordinates` dimensions or `sim.model.constants.num_classical_coordinates` 
walkers each in one dimension (depending on if `mcmc_h_c_separable==True`) and evolves the walkers from the initial point `mcmc_init_z` by sampling a Gaussian distribution with
with standard deviation `mcmc_std` for `mcmc_burn_in_size` steps. It then evolves the walkers another `mcmc_sample_size` steps to collect a distribution of initial coordinates from which 
the required number of initial conditions are drawn uniformly. 


.. list-table::
   :header-rows: 1

   * - Variable name
     - Description
     - Default value
   * - `mcmc_burn_in_size`
     - Number of burn-in steps. 
     - 10000
   * - `mcmc_sample_size`
     - Number of samples to collect from which initial conditions are drawn. To ensure a full exploration of the phase-space this should be as large as practical.
     - 100000
   * - `mcmc_h_c_separable`
     - A boolean indicating if the classical Hamiltonian is separable into independent terms for each coordinate. If True each coordinate will be independently sampled improving the performance of the algorithm. If False the sampling will occur in the full dimensional space. 
     - True
   * - `mcmc_init_z`
     - The initial coordinate that the random walker is initialized at. 
     - A point in the interval (0,1) for both real and imaginary parts in each coordinate. (This is deterministically chosen for reproducability).
   * - `mcmc_std``
     - The standard deviation of the Gaussian used to generate the random walk.
     - 1


Classical Hamiltonian gradients 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. function:: sim.model.dh_c_dzc(model, constants, parameters, z = z)

    :param z: Complex classical coordinate. 
    :type z: np.ndarray((batch_size, sim.model.constants.num_classical_coordinates), dtype=complex)
    :returns: Gradient of the classical Hamiltonian. 
    :rtype: np.ndarray((batch_size, sim.model.constants.num_classical_coordinates), dtype=complex)


QC Lab utilizes a finite difference method to calculate the gradient of the classical Hamiltonian. 

.. list-table::
   :header-rows: 1

   * - Variable name
     - Description
     - Default value
   * - `dh_qc_dzc_finite_differences_delta`
     - finite difference that each coordinate is varied by.
     - 1e-6



Quantum-classical Hamiltonian gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. function:: sim.model.dh_c_dzc(model, constants, parameters, z = z)

    :param z: Complex classical coordinate. 
    :type z: np.ndarray((batch_size, sim.model.constants.num_classical_coordinates), dtype=complex)
    :returns: indices of nonzero values
    :rtype: np.ndarray((# of nonzero values, 4), dtype=int)
    :returns: values
    :rtype: np.ndarray((# of nonzero values), dtype=complex)
    :returns: shape of dense gradient: (batch_size, sim.model.constants.num_classical_coordinates, sim.model.constants.num_quantum_states, sim.model.constants.num_quantum_states)
    :rtype: tuple


QC Lab utilizes a finite difference method to calculate the gradient of the quantum-classical Hamiltonian. Unlike that of the 
classical Hamiltonian, however, the output is in a sparse format.

.. list-table::
   :header-rows: 1

   * - Variable name
     - Description
     - Default value
   * - `dh_qc_dzc_finite_differences_delta`
     - finite difference that each coordinate is varied by.
     - 1e-6


Surface Hopping Switching Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: sim.model.hop_function(model, constants, parameters, z=z, delta_z=delta_z, ev_diff=ev_diff)

    :param z: complex classical coordinate (in a single trajectory)
    :type z: np.ndarray(sim.model.constants.num_classical_coordinates, dtype=complex)
    :param delta_z: rescaling direction.
    :type delta_z: np.ndarray(sim.model.constants.num_classical_coordinates, dtype=complex)
    :param ev_diff: energy difference between final and initial surface (final - initial)
    :type ev_diff: float
    :returns: rescaled coordinate
    :rtype: np.ndarray(sim.model.constants.num_classical_coordinates, dtype=complex)
    :returns: True or False depending on if a hop happened.
    :rtype: Bool

QC Lab implements a numerical method to find the scalar factor (gamma) required to rescale classical coordinates in the surface hopping Algorithm. It works by constructing a uniform grid with 
`numerical_fssh_hop_num_points` points 
from negative to positive and determines the point at which energy is conserved the closest. It then recenters the 
grid at that point and reduces the range by 0.5 and once again searches for the point at which energy is conserved the closest. It repeats that step for `numerical_fssh_hop_max_iter`
iterations or until the energy difference is less than `numerical_fssh_hop_threshold`. If the energy it reaches is less than the threshold then the hop is 
accepted, if it is greater then the hop is rejected.

.. list-table::
   :header-rows: 1

   * - Variable name
     - Description
     - Default value
   * - `numerical_fssh_hop_gamma_range`
     - interval from minus to positive over which gamma is initially sampled.
     - 5
   * - `numerical_fssh_hop_num_points`
     - The number of points on the grid used to sample gamma. 
     - 10
   * - `numerical_fssh_hop_threshold`
     - The threshold used to determine if a hop is conserving energy at a given gamma.
     - 1e-6
   * - `numerical_fssh_hop_max_iter`
     - The maximum number of iterations before a search for gamma is halted. 
     - 20

