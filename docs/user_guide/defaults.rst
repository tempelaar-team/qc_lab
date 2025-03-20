.. _defaults:

Default Model Attributes
------------------------

For minimal models where only the Hamiltonian of the system is defined in the Model class, QC Lab employs numerical methods to carry out 
particular steps in the dynamics algorithms. This page descripes those default actions and also the constants that can be used to manipulate them. 
Because they are formally treated as model ingredients they  have the same ingredient format discussed in the model development guide. 


Initialization of classical coordinates 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    sim.model.init_classical(model, constants, parameters, seed=seed)

Here, `seed` is a numpy array of seeds (integers). 

By default, QC Lab uses a Markov-chain Monte Carlo implementation of the Metropolis-Hastings algorithm to sample a Boltzmann distribution corresponding to 
`sim.model.h_c` at the temperature `sim.model.constants.temp` assuming `kB=1`. We encourage caution and further validation before using it on arbitrary classical 
potentials as fine-tuning of the algorithm parameters may be required to obtain reliable results.

The implementation utilizes with a single random walker in `sim.model.constants.num_classical_coordinates` dimensions or `sim.model.constants.num_classical_coordinates` 
walkers each in one dimension (depending on if `mcmc_h_c_separable==True`) and evolves the walkers from the initial point `mcmc_init_z` by sampling a Gaussian distribution with
with standard deviation `mcmc_std` for `mcmc_burn_in_size` steps. It then evolves the walkers another `mcmc_sample_size` steps to collect a distribution of initial coordinates from which 
the required number of initial conditions are drawn uniformly. 


.. list-table:: `sim.model.init_classical`
   :header-rows: 1

   * - Variable name
     - Description
     - Default value
   * - `mcmc_burn_in_size`
     - Number of burn-in steps. 
     - 10000
   * - `mcmc_sample_size`
     - Number of samples to collect from which initial conditions are drawn. To ensure a full exploration of the phase-space this should be 
     as large as practical.
     - 100000
   * - `mcmc_h_c_separable`
     - A boolean indicating if the classical Hamiltonian is separable into independent terms for each coordinate. If True each coordinate will be 
     independently sampled improving the performance of the algorithm. If False the sampling will occur in the full dimensional space. 
     - True
   * - `mcmc_init_z`
     - The initial coordinate that the random walker is initialized at. 
     - A point in the interval (0,1) for both real and imaginary parts in each coordinate. (This is deterministically chosen for reproducability).
   * - `mcmc_std``
     - The standard deviation of the Gaussian used to generate the random walk.
     - 1


Classical Hamiltonian gradients 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Quantum-classical Hamiltonian gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




