.. _algorithm_dev:

Algorithm Development
=====================

In this guide, we will discuss how to make in-place modifications to Algorithms. Algorithm development is a 
more advanced topic and requires a good understanding of the underlying steps of the algorithm so we will not
go into detail (in this guide) about how to develop a new algorithm from scratch. Instead, we will focus 
on making changes to existing algorithms.

Before we proceed, let's discuss the structure of an algorithm in QC Lab. An algorithm in QC Lab is a Python
class that inherits from the `Algorithm` class in the `qc_lab.algorithm` module. The built-in algorithms are
found in the `qc_lab.algorithms` module and presently contain `qc_lab.algorithms.MeanField` and
`qc_lab.algorithms.FewestSwitchesSurfaceHopping`.

We can start by importing the `MeanField` algorithm from the `qc_lab.algorithms` module:

.. code-block:: python

    from qc_lab.algorithms import MeanField


Each algorithm consists of three lists of functions which are referred to as "recipes", the functions themselves are 
referred to as "tasks".  

.. code-block:: python

    # the initialization recipe
    print(MeanField.initialization_recipe)

    # the update recipe
    print(MeanField.update_recipe)

    # the output recipe
    print(MeanField.output_recipe)

As the name implies, the initialization_recipe initializes all the variables required for the algorithm. The update_recipe
updates the variables at each time step and the output_recipe is used to output the results of the algorithm.

In addition to the recipes, a list of variable names is needed to specify which variables the algorithm will store in the Data object. 

.. code-block:: python

    # the variables that the algorithm will store
    print(MeanField.output_variables)


Adding output obvservables
---------------------------

To add an additional variable to the output of an algorithm we must define a task that calculates the variable and add it to the output recipe.

Linear response functions
~~~~~~~~~~~~~~~~~~~~~~~~~

For example, let's calculate a linear response function that can be used to calculate the absorption spectrum of a system. Mathematically we will calculate

.. math::

    R(t) = \langle \psi(0) \vert \psi(t)\rangle

where :math:`\vert \psi(t)\rangle` is the diabatic wavefunction at time :math:`t`.

.. code-block:: python

    def update_response_function(sim, parameters, state, **kwargs):
        # First get the diabatic wavefunction.
        wf_db = state.wf_db
        # If we are at the first timestep we can store the diabatic wavefunction in the parameters object
        if sim.t_ind == 0:
            parameters.wf_db_0 = np.copy(wf_db)
        # Next calculate the response function and store it in the state object.
        state.response_function = np.sum(np.conj(parameters.wf_db_0) * wf_db, axis=-1)
        return parameters, state



Next we can add this task to the output recipe.

.. code-block:: python

    MeanField.output_recipe.append(update_response_function)

Finally we can add the relevant variable name to the output_variables list.

.. code-block:: python

    MeanField.output_variables.append('response_function')


We can then run a simulation and calculate the corresponding spectral function,


.. code-block:: python
    
    from qc_lab import Simulation 
    from qc_lab.dynamics import parallel_driver_multiprocessing
    from qc_lab.models import SpinBoson

    # instantiate a simulation
    sim = Simulation()
    print('default simulation settings: ', sim.default_settings)

    # change settings to customize simulation
    sim.settings.num_trajs = 1000
    sim.settings.batch_size = 250
    sim.settings.tmax = 50
    sim.settings.dt = 0.01

    # instantiate a model 
    sim.model = SpinBoson({'l_reorg': 0.2})
    print('default model constants: ', sim.model.default_constants) # print out default constants

    # instantiate an algorithm 
    sim.algorithm = MeanField()
    print('default algorithm settings: ', sim.algorithm.default_settings) # print out default settings



    # define an initial diabatic wavefunction 
    sim.state.wf_db = np.array([1, 0], dtype=complex)

    # run the simulation
    data = parallel_driver_multiprocessing(sim, num_tasks=4)

    # plot the data.
    print('calculated quantities:', data.data_dic.keys())
    num_trajs = len(data.data_dic['seed'])
    response_function = data.data_dic['response_function']/num_trajs
    time = sim.settings.tdat_output
    plt.plot(time, np.real(response_function), label='R(t)')
    plt.xlabel('time')
    plt.ylabel('response function')
    plt.legend()
    plt.show()

    plt.plot(np.real(np.roll(np.fft.fft(response_function), len(time)//2)))
    plt.xlabel('freq')
    plt.ylabel('Absorbrance')
    plt.show()


Adiabatic populations
~~~~~~~~~~~~~~~~~~~~~

Next, let's calculate the adiabatic populations of the system as is sometimes done in scattering problems. Obviously these populations 
will only have a well-defined meaning in regimes with no nonadiabatic coupling.

.. code-block:: python

    def update_adiabatic_populations(sim, parameters, state, **kwargs):
        # First get the Hamiltonian and calculate its eigenvalues and eigenvectors.
        H = state.h_quantum # this is the quantum plus quantum-classical Hamiltonian.
        # Next obtain its eigenvalues and eigenvectors.
        evals, evecs = np.linalg.eigh(H)
        # Now calculate the adiabatic wavefunction.
        wf_adb = np.einsum('tji,tj->ti', np.conj(evecs), state.wf_db)
        # Finally calculate the populations (note that we do not sum over the batch).
        pops_adb = np.abs(wf_adb)**2
        # Store the populations in the state object.
        state.pops_adb = pops_adb
        return parameters, state

Next we can add this task to the output recipe.

.. code-block:: python

    MeanField.output_recipe.append(update_adiabatic_populations)

Finally we can add the relevant variable name to the output_variables list.

.. code-block:: python

    MeanField.output_variables.append('pops_adb')

We can then run a simulation and plot the populations, note that since the Spin-Boson model is always in a coupling regime these
 populations will not have a well-defined meaning.

.. code-block:: python

    from qc_lab import Simulation 
    from qc_lab.dynamics import serial_driver
    from qc_lab.models import SpinBoson

    # instantiate a simulation
    sim = Simulation()
    print('default simulation settings: ', sim.default_settings)

    # change settings to customize simulation
    sim.settings.num_trajs = 100
    sim.settings.batch_size = 100
    sim.settings.tmax = 25
    sim.settings.dt = 0.01

    # instantiate a model 
    sim.model = SpinBoson()
    print('default model constants: ', sim.model.default_constants) # print out default constants

    # instantiate an algorithm 
    sim.algorithm = MeanField()
    print('default algorithm settings: ', sim.algorithm.default_settings) # print out default settings



    # define an initial diabatic wavefunction 
    sim.state.wf_db = np.array([1, 0], dtype=complex)

    # run the simulation
    data = serial_driver(sim)

    # plot the data.
    print('calculated quantities:', data.data_dic.keys())
    num_trajs = len(data.data_dic['seed'])
    classical_energy = data.data_dic['classical_energy']/num_trajs
    quantum_energy = data.data_dic['quantum_energy']/num_trajs
    populations = np.real(np.einsum('tii->ti', data.data_dic['dm_db']/num_trajs))
    adiabatic_populations = np.real(data.data_dic['pops_adb']/num_trajs)
    time = sim.settings.tdat_output
    plt.plot(time, adiabatic_populations[:,0], label='adiabatic state 0')
    plt.plot(time, adiabatic_populations[:,1], label='adiabatic state 1')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.legend()
    #plt.savefig('../docs/user_guide/quickstart/quickstart_populations.png')
    plt.show()

.. note::

    In the above code we chose to modify the MeanField class itself rather than an instance of it. This can lead to troublesome 
    behavior in a Jupyter notebook where the class will not be reloaded if the cell is rerun. Restarting the kernel 
    will fix this issue. Otherwise one can modify an instance of the class by creating a new instance and modifying it.

Modifying algorithm behavior
----------------------------

In the same way that we could modify the output recipe, it is possible to modify the initialization and update recipes in the same way. 
We will not go into detail on how to do this here but the process is the same as for the output recipe (except there is no output variable in those cases).