.. _quickstart:

Quick Start Guide
-----------------


QC Lab is organized into models and algorithms which are combined into a simulation object. 
The simulation object fully defines a quantum-classical dynamics simulation which is then carried out by a dynamics driver. 
This guide will walk you through the process of setting up a simulation object and running a simulation.


Importing Modules
~~~~~~~~~~~~~~~~~

First, we import the necessary modules:

::

    import numpy as np
    import matplotlib.pyplot as plt
    from qc_lab import Simulation # import simulation class 
    from qc_lab.models import SpinBoson # import model class 
    from qc_lab.algorithms import MeanField # import algorithm class 
    from qc_lab.dynamics import serial_driver # import dynamics driver


Instantiating Simulation Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we instantiate a simulation object. Each object has a set of default parameters which can be accessed by calling `sim.default_parameters`.
Passing a dictionary to the simulation object when instantiating it will override the default parameters.

::

    sim = Simulation()
    print('default simulation parameters: ', sim.default_parameters)
    # default simulation parameters:  {'tmax': 10, 'dt': 0.01, 'dt_output': 0.1, 'num_trajs': 10, 'batch_size': 1}

Alternatively, you can directly modify the simulation parameters by assigning new values to the parameters attribute of the simulation object. Here we change the number
of trajectories that the simulation will run, and how many trajectories are run at a time (the batch size). We also change the total time of each trajectory (tmax) and the 
timestep used for propagation (dt). Importantly, QC Lab expects that `num_trajs` is an integer multiple of `batch_size`. If not, it will use the lower integer multiple (which could be zero!).

::

    # change parameters to customize simulation
    sim.parameters.num_trajs = 200
    sim.parameters.batch_size = 20 
    sim.parameters.tmax = 30
    sim.parameters.dt = 0.001

Instantiating Model Object
~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we instantiate a model object. Like the simulation object, it has a set of default parameters. 

::

    sim.model = SpinBoson()
    print('default model parameters: ', sim.model.default_parameters)
    # default model parameters:  {'temp': 1, 'V': 0.5, 'E': 0.5, 'A': 100, 'W': 0.1, 'l_reorg': 0.005, 'boson_mass': 1}

Instantiating Algorithm Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we instantiate an algorithm object. 

:: 
    
    sim.algorithm = MeanField()
    print('default algorithm parameters: ', sim.algorithm.default_parameters)
    # default algorithm parameters:  {}

Setting Initial State
~~~~~~~~~~~~~~~~~~~~~

Before using the dynamics driver to run the simulation, it is necessary to provide the simulation with an initial state. This initial state is
dependent on both the model and algorithm. For mean-field dynamics, we require a diabatic wavefunction called "wf_db". Because we are using a spin-boson model,
this wavefunction should have dimension 2. 

The initial state is stored in `sim.state` which can be accessed as follows,

::

    sim.state.wf_db= np.array([1, 0], dtype=complex)

Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~

Finally, we run the simulation using the dynamics driver. Here, we are using the serial driver. QC Lab comes with several different types of parallel drivers which are discussed elsewhere.

::

    data = serial_driver(sim)

Analyzing Results
~~~~~~~~~~~~~~~~~

The data object returned by the dynamics driver contains the results of the simulation in a dictionary with keys corresponding
to the names of the observables that were requested to be recorded during the simulation.

:: 

    print('calculated quantities:', data.data_dic.keys())
    # calculated quantities: dict_keys(['seed', 'dm_db', 'classical_energy', 'quantum_energy'])

Each of the calculated quantities must be normalized with respect to the number of trajectories. In mean-field dynamics this is equivalent 
to the number of seeds.

::
    
    num_trajs = len(data.data_dic['seed'])
    classical_energy = data.data_dic['classical_energy'] / num_trajs
    quantum_energy = data.data_dic['quantum_energy'] / num_trajs
    populations = np.real(np.einsum('tii->ti', data.data_dic['dm_db'] / num_trajs))

The time axis can be retrieved from the simulation object through its settings

::

    time = sim.settings.tdat_output 

Plotting Results
~~~~~~~~~~~~~~~~

Finally, we can plot the results of the simulation like the population dynamics:

::

    plt.plot(time, populations[:, 0], label='upper state')
    plt.plot(time, populations[:, 1], label='lower state')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.legend()
    plt.show()

.. image:: quickstart_populations.png
    :alt: Population dynamics.
    :align: center

We can verify that the total energy of the simulation was conserved by inspecting the change in energy of quantum and classical subsystems over time.

::

    plt.plot(time, classical_energy - classical_energy[0], label='classical energy')
    plt.plot(time, quantum_energy - quantum_energy[0], label='quantum energy')
    plt.plot(time, classical_energy + quantum_energy - classical_energy[0] - quantum_energy[0], label='total energy')
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.legend()
    plt.show()

.. image:: quickstart_energies.png
    :alt: Change in energy.
    :align: center

Changing the Algorithm
~~~~~~~~~~~~~~~~~~~~~~

If you want to do a surface hopping calculation rather than a mean-field one, QC Lab makes it very easy to do so. 
Simply import the relevant Algorithm class and set `sim.algorithm` to it and rerun the calculation: 


::

    from qc_lab.algorithms import FewestSwitchesSurfaceHopping

    sim.algorithm = FewestSwitchesSurfaceHopping()

    data = serial_driver(sim)