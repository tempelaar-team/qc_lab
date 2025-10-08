.. _simulation:


The Simulation Object
========================
The simulation object is the vehicle for setting up a simulation in QC Lab. It hosts the model and algorithm objects which are sent to the 
dynamics driver to run the simulation. The simulation object also contains settings that can be modified to change the behavior of the simulation.

By default QC Lab uses the following settings in the simulation object. These settings can be adjusted by changing the values in the `sim` object.

.. code-block:: python

    sim = Simulation()
    sim.settings.var = val # Can change the value of a setting like this

    # or by passing the setting directly to the simulation object.
    sim = Simulation({'var': val})


.. list-table:: Default Simulation Settings
   :header-rows: 1

   * - Variable
     - Description
     - Default Value
   * - `num_trajs`
     - The total number of trajectories to run.
     - 10
   * - `batch_size`
     - The (maximum) number of trajectories to run simultaneously.
     - 1
   * - `tmax`
     - The total time of each trajectory.
     - 10
   * - `dt_update`
     - The timestep used for executing the update recipe (the dynamics propagation).
     - 0.01
   * - `dt_collect`
     - The timestep used for executing the output recipe (the calculation of observables).
     - 0.1

.. note::

    QC Lab will change `dt_collect` to be integer multiple of `dt_update` and ensure that `tmax` is an integer multiple of both `dt_update` and `dt_collect`.
    