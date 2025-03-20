.. _serial_driver:

Serial Driver
~~~~~~~~~~~~~

The `serial_driver` function in the `qclab.dynamics` module is used to run simulations in serial.


Function Signature
------------------

.. code-block:: python

    qclab.dynamics.serial_driver(sim, seeds=None, data=None, num_tasks=None)

Parameters
----------

- **sim** (*Simulation*): The simulation object that contains the model, settings, and state.
- **seeds** (*array-like, optional*): An array of seed values for the random number generator. If not provided, seeds will be generated automatically.
- **data** (*Data, optional*): A Data object to store the results of the simulation. If not provided, a new Data object will be created.

Returns
-------

- **data** (*Data*): A Data object containing the results of the simulation.

Example
-------

Here is an example of how to use the `serial_driver` function to run a simulation assuming
that the simulation object has been set up according to the quickstart guide.:

.. code-block:: python

    # Import the serial driver
    from qc_lab.dynamics import serial_driver

    # Run the simulation using the parallel driver
    data = serial_driver(sim)

Notes
-----

- The total number of trajectories must be an integer multiple of the batch size. If not,
    the driver will use the lower integer multiple (which could be zero!).



