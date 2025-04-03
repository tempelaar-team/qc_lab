.. _parallel_driver_multiprocessing:

Parallel Multiprocessing Driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `parallel_driver_multiprocessing` function in the `qc_lab.dynamics` module is used to run simulations in parallel 
using the `multiprocessing` library in Python. This driver is compatible with Jupyter notebooks and is useful for 
calculations on a single node.

Function Signature
------------------

.. code-block:: python

    qc_lab.dynamics.parallel_driver_multiprocessing(sim, seeds=None, data=None, num_tasks=None)

Parameters
----------

- **sim** (*Simulation*): The simulation object that contains the model, settings, and state.
- **seeds** (*array-like, optional*): An array of seed values for the random number generator. If not provided, seeds will be generated automatically.
- **data** (*Data, optional*): A Data object to store the results of the simulation. If not provided, a new Data object will be created.
- **num_tasks** (*int, optional*): The number of parallel tasks (processes) to use for the simulation. If not provided, the number of available CPU cores will be used.

Returns
-------

- **data** (*Data*): A Data object containing the results of the simulation.

Example
-------

Here is an example of how to use the `parallel_driver_multiprocessing` function to run a simulation in parallel assuming
that the simulation object has been set up according to the quickstart guide.:

.. code-block:: python

    # Import the parallel driver
    from qc_lab.dynamics import parallel_driver_multiprocessing

    # Run the simulation using the parallel driver
    data = parallel_driver_multiprocessing(sim, num_tasks=4)

Notes
-----

- This driver is suitable for use in Jupyter notebooks and single-node calculations. 
- For cluster-based calculations, consider using the MPI driver.

References
----------

- `multiprocessing library <https://docs.python.org/3/library/multiprocessing.html>`_

