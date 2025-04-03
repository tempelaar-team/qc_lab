.. _drivers:

Dynamics Drivers
----------------

Drivers in QC Lab are functions that interface a Simulation object with the Dynamics core. Drivers are responsible for initializing the 
objects needed for the Dynamics core to operate and handle the assignment of random seeds and the grouping of simulations into batches.


.. toctree::
    :maxdepth: 1

    serial_driver
    parallel_driver_multiprocessing
    parallel_driver_mpi

