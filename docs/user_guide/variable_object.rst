.. _variable_object:


Variable Object
-----------

The variable object in QC Lab is used to store varibles in a simulation. It is created internally by the dynamics drivers calling:

.. code-block:: python

    from qc_lab.variable import initialize_variable_objects
    
    state, parameters = initialize_variable_objects(sim, seeds)


which results in two variable objects: `parameters` and `state`. The `sim` argument is the simulation object, and the `seeds` argument
 is a list of seeds for each trajectory. Any quantities initially in `sim.state` are now stored in the `state` variable object with a new 
 index that defines the trajectory it is associated with. By default the `parameters` variable object is empty, but it can be populated with 
 time-dependent parameters that are needed in the simulation by the algorithm.


