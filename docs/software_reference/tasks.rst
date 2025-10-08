.. _tasks::

Tasks
=====

Tasks in QC Lab are the building blocks of algorithms. They define well-characterized operations that manipulate quantities in the variable objects.

The generic form of a task when used in an algorithm is:

.. code-block:: python

    def task_name(sim, state, parameters):
        # Perform operations on model and parameters
        return state, parameters

where we include `algorithm` as the first argument because the tasks are always bound methods of the algorithm class. 

In order to make tasks more general, QC Lab defines them with additional keyword arguments which can be used to customize their behavior. In such cases,
the tasks are redefined within each algorithm so that they automatically pass the necessary keyword arguments but still conform to the above structure. 

An example of this is the task that updates the quantum Hamiltonian which accepts the `z` keyword argument to specify the classical coordinate at 
which the Hamiltonian is evaluated:

.. code-block:: python

    def update_h_quantum(sim, state, parameters, **kwargs):
        """
        Update the quantum + quantum-classical Hamiltonian.

        Required constants:
            None.
        """
        z = kwargs.get("z", state.z)
        h_q, _ = sim.model.get("h_q")
        h_qc, _ = sim.model.get("h_qc")
        state.h_quantum = h_q(sim.model, parameters) + h_qc(sim.model, parameters, z=z)
    return state, parameters

In order to be used each algorithm class defines an internal methods that passes the necessary keyword arguments to the task:

.. code-block:: python

    def _update_h_quantum(self, sim, state, parameters):
        return tasks.update_h_quantum(self, sim, state, parameters, z=state.z)



The tasks in QC Lab are defined in the `qc_lab.tasks` module. They are organized into three groups, the initialization tasks which are used at the outset of the simulation
to define necessary quantities, the update tasks which are used to update the quantities at each update timestep during the simulation, and the collect tasks that collect the quantities which the user 
wants to save in the data object at each collect timestep. 


Initialization Tasks 
----------------------


.. automodule:: qc_lab.initialization_tasks
    :members:
    :undoc-members:
    :show-inheritance:


Update Tasks
----------------------

.. automodule:: qc_lab.tasks.update_tasks
    :members:
    :undoc-members:
    :show-inheritance:


Collect Tasks
----------------------

.. automodule:: qc_lab.tasks.collect_tasks
    :members:
    :undoc-members:
    :show-inheritance: