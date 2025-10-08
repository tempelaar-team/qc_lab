.. tasks:

Tasks
-----

Tasks are methods of an Algorithm object that carry out elementary steps by modifying the attributes of
 the ``state`` and ``parameters`` objects. A generic task has a standardized set of inputs and outputs
 of the form: 


.. code-block:: python

   def example_task(sim, state, parameters, **kwargs):
       # Task modifies attributes of parameters and state.
       return state, parameters

A task that does not require us to pass any keyword arguments can be directly added to one of the recipes of
an Algorithm object by using built-in Python methods for list modification.

For example, adding the above example task to the MeanField algorithm can be done as follows:

.. code-block:: python

    from qc_lab.algorithms import MeanField

    myMF = MeanField()
    myMF.initialization_tasks.append(example_task)


Most tasks in QC Lab use keyword arguments to specify which attributes of the ``state`` and ``parameters``
objects they will modify. This allows the same task to be used in different algorithms or for different
purposes. For example, consider a task that updates a wavefunction attribute of the ``state`` object using
the 4th-order Runge-Kutta method. The task can be defined as follows:

.. code-block:: python

    def update_wf_rk4(sim, state, parameters, **kwargs):
        """
        Update the wavefunction using the 4th-order Runge-Kutta method.

        Required constants:
            None.
        """
        wf = getattr(state, kwargs["wf"])
        dt_update = sim.settings.dt_update
        h_quantum = state.h_quantum
        setattr(state, kwargs["wf"], wf_rk4(h_quantum, wf, dt_update))
        return state, parameters


Here, the keyword argument ``wf`` specifies which wavefunction attribute of the ``state`` object to update.

In order to use it in an algorithm, we need to specify the keyword argument prior to adding it to the recipe. 
We do this by using the ``functools.partial`` function to create a new function with the desired keyword argument:

.. code-block:: python

    from functools import partial
    from qc_lab.algorithms import MeanField

    myMF = MeanField()
    update_wf = partial(update_wf_rk4, wf="wf")
    myMF.update_recipe.append(update_wf)

