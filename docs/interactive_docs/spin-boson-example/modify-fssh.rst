.. _modify-fssh:


Modifying the FSSH Algorithm
============================

Let's try modifying the FSSH algorithm so that the directions of the velocities of frustrated trajectories are reversed.
In the `complex coordinate formalism <https://doi.org/10.1021/acs.jctc.4c00555>`_, this means conjugating the `z` coordinate of the frustrated trajectories.
To this end, we write the following function:

.. code-block:: python


    def update_z_reverse_frustrated_fssh(sim, state, parameters):
        """
        Reverse the velocities of frustrated trajectories in the FSSH algorithm.
        """
        # Get the indices of trajectories that were frustrated
        # (i.e., did not successfully hop but were eligible to hop).
        frustrated_indices = state.hop_ind[~state.hop_successful]
        # Reverse the velocities for these indices, in the complex classical coordinate 
        # formalism, this means conjugating the z coordinate.
        state.z[frustrated_indices] = state.z[frustrated_indices].conj()
        return state, parameters


Now we can insert this function as a task into an instance of the FSSH algorithm object. To know where we should insert it, we can look 
at the `update_recipe <../../user_guide/FSSH_recipe.html>`_ attribute of the FSSH algorithm object.



.. code-block:: python

    # Print the update recipe to see where to insert our task.
    for ind, task in enumerate(sim.model.algorithm.update_recipe):
        print(f"Task #{ind}", task)


.. code-block:: console
    :caption: Output

    Task #0 <function FewestSwitchesSurfaceHopping._assign_eigvecs_to_state at 0x777d902d2710>
    Task #1 <function FewestSwitchesSurfaceHopping._update_z_rk4 at 0x777d902d28c0>
    Task #2 <function FewestSwitchesSurfaceHopping._update_wf_db_eigs at 0x777d902d2950>
    Task #3 <function FewestSwitchesSurfaceHopping._update_h_quantum at 0x777d902d2560>
    Task #4 <function FewestSwitchesSurfaceHopping._diagonalize_matrix at 0x777d902d25f0>
    Task #5 <function FewestSwitchesSurfaceHopping._gauge_fix_eigs_update at 0x777d902d29e0>
    Task #6 <function update_hop_probs_fssh at 0x777d902d13f0>
    Task #7 <function update_hop_inds_fssh at 0x777d902d1480>
    Task #8 <function update_hop_vals_fssh at 0x777d902d1510>
    Task #9 <function update_z_hop_fssh at 0x777d902d15a0>
    Task #10 <function update_act_surf_hop_fssh at 0x777d902d1630>
    Task #11 <function update_act_surf_wf at 0x777d902d0e50>

A good place to invert the velocities of frustrated trajectories is after the `update_z_hop_fssh` task, which updates the `z` coordinates after a hop.
QC Lab makes this particularly easy to do by using Python's built-in list methods to insert our new task into the correct position.

.. code-block:: python

    # Insert the function for reversing velocities as a task into the update recipe.
    sim.algorithm.update_recipe.insert(10, update_z_reverse_frustrated_fssh)
    # Now we can verify we put it in the right place by printing the update recipe again.
    for ind, task in enumerate(sim.model.algorithm.update_recipe):
        print(f"Task #{ind}", task)

.. code-block:: console
    :caption: Output

    Task #0 <function FewestSwitchesSurfaceHopping._assign_eigvecs_to_state at 0x777d902d2710>
    Task #1 <function FewestSwitchesSurfaceHopping._update_z_rk4 at 0x777d902d28c0>
    Task #2 <function FewestSwitchesSurfaceHopping._update_wf_db_eigs at 0x777d902d2950>
    Task #3 <function FewestSwitchesSurfaceHopping._update_h_quantum at 0x777d902d2560>
    Task #4 <function FewestSwitchesSurfaceHopping._diagonalize_matrix at 0x777d902d25f0>
    Task #5 <function FewestSwitchesSurfaceHopping._gauge_fix_eigs_update at 0x777d902d29e0>
    Task #6 <function update_hop_probs_fssh at 0x777d902d13f0>
    Task #7 <function update_hop_inds_fssh at 0x777d902d1480>
    Task #8 <function update_hop_vals_fssh at 0x777d902d1510>
    Task #9 <function update_z_hop_fssh at 0x777d902d15a0>
    Task #10 <function update_z_reverse_frustrated_fssh at 0x777d580da200>
    Task #11 <function update_act_surf_hop_fssh at 0x777d902d1630>
    Task #12 <function update_act_surf_wf at 0x777d902d0e50>


The output has now changed to:


.. image:: fssh_lreorg_inv_vel.png
   :alt: Modified FSSH populations.
   :align: center
   :width: 50%