import numpy as np
from qc_lab.tasks.default_ingredients import *

def assign_norm_factor_mf(algorithm, sim, parameters, state, **kwargs):
    """
    Assign the normalization factor to the state object for MF dynamics.

    Required constants:
        - None.
    """
    del kwargs
    state.norm_factor = sim.settings.batch_size
    return parameters, state


def assign_norm_factor_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Assign the normalization factor to the state object for FSSH.

    Required constants:
        - None.
    """
    del kwargs
    if sim.algorithm.settings.fssh_deterministic:
        state.norm_factor = (
            sim.settings.batch_size // sim.model.constants.num_quantum_states
        )
    else:
        state.norm_factor = sim.settings.batch_size
    return parameters, state


def initialize_branch_seeds(algorithm, sim, parameters, state, **kwargs):
    """
    Initialize the seeds in each branch.

    Required constants:
        - num_quantum_states (int): Number of quantum states. Default: None.
    """
    del kwargs
    # First ensure that the number of branches is correct.
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size
    assert (
        batch_size % num_branches == 0
    ), "Batch size must be divisible by number of quantum states for deterministic surface hopping."
    # Next, determine the number of trajectories that have been run by assuming that
    # the minimum seed in the current batch of seeds is the number of trajectories
    # that have been run modulo num_branches.
    orig_seeds = state.seed
    # Now construct a branch index for each trajectory in the expanded batch.
    state.branch_ind = (
        np.zeros((batch_size // num_branches, num_branches), dtype=int)
        + np.arange(num_branches)[np.newaxis, :]
    ).flatten()
    # Now generate the new seeds for each trajectory in the expanded batch.
    new_seeds = orig_seeds // num_branches
    parameters.seed = new_seeds
    state.seed = new_seeds
    return parameters, state


def initialize_z(algorithm, sim, parameters, state, **kwargs):
    """
    Initialize the classical coordinate by using the init_classical function from the model object.

    Required constants:
        - None.
    """
    seed = kwargs["seed"]
    init_classical, has_init_classical = sim.model.get("init_classical")
    if has_init_classical:
        state.z = init_classical(sim.model, parameters, seed=seed)
        return parameters, state
    state.z = numerical_boltzmann_mcmc_init_classical(sim.model, parameters, seed=seed)
    return parameters, state


def assign_to_parameters(algorithm, sim, parameters, state, **kwargs):
    """
    Assign the value of the variable "val" to the parameters object with the name "name".

    Required constants:
        - None.
    """
    del sim
    name = kwargs["name"]
    val = kwargs["val"]
    setattr(parameters, name, val)
    return parameters, state


def assign_to_state(algorithm, sim, parameters, state, **kwargs):
    """
    Creates a new state variable with the name "name" and the value "val".

    Required constants:
        - None.
    """
    del sim
    name = kwargs["name"]
    val = kwargs["val"]
    setattr(state, name, np.copy(val))
    return parameters, state



def initialize_active_surface(algorithm, sim, parameters, state, **kwargs):
    """
    Initializes the active surface (act_surf), active surface index
    (act_surf_ind) and initial active surface index (act_surf_ind_0)
    for FSSH.

    If fssh_deterministic is true it will set act_surf_ind_0 to be the same as
    the branch index and assert that the number of branches (num_branches)
    is equal to the number of quantum states (num_states).

    If fssh_deterministic is false it will stochastically sample the active
    surface from the density specified by the initial quantum wavefunction in the
    adiabatic basis.

    Required constants:
        - num_quantum_states (int): Number of quantum states. Default: None.
    """
    del kwargs
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_states = sim.model.constants.num_quantum_states
    num_trajs = sim.settings.batch_size // num_branches
    if sim.algorithm.settings.fssh_deterministic:
        act_surf_ind_0 = np.arange(num_branches, dtype=int)[np.newaxis, :] + np.zeros(
            (num_trajs, num_branches)
        ).astype(int)
    else:
        intervals = np.cumsum(
            np.real(
                np.abs(state.wf_adb.reshape((num_trajs, num_branches, num_states))) ** 2
            ),
            axis=-1,
        )
        bool_mat = intervals > state.stochastic_sh_rand_vals[:, :, np.newaxis]
        act_surf_ind_0 = np.argmax(bool_mat, axis=-1).astype(int)
    state.act_surf_ind_0 = np.copy(act_surf_ind_0)
    state.act_surf_ind = np.copy(act_surf_ind_0)
    act_surf = np.zeros((num_trajs, num_branches, num_states), dtype=int)
    traj_inds = (
        (np.arange(num_trajs)[:, np.newaxis] * np.ones((num_trajs, num_branches)))
        .flatten()
        .astype(int)
    )
    branch_inds = (
        (np.arange(num_branches)[np.newaxis, :] * np.ones((num_trajs, num_branches)))
        .flatten()
        .astype(int)
    )
    act_surf[traj_inds, branch_inds, act_surf_ind_0.flatten()] = 1
    state.act_surf = act_surf.astype(int).reshape(
        (num_trajs * num_branches, num_states)
    )
    parameters.act_surf_ind = state.act_surf_ind
    return parameters, state


def initialize_random_values_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Initialize a set of random variables using the trajectory seeds for FSSH.

    Required constants:
        - None.
    """
    del kwargs
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    state.hopping_probs_rand_vals = np.zeros((batch_size, len(sim.settings.tdat)))
    state.stochastic_sh_rand_vals = np.zeros((batch_size, num_branches))
    for nt in range(batch_size):
        np.random.seed(state.seed[int(nt * num_branches)])
        state.hopping_probs_rand_vals[nt] = np.random.rand(len(sim.settings.tdat))
        state.stochastic_sh_rand_vals[nt] = np.random.rand(num_branches)
    return parameters, state


def initialize_dm_adb_0_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Initialize the initial adiabatic density matrix for FSSH.

    Required constants:
        - None.
    """
    del sim, kwargs
    state.dm_adb_0 = np.einsum(
        "...i,...j->...ij",
        state.wf_adb,
        np.conj(state.wf_adb),
        optimize="greedy",
    )
    return parameters, state