"""
This module contains the tasks that initialize quantities in the state and parameters
objects.

These are typically used in the initialization recipe of the algorithm object.
"""

import logging
import numpy as np
from qclab import functions

logger = logging.getLogger(__name__)


def initialize_variable_objects(sim, state, parameters, **kwargs):
    """
    Populates the state object with non-private variables in ``sim.initial_state``, and an empty
    dictionary for storing output quantities.

    For any non-private (i.e. not beginning with "_") ndarray in ``sim.initial_state``,
    a new array is created in ``state`` with shape `(batch_size, *original_shape)`` where
    ``original_shape`` is the shape of the array in ``sim.initial_state``.
    The new array is initialized by copying the original array into each slice along the first axis.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    None

    .. rubric:: Variable Modifications
    state[name] : ndarray
        Initialized state variable with shape (batch_size, *original_shape).
        ``name`` is the name of the variable in ``sim.initial_state``.
    state["output_dict"] : dict
        Dictionary to store output quantities during the simulation.

    """
    for name in sim.initial_state.keys():
        obj = sim.initial_state[name]
        if isinstance(obj, np.ndarray) and name[0] != "_":
            initial_shape = np.shape(obj)
            new_obj = np.ascontiguousarray(
                np.zeros((sim.settings.batch_size, *initial_shape), dtype=obj.dtype)
                + obj[np.newaxis]
            )
            logger.info(
                "Initializing state variable %s with shape %s.", name, new_obj.shape
            )
            state[name] = new_obj
        elif name[0] != "_":
            logger.warning(
                "Variable %s in sim.initial_state is not a numpy.ndarray, "
                "skipping initialization in state variable object.",
                name,
            )
    state["output_dict"] = {}
    return state, parameters


def initialize_norm_factor(sim, state, parameters, **kwargs):
    """
    Assigns the normalization factor to the state object.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    norm_factor_name : str, default: "norm_factor"
        Name of the normalization factor in the state object.

    .. rubric:: Variable Modifications
    state[norm_factor_name] : int
        Normalization factor for trajectory averages.
    """
    norm_factor_name = kwargs.get("norm_factor_name", "norm_factor")
    state[norm_factor_name] = sim.settings.batch_size
    return state, parameters


def initialize_branch_seeds(sim, state, parameters, **kwargs):
    """
    Converts seeds into branch seeds for deterministic surface hopping.

    This is done by first assuming that the number of branches is equal to the number of quantum states.
    Then, a branch index is created which gives the branch index of each seed in the batch.
    Then a new set of seeds is created by floor dividing the original seeds by the number
    of branches so that the seeds corresponding to different branches within the same
    trajectory are the same.

    Notably, this leads to the number of unique classical initial conditions
    being equal to the number of trajectories divided by the number of
    branches in deterministic surface hopping.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    seed_name : str, default: "seed"
        Name of the seeds array in ``state``.
    branch_ind_name : str, default: "branch_ind"
        Name of the branch index array in ``state``.

    .. rubric:: Variable Modifications
    state[branch_ind_name] : ndarray
        Branch index for each trajectory.
    state[seed_name] : ndarray
        Seeds remapped for branches.
    """
    seed_name = kwargs.get("seed_name", "seed")
    branch_ind_name = kwargs.get("branch_ind_name", "branch_ind")
    # First ensure that the number of branches is correct.
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size

    if batch_size % num_branches != 0:
        logger.error(
            "Batch size must be an integer multiple of sim.model.constants.num_quantum_states"
        )
        raise ValueError(
            "Batch size must be an integer multiple of sim.model.constants.num_quantum_states"
        )

    # Next, determine the number of trajectories that have been run by assuming that
    # the minimum seed in the current batch of seeds is the number of trajectories
    # that have been run modulo num_branches.
    orig_seed = state[seed_name]
    # Now construct a branch index for each trajectory in the expanded batch.
    state[branch_ind_name] = np.tile(
        np.arange(num_branches), batch_size // num_branches
    )
    # Now generate the new seeds for each trajectory in the expanded batch.
    new_seeds = orig_seed // num_branches
    state[seed_name] = new_seeds
    return state, parameters


def initialize_z_mcmc(sim, state, parameters, **kwargs):
    """
    Initializes classical coordinates according to Boltzmann statistics using Markov-
    Chain Monte Carlo with a Metropolis-Hastings algorithm.

    The algorithm has two modes, separable and non-separable. In the separable
    mode, each classical coordinate is evolved as an independent random walker.
    In the non-separable mode, the full classical coordinate vector is evolved as a
    single random walker. The separable mode converges much faster but assumes that
    the classical Hamiltonian can be written as a sum of independent terms depending
    on each classical coordinate.

    .. rubric:: Required Constants
    mcmc_burn_in_size : int, default: 1000
        Burn-in step count.
    mcmc_sample_size : int, default: 10000
        Number of retained samples.
    mcmc_std : float, default: 1.0
        Sampling standard deviation.
    mcmc_h_c_separable : bool, default: True
        Whether the classical Hamiltonian is separable.
    mcmc_init_z : ndarray, default: output of ``gen_sample_gaussian``
        Initial coordinate sample.
    kBT : float
        Thermal quantum.

    .. rubric:: Keyword Arguments
    seed_name : str
        Attribute name of the seeds array in ``state``.
    z_name : str
        Name of destination attribute in the ``state`` object.

    .. rubric:: Variable Modifications
    state[z_name] : ndarray
        Initialized classical coordinates.
    """
    seed = getattr(state, kwargs["seed_name"])
    z_name = kwargs["z_name"]
    burn_in_size = sim.model.constants.get("mcmc_burn_in_size", 1000)
    sample_size = sim.model.constants.get("mcmc_sample_size", 10000)
    mcmc_h_c_separable = sim.model.constants.get("mcmc_h_c_separable", True)
    burn_in_seeds = np.arange(burn_in_size)
    sample_seeds = np.arange(sample_size)
    save_inds = np.zeros(len(seed), dtype=int)
    out_tmp = np.zeros(
        (sample_size, sim.model.constants.num_classical_coordinates), dtype=complex
    )
    for s, seed_s in enumerate(seed):
        np.random.seed(seed_s)
        save_inds[s] = np.random.randint(0, sample_size)
    mcmc_z_initial, _ = functions.gen_sample_gaussian(
        sim.model.constants, z_initial=None, seed=0, separable=False
    )
    sample = sim.model.constants.get("mcmc_initial_z", mcmc_z_initial)
    h_c, _ = sim.model.get("h_c")
    if mcmc_h_c_separable:
        for s, seed_s in enumerate(burn_in_seeds):
            previous_sample = np.copy(sample)
            previous_z = np.diag(previous_sample)
            previous_energy = h_c(
                sim.model, parameters, z=previous_z, batch_size=len(previous_z)
            )
            proposed_sample, rand = functions.gen_sample_gaussian(
                sim.model.constants,
                z_initial=previous_sample,
                seed=seed_s,
                separable=True,
            )
            new_z = np.diag(proposed_sample)
            new_energy = h_c(sim.model, parameters, z=new_z, batch_size=len(new_z))
            thresh = np.minimum(
                np.ones(sim.model.constants.num_classical_coordinates),
                np.exp(-(new_energy - previous_energy) / sim.model.constants.kBT),
            )
            sample[rand < thresh] = proposed_sample[rand < thresh]
        for s, seed_s in enumerate(sample_seeds):
            previous_sample = np.copy(sample)
            previous_z = np.diag(previous_sample)
            previous_energy = h_c(
                sim.model, parameters, z=previous_z, batch_size=len(previous_z)
            )
            proposed_sample, rand = functions.gen_sample_gaussian(
                sim.model.constants,
                z_initial=previous_sample,
                seed=seed_s,
                separable=True,
            )
            new_z = np.diag(proposed_sample)
            new_energy = h_c(sim.model, parameters, z=new_z, batch_size=len(new_z))
            thresh = np.minimum(
                np.ones(sim.model.constants.num_classical_coordinates),
                np.exp(-(new_energy - previous_energy) / sim.model.constants.kBT),
            )
            sample[rand < thresh] = proposed_sample[rand < thresh]
            out_tmp[s] = sample
            state[z_name] = out_tmp[save_inds]
    else:
        # If not separable, do the full MCMC.
        for s, seed_s in enumerate(burn_in_seeds):
            previous_sample = np.copy(sample)
            previous_energy = h_c(
                sim.model,
                parameters,
                z=previous_sample,
                batch_size=len(previous_sample),
            )
            proposed_sample, rand = functions.gen_sample_gaussian(
                sim.model.constants,
                z_initial=previous_sample,
                seed=seed_s,
                separable=False,
            )
            new_energy = h_c(
                sim.model,
                parameters,
                z=proposed_sample,
                batch_size=len(proposed_sample),
            )
            thresh = min(
                1, np.exp(-(new_energy - previous_energy) / sim.model.constants.kBT)
            )
            if rand < thresh:
                sample = proposed_sample
        for s, seed_s in enumerate(sample_seeds):
            previous_sample = np.copy(sample)
            previous_energy = h_c(
                sim.model,
                parameters,
                z=previous_sample,
                batch_size=len(previous_sample),
            )
            proposed_sample, rand = functions.gen_sample_gaussian(
                sim.model.constants,
                z_initial=previous_sample,
                seed=seed_s,
                separable=False,
            )
            new_energy = h_c(
                sim.model,
                parameters,
                z=proposed_sample,
                batch_size=len(proposed_sample),
            )
            thresh = min(
                1, np.exp(-(new_energy - previous_energy) / sim.model.constants.kBT)
            )
            if rand < thresh:
                sample = proposed_sample
            out_tmp[s] = sample
        state[z_name] = out_tmp[save_inds]
    return state, parameters


def initialize_z(sim, state, parameters, **kwargs):
    """
    Initializes the classical coordinate by using the init_classical function from the
    model object.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    seed_name : str
        Name of seed attribute in state object.
    z_name : str
        Name of classical coordinates in the state object.

    .. rubric:: Variable Modifications
    state[z_name] : ndarray
        initialized classical coordinates.
    """
    seed_name = kwargs.get("seed_name", "seed")
    seed = state[seed_name]
    z_name = kwargs.get("z_name", "z")
    init_classical, has_init_classical = sim.model.get("init_classical")
    if has_init_classical:
        state[z_name] = init_classical(sim.model, parameters, seed=seed)
    else:
        state, parameters = initialize_z_mcmc(
            sim, state, parameters, seed_name=seed_name, z_name=z_name
        )
    return state, parameters


def copy_in_state(sim, state, parameters, **kwargs):
    """
    Creates a copy of a variable in the state object with a new name.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    copy_name : str
        Name of the copy in the state object.
    orig_name : str
        Name of the source in the state object.

    .. rubric:: Variable Modifications
    state[copy_name] : type of state[orig_name]
        Copy of ``state[orig_name]``.
    """
    state[kwargs["copy_name"]] = np.copy(state[kwargs["orig_name"]])
    return state, parameters


def initialize_active_surface(sim, state, parameters, **kwargs):
    """
    Initializes the active surface, active surface index, and initial active
    surface index for FSSH.

    If ``fssh_deterministic=True`` it will set the initial active surface index
    to be the same as the branch index and assert that the number of branches is
    equal to the number of quantum states.

    If ``fssh_deterministic=False`` it will stochastically sample the active
    surface from the density corresponding to the initial quantum wavefunction
    in the adiabatic basis.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    act_surf_ind_0_name : str, default: "act_surf_ind_0"
        Name of the initial active surface index in the state object.
    act_surf_ind_name : str, default: "act_surf_ind"
        Name of the active surface index in the state object.
    act_surf_name : str, default: "act_surf"
        Name of the active surface in the state object.
    init_act_surf_rand_vals_name : str, default: "init_act_surf_rand_vals"
        Name of the random numbers for active surface initialization in FSSH.
    wf_adb_name : str, default: "wf_adb"
        Name of the adiabatic wavefunction in the state object.

    .. rubric:: Variable Modifications
    state[act_surf_ind_0_name] : ndarray
        Initial active surface index.
    state[act_surf_ind_name] : ndarray
        Current active surface index.
    state[act_surf_name] : ndarray
        Active surface wavefunctions.
    """
    act_surf_ind_0_name = kwargs.get("act_surf_ind_0_name", "act_surf_ind_0")
    act_surf_ind_name = kwargs.get("act_surf_ind_name", "act_surf_ind")
    act_surf_name = kwargs.get("act_surf_name", "act_surf")
    init_act_surf_rand_vals_name = kwargs.get(
        "init_act_surf_rand_vals_name", "init_act_surf_rand_vals"
    )
    wf_adb_name = kwargs.get("wf_adb_name", "wf_adb")
    wf_adb = state[wf_adb_name]
    init_act_surf_rand_vals = state[init_act_surf_rand_vals_name]
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_states = sim.model.constants.num_quantum_states
    num_trajs = sim.settings.batch_size // num_branches
    if sim.algorithm.settings.fssh_deterministic:
        act_surf_ind_0 = np.tile(np.arange(num_branches, dtype=int), (num_trajs, 1))
    else:
        cumulative_pops = np.cumsum(
            np.real(np.abs(wf_adb.reshape((num_trajs, num_branches, num_states))) ** 2),
            axis=-1,
        )
        act_surf_ind_0 = np.argmax(
            cumulative_pops > init_act_surf_rand_vals[:, :, np.newaxis], axis=-1
        ).astype(int)
    act_surf_ind_0 = act_surf_ind_0.reshape(-1)
    state[act_surf_ind_0_name] = np.copy(act_surf_ind_0)
    state[act_surf_ind_name] = np.copy(act_surf_ind_0)
    act_surf = np.zeros((num_trajs * num_branches, num_states), dtype=int)
    traj_inds = np.repeat(np.arange(num_trajs), num_branches)
    branch_inds = np.tile(np.arange(num_branches), num_trajs)
    traj_branch_ind = traj_inds * num_branches + branch_inds
    act_surf[traj_branch_ind, act_surf_ind_0] = 1
    state[act_surf_name] = act_surf
    return state, parameters


def initialize_random_values_fssh(sim, state, parameters, **kwargs):
    """
    Initializes a set of random numbers using the trajectory seeds for FSSH.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    hop_prob_rand_vals_name : str, default: "hop_prob_rand_vals"
        Name of the random numbers for hop decisions in the state object.
    init_act_surf_rand_vals_name : str, default: "init_act_surf_rand_vals"
        Name of the random numbers for active surface initialization in FSSH.
    seed_name : str, default: "seed"
        Name of the seeds array in the state object.

    .. rubric:: Variable Modifications
    state[hop_prob_rand_vals_name] : ndarray
        Random numbers for hop decisions.
    state[init_act_surf_rand_vals_name] : ndarray
        Random numbers for active surface selection in stochastic FSSH.
    """
    hop_prob_rand_vals_name = kwargs.get(
        "hop_prob_rand_vals_name", "hop_prob_rand_vals"
    )
    init_act_surf_rand_vals_name = kwargs.get(
        "init_act_surf_rand_vals_name", "init_act_surf_rand_vals"
    )
    seed_name = kwargs.get("seed_name", "seed")
    seed = state[seed_name]
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    hop_prob_rand_vals = np.zeros((batch_size, len(sim.settings.t_update)))
    init_act_surf_rand_vals = np.zeros((batch_size, num_branches))
    for nt in range(batch_size):
        np.random.seed(seed[int(nt * num_branches)])
        hop_prob_rand_vals[nt] = np.random.rand(len(sim.settings.t_update))
        init_act_surf_rand_vals[nt] = np.random.rand(num_branches)
    state[hop_prob_rand_vals_name] = hop_prob_rand_vals
    state[init_act_surf_rand_vals_name] = init_act_surf_rand_vals
    return state, parameters


def initialize_dm_adb_0_fssh(sim, state, parameters, **kwargs):
    """
    Initializes the initial adiabatic density matrix for FSSH.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    dm_adb_0_name : str, default: "dm_adb_0"
        Name of the initial adiabatic density matrix in the state object.
    wf_adb_name : str, default: "wf_adb"
        Name of the adiabatic wavefunction in the state object.

    .. rubric:: Variable Modifications
    state[dm_adb_0_name] : ndarray
        Initial adiabatic density matrix.
    """
    dm_adb_0_name = kwargs.get("dm_adb_0_name", "dm_adb_0")
    wf_adb_name = kwargs.get("wf_adb_name", "wf_adb")
    wf_adb = state[wf_adb_name]
    state[dm_adb_0_name] = np.einsum(
        "ti,tj->tij",
        wf_adb,
        np.conj(wf_adb),
        optimize="greedy",
    )
    return state, parameters
