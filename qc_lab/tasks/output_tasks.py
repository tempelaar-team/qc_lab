import numpy as np
from qc_lab.tasks.update_tasks import *


def update_t(algorithm, sim, parameters, state, **kwargs):
    """
    Update the time in the state object.

    Required constants:
        - None.
    """
    time_axis = sim.settings.tdat
    state.t = time_axis[sim.t_ind]*np.ones(sim.settings.batch_size)
    return parameters, state


def update_dm_db_mf(algorithm, sim, parameters, state, **kwargs):
    """
    Update the density matrix in the mean-field approximation.

    Required constants:
        - None.
    """
    del sim, kwargs
    wf_db = state.wf_db
    state.dm_db = np.einsum("ti,tj->tij", wf_db, np.conj(wf_db), optimize="greedy")
    return parameters, state


def update_classical_energy(algorithm, sim, parameters, state, **kwargs):
    """
    Update the classical energy.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    h_c, _ = sim.model.get("h_c")
    state.classical_energy = np.real(h_c(sim.model, parameters, z=z, batch_size=len(z)))
    return parameters, state


def update_classical_energy_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the classical energy as a sum of equally-weighted contributions from each branch.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_states = sim.model.constants.num_quantum_states
    h_c, _ = sim.model.get("h_c")
    if sim.algorithm.settings.fssh_deterministic:
        state.classical_energy = 0
        branch_weights = np.sqrt(
            np.einsum(
                "tbbb->tb",
                state.dm_adb_0.reshape(
                    (batch_size, num_branches, num_states, num_states)
                ),
            )
        )
        for branch_ind in range(num_branches):
            z_branch = (
                z[state.branch_ind == branch_ind]
                * branch_weights[:, branch_ind][:, np.newaxis]
            )
            state.classical_energy = state.classical_energy + h_c(
                sim.model,
                parameters,
                z=z_branch,
                batch_size=len(z_branch),
            )
    else:
        state.classical_energy = 0
        for branch_ind in range(num_branches):
            z_branch = z[state.branch_ind == branch_ind]
            state.classical_energy = state.classical_energy + h_c(
                sim.model,
                parameters,
                z=z_branch,
                batch_size=len(z_branch),
            )
        state.classical_energy = state.classical_energy / num_branches
    state.classical_energy = np.real(state.classical_energy)
    return parameters, state


def update_quantum_energy(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum energy w.r.t the wavefunction specified by wf.

    Required constants:
        - None.
    """
    del sim
    wf = kwargs["wf"]
    state.quantum_energy = np.real(
        np.einsum("ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy")
    )
    return parameters, state


def update_quantum_energy_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum energy w.r.t the wavefunction specified by wf.

    Required constants:
        - None.
    """
    wf = kwargs["wf"]

    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
        batch_size = sim.settings.batch_size // num_branches
        num_states = sim.model.constants.num_quantum_states
        wf = wf * np.sqrt(
            np.einsum(
                "tbbb->tb",
                state.dm_adb_0.reshape(
                    (batch_size, num_branches, num_states, num_states)
                ),
            ).flatten()[:, np.newaxis]
        )
        state.quantum_energy = np.einsum(
            "ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy"
        )
    else:
        state.quantum_energy = np.einsum(
            "ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy"
        )
        state.quantum_energy = state.quantum_energy
    state.quantum_energy = np.real(state.quantum_energy)
    return parameters, state


def update_dm_db_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the diabatic density matrix for FSSH.

    Required constants:
        - None.
    """
    del kwargs
    dm_adb_branch = np.einsum(
        "...i,...j->...ij",
        state.wf_adb,
        np.conj(state.wf_adb),
        optimize="greedy",
    )
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_quantum_states = sim.model.constants.num_quantum_states
    for nt, _ in enumerate(dm_adb_branch):
        np.einsum("...jj->...j", dm_adb_branch[nt])[...] = state.act_surf[nt]
    if sim.algorithm.settings.fssh_deterministic:
        dm_adb_branch = (
            np.einsum(
                "tbbb->tb",
                state.dm_adb_0.reshape(
                    (batch_size, num_branches, num_quantum_states, num_quantum_states)
                ),
            ).flatten()[:, np.newaxis, np.newaxis]
            * dm_adb_branch
        )
    else:
        dm_adb_branch = dm_adb_branch / num_branches
    parameters, state = basis_transform_mat(
        algorithm,
        sim,
        parameters,
        state,
        input_mat=dm_adb_branch.reshape(
            (
                batch_size * num_branches,
                num_quantum_states,
                num_quantum_states,
            )
        ),
        basis=state.eigvecs,
        output_name="dm_db_branch",
    )
    state.dm_db = np.sum(
        state.dm_db_branch.reshape(
            (
                batch_size,
                num_branches,
                num_quantum_states,
                num_quantum_states,
            )
        ),
        axis=-3,
    )
    return parameters, state