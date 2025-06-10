import numpy as np
import warnings
from numba import njit
from qc_lab.tasks.default_ingredients import *



def update_dh_c_dzc(algorithm, sim, parameters, state, **kwargs):
    """
    Update the gradient of the classical Hamiltonian
    w.r.t the conjugate classical coordinate.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    dh_c_dzc, has_dh_c_dzc = sim.model.get("dh_c_dzc")
    if has_dh_c_dzc:
        state.dh_c_dzc = dh_c_dzc(sim.model, parameters, z=z)
        return parameters, state
    state.dh_c_dzc = dh_c_dzc_finite_differences(sim.model, parameters, z=z)
    return parameters, state


def update_dh_qc_dzc(algorithm, sim, parameters, state, **kwargs):
    """
    Update the gradient of the quantum-classical Hamiltonian
    w.r.t the conjugate classical coordinate.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    dh_qc_dzc, has_dh_qc_dzc = sim.model.get("dh_qc_dzc")
    if has_dh_qc_dzc:
        state.dh_qc_dzc = dh_qc_dzc(sim.model, parameters, z=z)
        return parameters, state
    state.dh_qc_dzc = dh_qc_dzc_finite_differences(sim.model, parameters, z=z)
    return parameters, state


def update_classical_forces(algorithm, sim, parameters, state, **kwargs):
    """
    Update the classical forces.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    parameters, state = update_dh_c_dzc(algorithm, sim, parameters, state, z=z)
    state.classical_forces = state.dh_c_dzc
    return parameters, state


@njit
def calc_sparse_inner_product(inds, mels, shape, vec_l, vec_r):
    """
    Given the indices, matrix elements and shape of a sparse matrix, calculate the expectation value with a vector.

    Required constants:
        - None.
    """
    out = np.zeros((shape[:2])) + 0.0j
    for i in range(len(inds[0])):
        out[inds[0][i], inds[1][i]] = (
            out[inds[0][i], inds[1][i]]
            + np.conj(vec_l[inds[0][i], inds[2][i]])
            * mels[i]
            * vec_r[inds[0][i], inds[3][i]]
        )
    return out


def update_quantum_classical_forces(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum-classical forces w.r.t the state defined by wf.

    If the model has a gauge_field_force ingredient, this term will be added
    to the quantum-classical forces.

    Required constants:
        - None.
    """
    z = kwargs["z"]
    wf = kwargs["wf"]
    use_gauge_field_force = kwargs.get("use_gauge_field_force", False)
    parameters, state = update_dh_qc_dzc(algorithm, sim, parameters, state, z=z)
    inds, mels, shape = state.dh_qc_dzc
    state.quantum_classical_forces = calc_sparse_inner_product(
        inds, mels, shape, wf, wf
    )
    gauge_field_force, has_gauge_field_force = sim.model.get("gauge_field_force")
    if has_gauge_field_force and use_gauge_field_force:
        state.quantum_classical_forces += gauge_field_force(parameters, z=z, wf=wf)
    return parameters, state



def diagonalize_matrix(algorithm, sim, parameters, state, **kwargs):
    """
    Diagonalizes a given matrix and stores the eigenvalues and eigenvectors in the state object.

    Required constants:
        - None.
    """
    del sim
    matrix = kwargs["matrix"]
    eigvals_name = kwargs["eigvals_name"]
    eigvecs_name = kwargs["eigvecs_name"]
    eigvals, eigvecs = np.linalg.eigh(matrix)
    setattr(state, eigvals_name, eigvals)
    setattr(state, eigvecs_name, eigvecs)
    return parameters, state



def analytic_der_couple_phase(algorithm, sim, parameters, state, eigvals, eigvecs):
    """
    Calculates the phase change needed to fix the gauge using analytic derivative couplings.

    Required constants:
        - None.
    """
    del parameters
    der_couple_q_phase = np.ones(
        (
            sim.settings.batch_size,
            sim.model.constants.num_quantum_states,
        ),
        dtype=complex,
    )
    der_couple_p_phase = np.ones(
        (
            sim.settings.batch_size,
            sim.model.constants.num_quantum_states,
        ),
        dtype=complex,
    )
    for i in range(sim.model.constants.num_quantum_states - 1):
        j = i + 1
        evec_i = eigvecs[..., i]
        evec_j = eigvecs[..., j]
        eval_i = eigvals[..., i]
        eval_j = eigvals[..., j]
        ev_diff = eval_j - eval_i
        plus = np.zeros_like(ev_diff)
        if np.any(np.abs(ev_diff) < 1e-10):
            plus[np.where(np.abs(ev_diff) < 1e-10)] = 1
            warnings.warn("Degenerate eigenvalues detected.")
        der_couple_zc = np.zeros(
            (
                sim.settings.batch_size,
                sim.model.constants.num_classical_coordinates,
            ),
            dtype=complex,
        )
        der_couple_z = np.zeros(
            (
                sim.settings.batch_size,
                sim.model.constants.num_classical_coordinates,
            ),
            dtype=complex,
        )
        inds, mels, _ = state.dh_qc_dzc
        np.add.at(
            der_couple_zc,
            (inds[0], inds[1]),
            np.conj(evec_i)[inds[0], inds[2]]
            * mels
            * evec_j[inds[0], inds[3]]
            / ((ev_diff + plus)[inds[0]]),
        )
        np.add.at(
            der_couple_z,
            (inds[0], inds[1]),
            np.conj(evec_i)[inds[0], inds[3]]
            * np.conj(mels)
            * evec_j[inds[0], inds[2]]
            / ((ev_diff + plus)[inds[0]]),
        )
        der_couple_p = (
            1.0j
            * np.sqrt(
                1
                / (
                    2
                    * sim.model.constants.classical_coordinate_weight
                    * sim.model.constants.classical_coordinate_mass
                )
            )[..., :]
            * (der_couple_z - der_couple_zc)
        )
        der_couple_q = np.sqrt(
            sim.model.constants.classical_coordinate_weight
            * sim.model.constants.classical_coordinate_mass
            / 2
        )[..., :] * (der_couple_z + der_couple_zc)
        der_couple_q_angle = np.angle(
            der_couple_q[
                np.arange(len(der_couple_q)),
                np.argmax(np.abs(der_couple_q), axis=-1),
            ]
        )
        der_couple_p_angle = np.angle(
            der_couple_p[
                np.arange(len(der_couple_p)),
                np.argmax(np.abs(der_couple_p), axis=-1),
            ]
        )
        der_couple_q_angle[np.where(np.abs(der_couple_q_angle) < 1e-12)] = 0
        der_couple_p_angle[np.where(np.abs(der_couple_p_angle) < 1e-12)] = 0
        der_couple_q_phase[..., i + 1 :] = (
            np.exp(1.0j * der_couple_q_angle[..., np.newaxis])
            * der_couple_q_phase[..., i + 1 :]
        )
        der_couple_p_phase[..., i + 1 :] = (
            np.exp(1.0j * der_couple_p_angle[..., np.newaxis])
            * der_couple_p_phase[..., i + 1 :]
        )
    return der_couple_q_phase, der_couple_p_phase


def gauge_fix_eigs(algorithm, sim, parameters, state, **kwargs):
    """
    Fixes the gauge of the eigenvectors as specified by the gauge_fixing parameter.

    if gauge_fixing >= 0:
        Only the sign of the eigenvector is changed

    if gauge_fixing >= 1:
        The phase of the eigenvector is determined from its overlap
        with the previous eigenvector and the phase is fixed.

    if gauge_fixing >= 2:
        The phase of the eigenvector is determined by calculating the derivative couplings.

    Required constants:
        - None.
    """
    eigvals = kwargs["eigvals"]
    eigvecs = kwargs["eigvecs"]
    eigvecs_previous = kwargs["eigvecs_previous"]
    output_eigvecs_name = kwargs["output_eigvecs_name"]
    if kwargs["gauge_fixing"] >= 1:
        phase = np.exp(
            -1.0j * np.angle(np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2))
        )
        eigvecs = np.einsum("tai,ti->tai", eigvecs, phase, optimize="greedy")
    if kwargs["gauge_fixing"] >= 2:
        z = kwargs["z"]
        parameters, state = update_dh_qc_dzc(algorithm, sim, parameters, state, z=z)
        der_couple_q_phase, _ = analytic_der_couple_phase(
            algorithm, sim, parameters, state, eigvals, eigvecs
        )
        eigvecs = np.einsum(
            "tai,ti->tai", eigvecs, np.conj(der_couple_q_phase), optimize="greedy"
        )
    if kwargs["gauge_fixing"] >= 0:
        signs = np.sign(np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2))
        eigvecs = np.einsum("tai,ti->tai", eigvecs, signs, optimize="greedy")
    if kwargs["gauge_fixing"] == 2:
        der_couple_q_phase_new, der_couple_p_phase_new = analytic_der_couple_phase(
            algorithm, sim, parameters, state, eigvals, eigvecs
        )
        if (
            np.sum(
                np.abs(np.imag(der_couple_q_phase_new)) ** 2
                + np.abs(np.imag(der_couple_p_phase_new)) ** 2
            )
            > 1e-10
        ):
            warnings.warn(
                "Phase error encountered when fixing gauge analytically.", UserWarning
            )
    setattr(state, output_eigvecs_name, eigvecs)
    return parameters, state


def basis_transform_vec(algorithm, sim, parameters, state, **kwargs):
    """
    Transforms a vector "input_vec" to a new basis defined by "basis".

    Required constants:
        - None.
    """
    del sim
    # Default transformation is adiabatic to diabatic.
    input_vec = kwargs["input_vec"]
    basis = kwargs["basis"]
    output_name = kwargs["output_name"]
    setattr(
        state,
        output_name,
        np.einsum("tij,tj->ti", basis, input_vec, optimize="greedy"),
    )
    return parameters, state


def basis_transform_mat(algorithm, sim, parameters, state, **kwargs):
    """
    Transforms a matrix "input_mat" to a new basis
    defined by "basis" and stores it in the state object
    with name "output_name".

    Required constants:
        - None.
    """
    del sim
    # Default transformation is adiabatic to diabatic.
    input_mat = kwargs["input_mat"]
    basis = kwargs["basis"]
    output_name = kwargs["output_name"]
    setattr(
        state,
        output_name,
        np.einsum(
            "tij,tjl->til",
            basis,
            np.einsum("tjk,tlk->tjl", input_mat, np.conj(basis), optimize="greedy"),
            optimize="greedy",
        ),
    )
    return parameters, state


def update_act_surf_wf(algorithm, sim, parameters, state, **kwargs):
    """
    Update the wavefunction corresponding to the active surface.

    Required constants:
        - None.
    """
    del kwargs
    num_trajs = sim.settings.batch_size
    act_surf_wf = state.eigvecs[
        np.arange(num_trajs, dtype=int),
        :,
        state.act_surf_ind.flatten().astype(int),
    ]
    state.act_surf_wf = act_surf_wf
    return parameters, state


def update_wf_db_eigs(algorithm, sim, parameters, state, **kwargs):
    """
    Evolve the diabatic wavefunction using the electronic eigenbasis.

    Required constants:
        - None.
    """
    wf_db = kwargs["wf_db"]
    adb_name = kwargs["adb_name"]
    output_name = kwargs["output_name"]
    eigvals = kwargs["eigvals"]
    eigvecs = kwargs["eigvecs"]
    evals_exp = np.exp(-1.0j * eigvals * sim.settings.dt)
    parameters, state = basis_transform_vec(
        algorithm,
        sim=sim,
        parameters=parameters,
        state=state,
        input_vec=wf_db,
        basis=np.einsum("...ij->...ji", eigvecs).conj(),
        output_name=adb_name,
    )
    setattr(state, adb_name, (state.wf_adb * evals_exp))
    parameters, state = basis_transform_vec(
        algorithm,
        sim=sim,
        parameters=parameters,
        state=state,
        input_vec=state.wf_adb,
        basis=eigvecs,
        output_name=output_name,
    )
    return parameters, state



@njit
def matprod(mat, vec):
    """
    Perform matrix-vector multiplication.

    Required constants:
        - None.
    """
    out = np.zeros(np.shape(vec)) + 0.0j
    for t in range(len(mat)):
        for i in range(len(mat[0])):
            sum = 0 + 0.0j
            for j in range(len(mat[0,])):
                sum = sum + mat[t, i, j] * vec[t, j]
            out[t, i] = sum
    return out

@njit
def wf_db_rk4(h_quantum, wf_db, dt):
    """
    Low-level function for quantum RK4 propagation.

    Required constants:
        - None.
    """
    k1 = -1j * matprod(h_quantum, wf_db)
    k2 = -1j * matprod(h_quantum, (wf_db + 0.5 * dt * k1))
    k3 = -1j * matprod(h_quantum, (wf_db + 0.5 * dt * k2))
    k4 = -1j * matprod(h_quantum, (wf_db + dt * k3))
    return wf_db + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)


def update_wf_db_rk4(algorithm, sim, parameters, state, **kwargs):
    """
    Update the wavefunction using the 4th-order Runge-Kutta method.

    Required constants:
        - None.
    """
    del kwargs
    dt = sim.settings.dt
    wf_db = state.wf_db
    h_quantum = state.h_quantum
    state.wf_db = wf_db_rk4(h_quantum, wf_db, dt)
    return parameters, state


def calc_delta_z_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the rescaling direction state.delta_z in FSSH.
    """
    traj_ind, final_state_ind, init_state_ind = (
        kwargs["traj_ind"],
        kwargs["final_state_ind"],
        kwargs["init_state_ind"],
    )
    # if hasattr(sim.model, "rescaling_direction_fssh"):
    rescaling_direction_fssh, has_rescaling_direction_fssh = sim.model.get(
        "rescaling_direction_fssh"
    )
    if has_rescaling_direction_fssh:
        delta_z = rescaling_direction_fssh(
            parameters,
            z=state.z[traj_ind],
            init_state_ind=init_state_ind,
            final_state_ind=final_state_ind,
        )
        return delta_z

    inds, mels, _ = state.dh_qc_dzc
    eigvecs_flat = state.eigvecs
    eigvals_flat = state.eigvals
    evec_init_state = eigvecs_flat[traj_ind][:, init_state_ind]
    evec_final_state = eigvecs_flat[traj_ind][:, final_state_ind]
    eval_init_state = eigvals_flat[traj_ind][init_state_ind]
    eval_final_state = eigvals_flat[traj_ind][final_state_ind]
    ev_diff = eval_final_state - eval_init_state
    inds_traj_ind = (
        inds[0][inds[0] == traj_ind],
        inds[1][inds[0] == traj_ind],
        inds[2][inds[0] == traj_ind],
        inds[3][inds[0] == traj_ind],
    )
    mels_traj_ind = mels[inds[0] == traj_ind]
    dkj_z = np.zeros((sim.model.constants.num_classical_coordinates), dtype=complex)
    dkj_zc = np.zeros((sim.model.constants.num_classical_coordinates), dtype=complex)
    np.add.at(
        dkj_z,
        (inds_traj_ind[1]),
        np.conj(evec_init_state)[inds_traj_ind[2]]
        * mels_traj_ind
        * evec_final_state[inds_traj_ind[3]]
        / ev_diff,
    )
    np.add.at(
        dkj_zc,
        (inds_traj_ind[1]),
        np.conj(evec_init_state)[inds_traj_ind[3]]
        * np.conj(mels_traj_ind)
        * evec_final_state[inds_traj_ind[2]]
        / ev_diff,
    )
    dkj_p = (
        1.0j
        * np.sqrt(
            1
            / (
                2
                * sim.model.constants.classical_coordinate_weight
                * sim.model.constants.classical_coordinate_mass
            )
        )
        * (dkj_z - dkj_zc)
    )
    dkj_q = np.sqrt(
        sim.model.constants.classical_coordinate_weight
        * sim.model.constants.classical_coordinate_mass
        / 2
    ) * (dkj_z + dkj_zc)

    max_pos_q = np.argmax(np.abs(dkj_q))
    max_pos_p = np.argmax(np.abs(dkj_p))
    # Check for complex nonadiabatic couplings.
    if (
        np.abs(dkj_q[max_pos_q]) > 1e-8
        and np.abs(np.sin(np.angle(dkj_q[max_pos_q]))) > 1e-2
    ):
        warnings.warn(
            "dkj_q Nonadiabatic coupling is complex, needs gauge fixing!",
            UserWarning,
        )
    if (
        np.abs(dkj_p[max_pos_p]) > 1e-8
        and np.abs(np.sin(np.angle(dkj_p[max_pos_p]))) > 1e-2
    ):
        warnings.warn(
            "dkj_p Nonadiabatic coupling is complex, needs gauge fixing!",
            UserWarning,
        )
    delta_z = dkj_zc
    return delta_z


def update_active_surface_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the active surface in FSSH. If a hopping function is not specified in the model
    class a numerical hopping procedure is used instead.

    Required constants:
        - None.
    """
    del kwargs
    rand = state.hopping_probs_rand_vals[:, sim.t_ind]
    act_surf_ind = state.act_surf_ind
    act_surf_ind_flat = act_surf_ind.flatten().astype(int)
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    traj_ind = (
        (np.arange(num_trajs)[:, np.newaxis] * np.ones((num_trajs, num_branches)))
        .flatten()
        .astype(int)
    )
    prod = np.einsum(
        "bn,bni->bi",
        np.conj(
            state.eigvecs[
                np.arange(num_trajs * num_branches, dtype=int), :, act_surf_ind_flat
            ]
        ),
        state.eigvecs_previous,
        optimize="greedy",
    )
    hop_prob = -2 * np.real(
        prod
        * state.wf_adb
        / state.wf_adb[np.arange(num_trajs * num_branches), act_surf_ind_flat][
            :, np.newaxis
        ]
    )
    hop_prob[np.arange(num_branches * num_trajs), act_surf_ind_flat] *= 0
    cumulative_probs = np.cumsum(
        np.nan_to_num(hop_prob, nan=0, posinf=100e100, neginf=-100e100), axis=1
    )
    rand_branch = (rand[:, np.newaxis] * np.ones((num_trajs, num_branches))).flatten()
    traj_hop_ind = np.where(
        np.sum((cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1) > 0
    )[0]
    if len(traj_hop_ind) > 0:
        eigvals_flat = state.eigvals
        z = np.copy(state.z)
        act_surf_flat = state.act_surf
        for traj_ind in traj_hop_ind:
            final_state_ind = np.argmax(
                (cumulative_probs[traj_ind] > rand_branch[traj_ind]).astype(int)
            )
            init_state_ind = act_surf_ind_flat[traj_ind]
            eval_init_state = eigvals_flat[traj_ind][init_state_ind]
            eval_final_state = eigvals_flat[traj_ind][final_state_ind]
            ev_diff = eval_final_state - eval_init_state
            delta_z = calc_delta_z_fssh(
                algorithm,
                sim,
                parameters,
                state,
                traj_ind=traj_ind,
                final_state_ind=final_state_ind,
                init_state_ind=init_state_ind,
            )
            hopped = False
            z_out = None
            hop_function, has_hop_function = sim.model.get("hop_function")
            if has_hop_function:
                z_out, hopped = hop_function(
                    sim.model,
                    parameters,
                    z=z[traj_ind],
                    delta_z=delta_z,
                    ev_diff=ev_diff,
                )
            if not has_hop_function:
                z_out, hopped = numerical_fssh_hop(
                    sim.model,
                    parameters,
                    z=z[traj_ind],
                    delta_z=delta_z,
                    ev_diff=ev_diff,
                )
            if hopped:
                act_surf_ind_flat[traj_ind] = final_state_ind
                act_surf_flat[traj_ind] = np.zeros_like(act_surf_flat[traj_ind])
                act_surf_flat[traj_ind][final_state_ind] = 1
                z[traj_ind] = z_out
                state.act_surf_ind = np.copy(
                    act_surf_ind_flat.reshape((num_trajs, num_branches))
                )
                state.act_surf = np.copy(act_surf_flat)
                state.z = np.copy(z)
    parameters.act_surf_ind = state.act_surf_ind
    return parameters, state


def update_h_quantum(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum + quantum-classical Hamiltonian.

    Required constants:
        - None.
    """
    z = kwargs.get("z", state.z)
    h_q, _ = sim.model.get("h_q")
    h_qc, _ = sim.model.get("h_qc")
    state.h_quantum = h_q(sim.model, parameters) + h_qc(sim.model, parameters, z=z)
    return parameters, state



def update_z_rk4(algorithm, sim, parameters, state, **kwargs):
    """
    Update the z-coordinates using the 4th-order Runge-Kutta method.
    If the gradient of the quantum-classical Hamiltonian depends on z then
    update_quantum_classical_forces_bool should be set to True.

    Required constants:
        - None.
    """
    dt = sim.settings.dt
    wf = kwargs["wf"]
    use_gauge_field_force = kwargs.get("use_gauge_field_force", False)
    if hasattr(sim.model, "linear_h_qc"):
        update_quantum_classical_forces_bool = not sim.model.linear_h_qc
    else:
        update_quantum_classical_forces_bool = True
    z_0 = kwargs["z"]
    output_name = kwargs["output_name"]
    parameters, state = update_classical_forces(
        algorithm, sim, parameters, state, z=z_0
    )
    parameters, state = update_quantum_classical_forces(
        algorithm,
        sim,
        parameters,
        state,
        wf=wf,
        z=z_0,
        use_gauge_field_force=use_gauge_field_force,
    )
    k1 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    parameters, state = update_classical_forces(
        algorithm, sim, parameters, state, z=z_0 + 0.5 * dt * k1
    )
    if update_quantum_classical_forces_bool:
        parameters, state = update_quantum_classical_forces(
            algorithm,
            sim,
            parameters,
            state,
            wf=wf,
            z=z_0 + 0.5 * dt * k1,
            use_gauge_field_force=use_gauge_field_force,
        )
    k2 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    parameters, state = update_classical_forces(
        algorithm, sim, parameters, state, z=z_0 + 0.5 * dt * k2
    )
    if update_quantum_classical_forces_bool:
        parameters, state = update_quantum_classical_forces(
            algorithm,
            sim,
            parameters,
            state,
            wf=wf,
            z=z_0 + 0.5 * dt * k2,
            use_gauge_field_force=use_gauge_field_force,
        )
    k3 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    parameters, state = update_classical_forces(
        algorithm, sim, parameters, state, z=z_0 + dt * k3
    )
    if update_quantum_classical_forces_bool:
        parameters, state = update_quantum_classical_forces(
            algorithm,
            sim,
            parameters,
            state,
            wf=wf,
            z=z_0 + dt * k3,
            use_gauge_field_force=use_gauge_field_force,
        )
    k4 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    setattr(state, output_name, z_0 + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4))
    return parameters, state
