"""
This module contains tasks that update the state and parameters objects
during propagation.
"""

import logging
import numpy as np
from qclab import functions
import qclab.numerical_constants as numerical_constants

logger = logging.getLogger(__name__)


def update_t(sim, state, parameters, **kwargs):
    """
    Updates the time in the state object with the time index in each trajectory
    multiplied by the update timestep.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    t_name : str, default: "t"
        Name of the time variable in the state object.

    .. rubric:: Variable Modifications
    state.{t_name} : ndarray
        Time of each trajectory.
    """
    t_name = kwargs.get("t_name", "t")
    batch_size = sim.settings.batch_size
    setattr(
        state,
        t_name,
        np.broadcast_to(sim.t_ind * sim.settings.dt_update, (batch_size,)),
    )
    return state, parameters


def update_dh_c_dzc_finite_differences(sim, state, parameters, **kwargs):
    """
    Calculates the gradient of the classical Hamiltonian using finite differences.

    .. rubric:: Required Constants
    dh_c_dzc_finite_difference_delta : float, default : numerical_constants.FINITE_DIFFERENCE_DELTA
        Finite-difference step size.

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of the classical coordinates in the state object.
    dh_c_dzc_name : str, default: "dh_c_dzc"
        Name under which to store the finite-difference gradient in the state object.

    .. rubric:: Variable Modifications
    state.{dh_c_dzc_name} : ndarray
        Gradient of the classical Hamiltonian.
    """
    z = getattr(state, kwargs.get("z_name", "z"))
    dh_c_dzc_name = kwargs.get("dh_c_dzc_name", "dh_c_dzc")
    delta_z = sim.model.constants.get(
        "dh_c_dzc_finite_difference_delta", numerical_constants.FINITE_DIFFERENCE_DELTA
    )
    batch_size = len(z)
    num_classical_coordinates = sim.model.constants.num_classical_coordinates
    # Calculate increments in the real and imaginary directions.
    dz_re = delta_z * np.eye(num_classical_coordinates, dtype=z.dtype)
    dz_im = 1j * dz_re
    # Stack real/imag offset z coordinates.
    z_offset = z[:, None, :]
    z_offset_all = np.concatenate((z_offset + dz_re, z_offset + dz_im), axis=1).reshape(
        -1, num_classical_coordinates
    )
    # Get the quantum-classical Hamiltonian function.
    h_c, _ = sim.model.get("h_c")
    # Calculate it at the original z coordinate.
    h_c_0 = h_c(sim.model, parameters, z=z)
    # Calculate h_c at the offset coordinates.
    h_c_all = h_c(sim.model, parameters, z=z_offset_all).reshape(
        batch_size, 2 * num_classical_coordinates
    )
    # Split real/imag blocks of the offset h_c.
    h_c_re = h_c_all[:, :num_classical_coordinates]
    h_c_im = h_c_all[:, num_classical_coordinates:]
    # Calculate finite-difference derivatives.
    h_c_0_exp = h_c_0[:, None]
    diff_re = (h_c_re - h_c_0_exp) / delta_z
    diff_im = (h_c_im - h_c_0_exp) / delta_z
    dh_c_dzc = 0.5 * (diff_re + 1j * diff_im)
    setattr(state, dh_c_dzc_name, dh_c_dzc)
    return state, parameters


def update_classical_forces(sim, state, parameters, **kwargs):
    """
    Updates the gradient of the classical Hamiltonian w.r.t. the conjugate classical
    coordinate.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of the classical coordinates in the state object.
    classical_forces_name : str, default: "classical_forces"
        Name under which to store the classical forces in the state object.

    .. rubric:: Variable Modifications
    state.{classical_forces_name} : ndarray
            Gradient of the classical Hamiltonian.
    """
    z_name = kwargs.get("z_name", "z")
    classical_forces_name = kwargs.get("classical_forces_name", "classical_forces")
    z = getattr(state, z_name)
    dh_c_dzc, has_dh_c_dzc = sim.model.get("dh_c_dzc")
    if has_dh_c_dzc:
        setattr(state, classical_forces_name, dh_c_dzc(sim.model, parameters, z=z))
        return state, parameters
    if sim.settings.debug:
        logger.info("dh_c_dzc not found; using finite differences.")
    return update_dh_c_dzc_finite_differences(
        sim, state, parameters, dh_c_dzc_name=classical_forces_name, z_name=z_name
    )


def update_dh_qc_dzc_finite_differences(sim, state, parameters, **kwargs):
    """
    Calculates the gradient of the quantum-classical Hamiltonian using finite
    differences.

    .. rubric:: Required Constants
    dh_qc_dzc_finite_difference_delta : float, default : numerical_constants.FINITE_DIFFERENCE_DELTA
        Finite-difference step size.

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of classical coordinates in the state object.
    dh_qc_dzc_name : str, default: "dh_qc_dzc"
        Name under which to store the gradient of the quantum-classical Hamiltonian in the state object.

    .. rubric:: Variable Modifications
    state.{dh_qc_dzc_name} : tuple
        Gradient of the quantum-classical Hamiltonian.
    """
    z_name = kwargs.get("z_name", "z")
    dh_qc_dzc_name = kwargs.get("dh_qc_dzc_name", "dh_qc_dzc")
    z = getattr(state, z_name)
    batch_size = len(z)
    delta_z = sim.model.constants.get(
        "dh_qc_dzc_finite_difference_delta", numerical_constants.FINITE_DIFFERENCE_DELTA
    )
    num_classical_coordinates = sim.model.constants.num_classical_coordinates
    num_quantum_states = sim.model.constants.num_quantum_states
    # Calculate increments in the real and imaginary directions.
    dz_re = delta_z * np.eye(num_classical_coordinates, dtype=z.dtype)
    dz_im = 1j * dz_re
    # Stack real/imag offset z coordinates.
    z_offset = z[:, None, :]
    z_offset_all = np.concatenate((z_offset + dz_re, z_offset + dz_im), axis=1).reshape(
        -1, num_classical_coordinates
    )
    # Get the quantum-classical Hamiltonian function.
    h_qc, _ = sim.model.get("h_qc")
    # Calculate it at the original z coordinate.
    h_qc_0 = h_qc(sim.model, parameters, z=z)
    # Calculate h_qc at the offset coordinates.
    h_qc_all = h_qc(sim.model, parameters, z=z_offset_all).reshape(
        batch_size,
        2 * num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    # Split real/imag blocks of the offset h_qc.
    h_qc_re = h_qc_all[:, :num_classical_coordinates, :, :]
    h_qc_im = h_qc_all[:, num_classical_coordinates:, :, :]
    # Calculate finite-difference derivatives.
    h_qc_0_exp = h_qc_0[:, None, :, :]
    diff_re = (h_qc_re - h_qc_0_exp) / delta_z
    diff_im = (h_qc_im - h_qc_0_exp) / delta_z
    dh_qc_dzc = 0.5 * (diff_re + 1j * diff_im)
    # Get sparse representation of dh_qc_dzc.
    inds = np.where(dh_qc_dzc != 0)
    mels = dh_qc_dzc[inds]
    shape = np.shape(dh_qc_dzc)
    # Update it in the state object.
    setattr(state, dh_qc_dzc_name, (inds, mels, shape))
    return state, parameters


def update_dh_qc_dzc(sim, state, parameters, **kwargs):
    """
    Updates the gradient of the quantum-classical Hamiltonian w.r.t. the conjugate
    classical coordinate.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of classical coordinates in state object.
    dh_qc_dzc_name : str, default: "dh_qc_dzc"
        Name under which to store the gradient of the quantum-classical Hamiltonian in the state object.

    .. rubric:: Variable Modifications
    state.{dh_qc_dzc_name} : tuple
        Gradient of the quantum-classical Hamiltonian.
    """
    z_name = kwargs.get("z_name", "z")
    dh_qc_dzc_name = kwargs.get("dh_qc_dzc_name", "dh_qc_dzc")
    z = getattr(state, z_name)
    if getattr(state, dh_qc_dzc_name) is None or sim.model.update_dh_qc_dzc:
        # If dh_qc_dzc has not been calculated yet, or if the
        # model requires it to be updated, calculate it.
        dh_qc_dzc, has_dh_qc_dzc = sim.model.get("dh_qc_dzc")
        if has_dh_qc_dzc:
            setattr(state, dh_qc_dzc_name, dh_qc_dzc(sim.model, parameters, z=z))
            return state, parameters
        if sim.settings.debug:
            logger.info("dh_qc_dzc not found; using finite differences.")
        return update_dh_qc_dzc_finite_differences(sim, state, parameters, **kwargs)
    # If dh_qc_dzc has already been calculated and does not need to be updated,
    # return the existing parameters and state objects.
    return state, parameters


def update_quantum_classical_forces(sim, state, parameters, **kwargs):
    """
    Updates the quantum-classical forces w.r.t. the wavefunction defined by ``wf_db``.

    If the model has a ``gauge_field_force`` ingredient, this term will be added
    to the quantum-classical forces.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of classical coordinates in state object.
    wf_db_name : str, default: "wf_db"
        Name of the wavefunction (in the diabatic basis) in the state object.
    dh_qc_dzc_name : str, default: "dh_qc_dzc"
        Name of the gradient of the quantum-classical Hamiltonian in the state object.
    quantum_classical_forces_name : str, default: "quantum_classical_forces"
        Name under which to store the quantum-classical forces in the state object.
    state_ind_name : str, default: "act_surf_ind"
        Name in the state object of the state index for which to obtain the gauge field force.
        Required if ``algorithm.settings.use_gauge_field_force`` is ``True``.
    wf_changed : bool, default: True
        If ``True``, the wavefunction has changed since the last time the forces were calculated.

    .. rubric:: Variable Modifications
    state.{dh_qc_dzc_name} : tuple
        Gradient of the quantum-classical Hamiltonian.
    state.{quantum_classical_forces_name} : ndarray
        Quantum-classical forces.
    """
    z_name = kwargs.get("z_name", "z")
    wf_db_name = kwargs.get("wf_db_name", "wf_db")
    dh_qc_dzc_name = kwargs.get("dh_qc_dzc_name", "dh_qc_dzc")
    quantum_classical_forces_name = kwargs.get(
        "quantum_classical_forces_name", "quantum_classical_forces"
    )
    state_ind_name = kwargs.get("state_ind_name", "act_surf_ind")
    wf_changed = kwargs.get("wf_changed", True)

    z = getattr(state, z_name)
    wf_db = getattr(state, wf_db_name)
    # Update the gradient of h_qc.
    state, parameters = update_dh_qc_dzc(
        sim, state, parameters, z_name=z_name, dh_qc_dzc_name=dh_qc_dzc_name
    )
    # Calculate the expectation value w.r.t. the wavefunction.
    # If not(wf_changed) and sim.model.update_dh_qc_dzc then recalculate.
    # If wf_changed then recalculate.
    # If state.quantum_classical_forces is None then recalculate.
    state_quantum_classical_forces = getattr(state, quantum_classical_forces_name)
    if (
        (state_quantum_classical_forces is None)
        or wf_changed
        or (not (wf_changed) and sim.model.update_dh_qc_dzc)
    ):
        if state_quantum_classical_forces is None:
            setattr(state, quantum_classical_forces_name, np.zeros_like(z))
            state_quantum_classical_forces = getattr(
                state, quantum_classical_forces_name
            )

        state_quantum_classical_forces = functions.calc_sparse_inner_product(
            *state.dh_qc_dzc,
            wf_db.conj(),
            wf_db,
            out=state_quantum_classical_forces.reshape(-1),
        ).reshape(np.shape(z))
    if sim.algorithm.settings.get("use_gauge_field_force"):
        state, parameters = add_gauge_field_force_fssh(
            sim, state, parameters, z=kwargs["z"], state_ind_name=state_ind_name
        )
    return state, parameters


def add_gauge_field_force_fssh(sim, state, parameters, **kwargs):
    """
    Adds the gauge field force to the quantum-classical forces if the model has a
    ``gauge_field_force`` ingredient.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of classical coordinates in state object.
    adb_state_ind_name : str, default: "act_surf_ind"
        Name of the adiabatic state index for which to obtain the gauge field force.
    quantum_classical_forces_name : str, default: "quantum_classical_forces"
        Name of the quantum-classical forces in the state object.

    .. rubric:: Variable Modifications
    state.{quantum_classical_forces_name} : ndarray
        Quantum-classical forces with gauge field force added.
    """
    z_name = kwargs.get("z_name", "z")
    adb_state_ind_name = kwargs.get("adb_state_ind_name", "act_surf_ind")
    quantum_classical_forces_name = kwargs.get(
        "quantum_classical_forces_name", "quantum_classical_forces"
    )
    z = getattr(state, z_name)
    adb_state_ind = getattr(state, adb_state_ind_name)
    gauge_field_force, has_gauge_field_force = sim.model.get("gauge_field_force")
    if has_gauge_field_force:
        gauge_field_force_val = gauge_field_force(
            sim.model, parameters, z=z, state_ind=adb_state_ind
        )
        setattr(
            state,
            quantum_classical_forces_name,
            getattr(state, quantum_classical_forces_name) + gauge_field_force_val,
        )
    else:
        if sim.settings.debug:
            logger.warning("gauge_field_force not found; skipping.")
    return state, parameters


def diagonalize_matrix(sim, state, parameters, **kwargs):
    """
    Diagonalizes a given matrix from the state object and stores the eigenvalues and
    eigenvectors in the state object.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    matrix_name : str
        Name of the matrix to diagonalize in the state object.
    eigvals_name : str
        Name of the eigenvalues in the state object.
    eigvecs_name : str
        Name of the eigenvectors in the state object.

    .. rubric:: Variable Modifications
    state.{eigvals} : ndarray
        Eigenvalues of the matrix.
    state.{eigvecs} : ndarray
        Eigenvectors of the matrix.
    """
    matrix = getattr(state, kwargs["matrix_name"])
    eigvals, eigvecs = np.linalg.eigh(matrix)
    setattr(state, kwargs["eigvals_name"], eigvals)
    setattr(state, kwargs["eigvecs_name"], eigvecs)
    return state, parameters


def gauge_fix_eigs(sim, state, parameters, **kwargs):
    """
    Fixes the gauge of the eigenvectors as specified by the gauge_fixing parameter.


    if gauge_fixing == "sign_overlap":
        The sign of the eigenvector is changed so the real part of its
        overlap with the previous eigenvector is positive.

    if gauge_fixing == "phase_overlap":
        The phase of the eigenvector is determined from its overlap
        with the previous eigenvector and used to maximize the real
        part of the overlap. The sign is then changed so the real
        part of the overlap is positive.

    if gauge_fixing == "phase_der_couple":
        The phase of the eigenvector is determined by calculating the
        derivative couplings and changed so that all the derivative
        couplings are real-valued.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    eigvals_name : str, default: "eigvals"
        Name of the eigenvalues in the state object.
    eigvecs_name : str, default: "eigvecs"
        Name of the eigenvectors in the state object.
    eigvecs_previous_name : str, default: "eigvecs_previous"
        Name of the previous eigenvectors in the state object.
    output_eigvecs_name : str, default: eigvecs_name
        Name of the output gauge-fixed eigenvectors in the state object.
    z_name : str, default: "z"
        Name of classical coordinates in the state object.
    gauge_fixing : str, default: sim.algorithm.settings.gauge_fixing
        Gauge-fixing method to use.

    .. rubric:: Variable Modifications
    state.{output_eigvecs_name} : ndarray
        Gauge-fixed eigenvectors.
    """
    eigvals_name = kwargs.get("eigvals_name", "eigvals")
    eigvecs_name = kwargs.get("eigvecs_name", "eigvecs")
    eigvecs_previous_name = kwargs.get("eigvecs_previous_name", "eigvecs_previous")
    output_eigvecs_name = kwargs.get("output_eigvecs_name", eigvecs_name)
    z_name = kwargs.get("z_name", "z")
    eigvals = getattr(state, eigvals_name)
    eigvecs = getattr(state, eigvecs_name)
    eigvecs_previous = getattr(state, eigvecs_previous_name)
    gauge_fixing = kwargs.get("gauge_fixing", sim.algorithm.settings.gauge_fixing)
    gauge_fixing_numerical_values = {
        "sign_overlap": 0,
        "phase_overlap": 1,
        "phase_der_couple": 2,
    }
    gauge_fixing_value = gauge_fixing_numerical_values[gauge_fixing]
    if gauge_fixing_value not in {0, 1, 2}:
        logger.critical("Invalid gauge_fixing value: %s", gauge_fixing)
        raise ValueError(f"Invalid gauge_fixing value: {gauge_fixing}")
    if gauge_fixing_value == 1:
        # Maximize the real part of the overlap (guaranteed to be positive).
        overlap = np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)
        phase = np.exp(-1j * np.angle(overlap))
        eigvecs *= phase[:, None, :]
    if gauge_fixing_value == 2:
        # Make the derivative couplings real-valued (but not necessarily positive).
        state, parameters = update_dh_qc_dzc(sim, state, parameters, z_name=z_name)
        der_couple_q_phase, _ = functions.analytic_der_couple_phase(
            sim, state.dh_qc_dzc, eigvals, eigvecs
        )
        eigvecs *= np.conj(der_couple_q_phase)[:, None, :]
    if gauge_fixing_value == 0 or gauge_fixing_value == 2:
        # Make the real part positive based on the sign of the real part of the overlap.
        overlap = np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)
        signs = np.sign(np.real(overlap))
        if sim.settings.debug:
            if np.any(np.abs(signs) < numerical_constants.SMALL):
                logger.error(
                    "Zero overlap encountered when fixing gauge.\n"
                    "This may indicate a trivial crossing or degeneracy.\n"
                    "Normalization will be broken and results will be incorrect."
                )
        eigvecs *= signs[:, None, :]

    if gauge_fixing_value == 2 and sim.settings.debug:
        der_couple_q_phase_new, der_couple_p_phase_new = (
            functions.analytic_der_couple_phase(sim, state.dh_qc_dzc, eigvals, eigvecs)
        )
        if (
            np.sum(
                np.abs(np.imag(der_couple_q_phase_new)) ** 2
                + np.abs(np.imag(der_couple_p_phase_new)) ** 2
            )
            > numerical_constants.SMALL
        ):
            logger.error(
                "Phase error encountered when fixing gauge analytically. %s",
                np.sum(
                    np.abs(np.imag(der_couple_q_phase_new)) ** 2
                    + np.abs(np.imag(der_couple_p_phase_new)) ** 2
                ),
            )
    setattr(state, output_eigvecs_name, eigvecs)
    return state, parameters


def basis_transform_vec(sim, state, parameters, **kwargs):
    """
    Transforms a vector ``state.{input}`` to a new basis defined by
    ``state.{basis}`` and stores it in ``state.{output}``.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    input_vec_name : str
        Name of the vector to transform in the state object.
    basis_name : str
        Name of the basis to transform to in the state object.
        Assumed to be column vectors corresponding to adiabatic
        states.
    output_vec_name : str
        Name of the output vector in the state object.
    adb_to_db : bool, default: False
        If True, transforms from adiabatic to diabatic. If False, transforms from
        adiabatic to diabatic.

    .. rubric:: Variable Modifications
    state.{output_vec_name} : ndarray
        Vector expressed in the new basis.
    """
    input_vec = getattr(state, kwargs["input_vec_name"])
    basis = getattr(state, kwargs["basis_name"])
    adb_to_db = kwargs["adb_to_db"]
    setattr(
        state,
        kwargs["output_vec_name"],
        functions.transform_vec(input_vec, basis, adb_to_db=adb_to_db),
    )
    return state, parameters


def update_act_surf_wf(sim, state, parameters, **kwargs):
    """
    Updates the wavefunction corresponding to the active surface.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    act_surf_wf_name : str, default: "act_surf_wf"
        Name of the active surface wavefunction in the state object.
    act_surf_ind_name : str, default: "act_surf_ind"
        Name of the active surface index in the state object.
    eigvecs_name : str, default: "eigvecs"
        Name of the eigenvectors in the state object.

    .. rubric:: Variable Modifications
    state.{act_surf_wf_name} : ndarray
        Wavefunction of the active surface.
    """
    act_surf_wf_name = kwargs.get("act_surf_wf_name", "act_surf_wf")
    act_surf_ind_name = kwargs.get("act_surf_ind_name", "act_surf_ind")
    eigvecs_name = kwargs.get("eigvecs_name", "eigvecs")
    num_trajs = sim.settings.batch_size
    act_surf_wf = getattr(state, eigvecs_name)[
        np.arange(num_trajs, dtype=int),
        :,
        getattr(state, act_surf_ind_name),
    ]
    setattr(state, act_surf_wf_name, act_surf_wf)
    return state, parameters


def update_wf_db_eigs(sim, state, parameters, **kwargs):
    """
    Evolves the diabatic wavefunction ``state.{wf_db_name}`` using the
    eigenvalues ``state.{eigvals_name}`` and eigenvectors ``state.{eigvecs_name}```.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    wf_db_name : str, default: "wf_db"
        Name of the diabatic wavefunction in the state object.
    eigvals_name : str, default: "eigvals"
        Name of the eigenvalues in the state object.
    eigvecs_name : str, default: "eigvecs"
        Name of the eigenvectors in the state object.

    .. rubric:: Variable Modifications
    state.{wf_db_name} : ndarray
        Updated diabatic wavefunction.
    """
    wf_db_name = kwargs.get("wf_db_name", "wf_db")
    eigvals_name = kwargs.get("eigvals_name", "eigvals")
    eigvecs_name = kwargs.get("eigvecs_name", "eigvecs")
    wf_db = getattr(state, wf_db_name)
    eigvals = getattr(state, eigvals_name)
    eigvecs = getattr(state, eigvecs_name)
    batch_size = sim.settings.batch_size
    num_quantum_states = sim.model.constants.num_quantum_states
    # Calculate the propagator in the adiabatic basis.
    prop_adb = np.zeros(
        (batch_size, num_quantum_states, num_quantum_states), dtype=complex
    )
    idx = np.arange(num_quantum_states)
    prop_adb[:, idx, idx] = np.exp(-1j * eigvals * sim.settings.dt_update)
    # Transform propagator to the diabatic basis.
    prop_db = functions.transform_mat(prop_adb, eigvecs, adb_to_db=True)
    # Apply the propagator to the diabatic wavefunction.
    setattr(
        state,
        wf_db_name,
        functions.multiply_matrix_vector(prop_db, wf_db),
    )
    return state, parameters


def update_wf_db_rk4(sim, state, parameters, **kwargs):
    """
    Updates the wavefunction ``state.{wf_db_name}`` using the 4th-order Runge-Kutta method
    with the hamiltonian ``state.{h_quantum_name}``.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    wf_db_name : str, default: "wf_db"
        Name of the diabatic wavefunction in the state object.
    h_quantum_name : str, default: "h_quantum"
        Name of the quantum Hamiltonian in the state object.

    .. rubric:: Variable Modifications
    state.{wf_db_name} : ndarray
        Updated diabatic wavefunction.
    """
    dt_update = sim.settings.dt_update
    wf_db_name = kwargs.get("wf_db_name", "wf_db")
    h_quantum_name = kwargs.get("h_quantum_name", "h_quantum")
    wf_db = getattr(state, wf_db_name)
    h_quantum = getattr(state, h_quantum_name)
    setattr(state, wf_db_name, functions.wf_db_rk4(h_quantum, wf_db, dt_update))
    return state, parameters


def update_hop_probs_fssh(sim, state, parameters, **kwargs):
    """
    Calculates the hopping probabilities for FSSH.

    :math:`P_{a\\rightarrow b} = -2\\Re((C_{b}/C_{a})\\langle a(t)| b(t-dt)\\rangle)`

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    act_surf_ind_name : str, default: "act_surf_ind"
        Name of the active surface index in the state object.
    wf_adb_name : str, default: "wf_adb"
        Name of the adiabatic wavefunction in the state object.
    eigvecs_name : str, default: "eigvecs"
        Name of the eigenvectors in the state object.
    eigvecs_previous_name : str, default: "eigvecs_previous"
        Name of the previous eigenvectors in the state object.
    hop_prob_name : str, default: "hop_prob"
        Name under which to store the hopping probabilities in the state object.

    .. rubric:: Variable Modifications
    state.{hop_prob_name} : ndarray
        Hopping probabilities between the active surface and all other surfaces.
    """
    act_surf_ind_name = kwargs.get("act_surf_ind_name", "act_surf_ind")
    wf_adb_name = kwargs.get("wf_adb_name", "wf_adb")
    eigvecs_name = kwargs.get("eigvecs_name", "eigvecs")
    eigvecs_previous_name = kwargs.get("eigvecs_previous_name", "eigvecs_previous")
    hop_prob_name = kwargs.get("hop_prob_name", "hop_prob")
    act_surf_ind = getattr(state, act_surf_ind_name)
    wf_adb = getattr(state, wf_adb_name)
    eigvecs = getattr(state, eigvecs_name)
    eigvecs_previous = getattr(state, eigvecs_previous_name)
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    # Check if any of the coefficients on the active surface are zero.
    if sim.settings.debug:
        if np.any(
            np.abs(wf_adb[np.arange(num_trajs * num_branches), act_surf_ind]) == 0.0
        ):
            logger.warning(
                "Zero coefficient on active surface encountered when calculating hopping probabilities."
            )
    # Calculates < act_surf(t) | b(t-dt) >
    prod = functions.multiply_matrix_vector(
        np.swapaxes(eigvecs_previous, -1, -2),
        np.conj(
            eigvecs[np.arange(num_trajs * num_branches, dtype=int), :, act_surf_ind]
        ),
    )
    # Calculates -2*Re( (C_b / C_act_surf) * < act_surf(t) | b(t-dt) > )
    hop_prob = -2.0 * np.real(
        prod
        * wf_adb
        / wf_adb[np.arange(num_trajs * num_branches), act_surf_ind][:, np.newaxis]
    )
    # Sets hopping probabilities to 0 at the active surface.
    hop_prob[np.arange(num_branches * num_trajs), act_surf_ind] = 0.0
    setattr(state, hop_prob_name, hop_prob)
    return state, parameters


def update_hop_inds_fssh(sim, state, parameters, **kwargs):
    """
    Determines which trajectories hop based on the hopping probabilities and which state
    they hop to. Note that these will only hop if they are not frustrated by the hopping
    function.

    Stores the indices of the hopping trajectories in ``state.hop_ind``. Stores the
    destination indices of the hops in ``state.hop_dest``.

    .. rubric:: Required Constants
    hop_prob_name : str, default: "hop_prob"
        Name of the hopping probabilities in the state object.
    hopping_probs_rand_vals_name : str, default: "hopping_probs_rand_vals"
        Name of the random values for hopping probabilities in the state object.
    hop_ind_name : str, default: "hop_ind"
        Name under which to store the indices of the hopping trajectories in the state object.
    hop_dest_name : str, default: "hop_dest"
        Name under which to store the destination indices of the hopping trajectories in the state object.

    .. rubric:: Keyword Arguments
    None

    .. rubric:: Variable Modifications
    state.{hop_ind_name} : ndarray
        Indices of trajectories that hop.
    state.{hop_dest_name} : ndarray
        Destination surface for each hop.
    """
    hop_prob_name = kwargs.get("hop_prob_name", "hop_prob")
    hopping_probs_rand_vals_name = kwargs.get(
        "hopping_probs_rand_vals_name", "hopping_probs_rand_vals"
    )
    hop_ind_name = kwargs.get("hop_ind_name", "hop_ind")
    hop_dest_name = kwargs.get("hop_dest_name", "hop_dest")
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    hop_prob = getattr(state, hop_prob_name)
    rand = getattr(state, hopping_probs_rand_vals_name)[:, sim.t_ind]
    cumulative_probs = np.cumsum(
        np.nan_to_num(hop_prob, nan=0, posinf=100e100, neginf=-100e100, copy=False),
        axis=1,
    )
    rand_branch = (rand[:, np.newaxis] * np.ones((num_trajs, num_branches))).flatten()
    hop_ind = np.where(
        np.sum((cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1) > 0
    )[0]
    hop_dest = np.argmax(
        (cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1
    )[hop_ind]
    setattr(state, hop_ind_name, hop_ind)
    setattr(state, hop_dest_name, hop_dest)
    return state, parameters


def update_z_shift_fssh(sim, state, parameters, **kwargs):
    """
    Determines if a hop occurs and calculates the shift in the classical coordinate
    at the single trajectory level.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_traj_name : str, default: "z_traj"
        Name of the classical coordinates for this trajectory in the state object.
    delta_z_traj_name : str, default: "delta_z_traj"
        Name of the rescaling direction for this trajectory in the state object.
    eigval_diff_traj_name : str, default: "eigval_diff_traj"
        Name of the difference in eigenvalues between the initial and final states
        ``(e_final - e_initial)`` for this trajectory in the state object.
    hop_successful_traj_name : str, default: "hop_successful_traj"
        Name under which to store whether the hop was successful for this trajectory in the state object.
    z_shift_traj_name : str, default: "z_shift_traj"
        Name under which to store the shift in classical coordinates for this trajectory in the state object.

    .. rubric:: Variable Modifications
    state.{hop_successful_traj_name} : bool
        Flag indicating if the hop was successful.
    state.{z_shift_traj_name} : ndarray
        Shift required to conserve energy.
    """
    z_traj_name = kwargs.get("z_traj_name", "z_traj")
    delta_z_traj_name = kwargs.get("delta_z_traj_name", "delta_z_traj")
    eigval_diff_traj_name = kwargs.get("eigval_diff_traj_name", "eigval_diff_traj")
    hop_successful_traj_name = kwargs.get(
        "hop_successful_traj_name", "hop_successful_traj"
    )
    z_shift_traj_name = kwargs.get("z_shift_traj_name", "z_shift_traj")
    z = getattr(state, z_traj_name)
    delta_z = getattr(state, delta_z_traj_name)
    eigval_diff = getattr(state, eigval_diff_traj_name)
    hop, has_hop = sim.model.get("hop")
    if has_hop:
        z_shift, hopped = hop(
            sim.model,
            parameters,
            z=z,
            delta_z=delta_z,
            eigval_diff=eigval_diff,
        )
    else:
        z_shift, hopped = functions.numerical_fssh_hop(
            sim.model,
            parameters,
            z=z,
            delta_z=delta_z,
            eigval_diff=eigval_diff,
        )
    setattr(state, hop_successful_traj_name, hopped)
    setattr(state, z_shift_traj_name, z_shift)
    return state, parameters


def update_hop_vals_fssh(sim, state, parameters, **kwargs):
    """
    Executes the hopping function for the hopping trajectories.

    Stores the rescaled coordinates in ``state.z_rescaled`` and a Boolean registering
    if the hop was successful in ``state.hop_successful``.

    If the model has a ``rescaling_direction_fssh`` ingredient, it will be used to
    determine the direction in which to rescale the coordinates. Otherwise, the
    direction will be calculated with ``functions.calc_delta_z_fssh``.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_shift_name : str, default: "z_shift"
        Name under which to store the shift in coordinates for each hopping trajectory in the state object.
    hop_successful_name : str, default: "hop_successful"
        Name under which to store flags indicating if each hop was successful in the state object.
    hop_ind_name: str, default: "hop_ind"
        Name of the indices of the trajectories that are attempting to hop in the state object.
    hop_dest_name: str, default: "hop_dest"
        Name of the destination states of the trajectories that are attempting to hop in the state object.
    eigvals_name : str, default: "eigvals"
        Name of the eigenvalues in the state object.
    eigvecs_name : str, default: "eigvecs"
        Name of the eigenvectors in the state object.
    z_name : str, default: "z"
        Name of classical coordinates in the state object.
    act_surf_ind_name : str, default: "act_surf_ind"
        Name of the active surface index in the state object.
    dh_qc_dzc_name : str, default: "dh_qc_dzc"
        Name of the gradient of the quantum-classical Hamiltonian in the state object.
    z_traj_name : str, default: "z_traj"
        Name of the classical coordinates for the intermediate hopping trajectory in the state object.
    delta_z_traj_name : str, default: "delta_z_traj"
        Name of the rescaling direction for the intermediate hopping trajectory in the state object.
    eigval_diff_traj_name : str, default: "eigval_diff_traj"
        Name of the difference in eigenvalues between the initial and final states
        ``(e_final - e_initial)`` for the intermediate hopping trajectory in the state object.
    hop_successful_traj_name : str, default: "hop_successful_traj"
        Name under which to store whether the hop was successful for the intermediate hopping trajectory in the state object.
    z_shift_traj_name : str, default: "z_shift_traj"
        Name under which to store the shift in classical coordinates for the intermediate hopping trajectory in the state object.

    .. rubric:: Variable Modifications
    state.{z_shift_name} : ndarray
        Shift in coordinates for each hopping trajectory.
    state.{hop_successful_name} : ndarray
        Flags indicating if each hop was successful.
    """
    z_shift_name = kwargs.get("z_shift_name", "z_shift")
    hop_successful_name = kwargs.get("hop_successful_name", "hop_successful")
    eigvals_name = kwargs.get("eigvals_name", "eigvals")
    eigvecs_name = kwargs.get("eigvecs_name", "eigvecs")
    z_name = kwargs.get("z_name", "z")
    act_surf_ind_name = kwargs.get("act_surf_ind_name", "act_surf_ind")
    hop_ind_name = kwargs.get("hop_ind_name", "hop_ind")
    hop_dest_name = kwargs.get("hop_dest_name", "hop_dest")
    dh_qc_dzc_name = kwargs.get("dh_qc_dzc_name", "dh_qc_dzc")
    z_traj_name = kwargs.get("z_traj_name", "z_traj")
    delta_z_traj_name = kwargs.get("delta_z_traj_name", "delta_z_traj")
    eigval_diff_traj_name = kwargs.get("eigval_diff_traj_name", "eigval_diff_traj")
    hop_successful_traj_name = kwargs.get(
        "hop_successful_traj_name", "hop_successful_traj"
    )
    z_shift_traj_name = kwargs.get("z_shift_traj_name", "z_shift_traj")
    hop_ind = getattr(state, hop_ind_name)
    hop_dest = getattr(state, hop_dest_name)
    setattr(
        state,
        z_shift_name,
        np.zeros(
            (len(hop_ind), sim.model.constants.num_classical_coordinates), dtype=complex
        ),
    )
    setattr(state, hop_successful_name, np.zeros(len(hop_ind), dtype=bool))
    eigvals = getattr(state, eigvals_name)
    eigvecs = getattr(state, eigvecs_name)
    z = getattr(state, z_name)
    act_surf_ind = getattr(state, act_surf_ind_name)

    state_hop_successful = getattr(state, hop_successful_name)
    state_z_shift = getattr(state, z_shift_name)

    hop_traj_ind = 0
    for traj_ind in hop_ind:
        final_state_ind = hop_dest[hop_traj_ind]
        init_state_ind = act_surf_ind[traj_ind]
        eigval_init_state = eigvals[traj_ind][init_state_ind]
        eigval_final_state = eigvals[traj_ind][final_state_ind]
        eigval_diff = eigval_final_state - eigval_init_state
        eigvec_init_state = eigvecs[traj_ind, :, init_state_ind]
        eigvec_final_state = eigvecs[traj_ind, :, final_state_ind]
        rescaling_direction_fssh, has_rescaling_direction_fssh = sim.model.get(
            "rescaling_direction_fssh"
        )
        if has_rescaling_direction_fssh:
            delta_z = rescaling_direction_fssh(
                parameters,
                z=z[traj_ind],
                init_state_ind=init_state_ind,
                final_state_ind=final_state_ind,
            )
        else:
            inds, mels, shape = getattr(state, dh_qc_dzc_name)
            dh_qc_dzc_traj_ind = inds[0] == traj_ind
            inds_traj = (
                inds[0][dh_qc_dzc_traj_ind],
                inds[1][dh_qc_dzc_traj_ind],
                inds[2][dh_qc_dzc_traj_ind],
                inds[3][dh_qc_dzc_traj_ind],
            )
            mels_traj = mels[dh_qc_dzc_traj_ind]
            shape_traj = (1, shape[1], shape[2], shape[3])
            dh_qc_dzc_traj = (inds_traj, mels_traj, shape_traj)
            delta_z = functions.calc_delta_z_fssh(
                sim, eigval_diff, eigvec_init_state, eigvec_final_state, dh_qc_dzc_traj
            )
        setattr(state, z_traj_name, z[traj_ind])
        setattr(state, delta_z_traj_name, delta_z)
        setattr(state, eigval_diff_traj_name, eigval_diff)
        state, parameters = update_z_shift_fssh(sim, state, parameters, **kwargs)
        state_hop_successful[hop_traj_ind] = getattr(state, hop_successful_traj_name)
        state_z_shift[hop_traj_ind] = getattr(state, z_shift_traj_name)
        hop_traj_ind += 1
    return state, parameters


def update_z_hop_fssh(sim, state, parameters, **kwargs):
    """
    Applies the shift in coordinates after successful hops.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_shift_name : str, default: "z_shift"
        Name of the shift in coordinates for each hopping trajectory in the state object.
    hop_ind_name : str, default: "hop_ind"
        Name of the indices of the trajectories that are attempting to hop in the state object.
    z_name : str, default: "z"
        Name of classical coordinates in the state object.

    .. rubric:: Variable Modifications
    state.{z_name} : ndarray
        Classical coordinates.
    """
    z_shift_name = kwargs.get("z_shift_name", "z_shift")
    hop_ind_name = kwargs.get("hop_ind_name", "hop_ind")
    z_name = kwargs.get("z_name", "z")
    z_orig = getattr(state, z_name)
    hop_ind = getattr(state, hop_ind_name)
    z_shift = getattr(state, z_shift_name)
    z_orig[hop_ind] += z_shift
    setattr(state, z_name, z_orig)
    return state, parameters


def update_act_surf_hop_fssh(sim, state, parameters, **kwargs):
    """
    Updates the active surface, active surface index, and active surface wavefunction
    following a hop in FSSH.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    hop_ind_name : str, default: "hop_ind"
        Name of the indices of the trajectories that are attempting to hop in the state object.
    hop_dest_name : str, default: "hop_dest"
        Name of the destination states of the trajectories that are attempting to hop in the state object.
    hop_successful_name : str, default: "hop_successful"
        Name of the flags indicating if each hop was successful in the state object.
    act_surf_name : str, default: "act_surf"
        Name of the active surface wavefunction in the state object.
    act_surf_ind_name : str, default: "act_surf_ind"
        Name of the active surface index in the state object.

    .. rubric:: Variable Modifications
    state.{act_surf_ind_name} : ndarray
        Active surface indices.
    state.{act_surf_name} : ndarray
        Active surface wavefunctions.
    """
    hop_ind_name = kwargs.get("hop_ind_name", "hop_ind")
    hop_dest_name = kwargs.get("hop_dest_name", "hop_dest")
    hop_successful_name = kwargs.get("hop_successful_name", "hop_successful")
    act_surf_name = kwargs.get("act_surf_name", "act_surf")
    act_surf_ind_name = kwargs.get("act_surf_ind_name", "act_surf_ind")
    hop_successful = getattr(state, hop_successful_name)
    hop_ind = getattr(state, hop_ind_name)
    hop_dest = getattr(state, hop_dest_name)
    act_surf = getattr(state, act_surf_name)
    act_surf_ind = getattr(state, act_surf_ind_name)
    # Get the index of the trajectories that successfully hopped.
    hop_successful_traj_ind = hop_ind[hop_successful]
    # Get their destination states.
    hop_dest_ind = hop_dest[hop_successful]
    # Zero out the active surface in the ones that hopped.
    act_surf[hop_successful_traj_ind] = 0
    # Set the new active surface to 1.
    act_surf[hop_successful_traj_ind, hop_dest_ind] = 1
    # Update the active surface index.
    act_surf_ind[hop_successful_traj_ind] = hop_dest_ind

    return state, parameters


def update_h_quantum(sim, state, parameters, **kwargs):
    """
    Updates the Hamiltonian matrix of the quantum subsystem.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of classical coordinates in state object.
    h_q_name : str, default: "h_q"
        Name of the quantum Hamiltonian in the state object.
    h_qc_name : str, default: "h_qc"
        Name of the quantum-classical coupling Hamiltonian in the state object.
    h_quantum_name : str, default: "h_quantum"
        Name of the total Hamiltonian of the quantum subsystem in the state object.
        (``h_q + h_qc``)


    .. rubric:: Variable Modifications
    state.{h_q_name} : ndarray
        Quantum Hamiltonian matrix.
    state.{h_qc_name} : ndarray
        Quantum-classical coupling matrix.
    state.{h_quantum_name} : ndarray
        Total Hamiltonian of the quantum subsystem.
    """
    z_name = kwargs.get("z_name", "z")
    h_q_name = kwargs.get("h_q_name", "h_q")
    h_qc_name = kwargs.get("h_qc_name", "h_qc")
    h_quantum_name = kwargs.get("h_quantum_name", "h_quantum")
    z = getattr(state, z_name)
    h_q, _ = sim.model.get("h_q")
    h_qc, _ = sim.model.get("h_qc")
    state_h_q = getattr(state, h_q_name)
    if sim.model.update_h_q or state_h_q is None:
        # Update the quantum Hamiltonian if required or if it is not set.
        setattr(
            state,
            h_q_name,
            h_q(sim.model, parameters, batch_size=sim.settings.batch_size),
        )
        state_h_q = getattr(state, h_q_name)
    # Update the quantum-classical Hamiltonian.
    setattr(state, h_qc_name, h_qc(sim.model, parameters, z=z))
    # Update the total Hamiltonian of the quantum subsystem.
    setattr(state, h_quantum_name, state_h_q + getattr(state, h_qc_name))
    return state, parameters


def update_z_rk4_k1(sim, state, parameters, **kwargs):
    """
    Computes the first RK4 intermediate for evolving the classical coordinates.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of input coordinates in state object.
    z_output_name : str, default: "z_1"
        Name of the output coordinates in the state object.
    k1_name : str, default: "z_rk4_k1"
        Name of the first RK4 slope in the state object.
    classical_forces_name : str, default: "classical_forces"
        Name of the classical forces in the state object.
    quantum_classical_forces_name : str, default: "quantum_classical_forces"
        Name of the quantum-classical forces in the state object.

    .. rubric:: Variable Modifications
    state.{z_output_name} : ndarray
        Output coordinates after half step.
    state.z_rk4_k1 : ndarray
        First RK4 slope.
    """
    z_name = kwargs.get("z_name", "z")
    z_output_name = kwargs.get("z_output_name", "z_1")
    k1_name = kwargs.get("k1_name", "z_rk4_k1")
    classical_forces_name = kwargs.get("classical_forces_name", "classical_forces")
    quantum_classical_forces_name = kwargs.get(
        "quantum_classical_forces_name", "quantum_classical_forces"
    )
    classical_forces = getattr(state, classical_forces_name)
    quantum_classical_forces = getattr(state, quantum_classical_forces_name)
    dt_update = sim.settings.dt_update
    z_k = getattr(state, z_name)
    out, k1 = functions.update_z_rk4_k123_sum(
        z_k, classical_forces, quantum_classical_forces, 0.5 * dt_update
    )
    setattr(state, z_output_name, out)
    setattr(state, k1_name, k1)
    return state, parameters


def update_z_rk4_k2(sim, state, parameters, **kwargs):
    """
    Computes the second RK4 intermediate for evolving the classical coordinates.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of input coordinates in state object.
    z_output_name : str, default: "z_2"
        Name of the output coordinates in the state object.
    k2_name : str, default: "z_rk4_k2"
        Name of the second RK4 slope in the state object.
    classical_forces_name : str, default: "classical_forces"
        Name of the classical forces in the state object.
    quantum_classical_forces_name : str, default: "quantum_classical_forces"
        Name of the quantum-classical forces in the state object.

    .. rubric:: Variable Modifications
    state.{z_output_name} : ndarray
        Output coordinates after half step.
    state.{k2_name} : ndarray
        Second RK4 slope.
    """
    z_name = kwargs.get("z_name", "z")
    z_output_name = kwargs.get("z_output_name", "z_2")
    k2_name = kwargs.get("k2_name", "z_rk4_k2")
    classical_forces_name = kwargs.get("classical_forces_name", "classical_forces")
    quantum_classical_forces_name = kwargs.get(
        "quantum_classical_forces_name", "quantum_classical_forces"
    )
    classical_forces = getattr(state, classical_forces_name)
    quantum_classical_forces = getattr(state, quantum_classical_forces_name)
    dt_update = sim.settings.dt_update
    z_k = getattr(state, z_name)
    out, k2 = functions.update_z_rk4_k123_sum(
        z_k, classical_forces, quantum_classical_forces, 0.5 * dt_update
    )
    setattr(state, z_output_name, out)
    setattr(state, k2_name, k2)
    return state, parameters


def update_z_rk4_k3(sim, state, parameters, **kwargs):
    """
    Computes the third RK4 intermediate for evolving the classical coordinates.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of input coordinates in state object.
    z_output_name : str, default: "z_3"
        Name of the output coordinates in the state object.
    k3_name : str, default: "z_rk4_k3"
        Name of the third RK4 slope in the state object.
    classical_forces_name : str, default: "classical_forces"
        Name of the classical forces in the state object.
    quantum_classical_forces_name : str, default: "quantum_classical_forces"
        Name of the quantum-classical forces in the state object.

    .. rubric:: Variable Modifications
    state.{z_output_name} : ndarray
        Output coordinates after a full step.
    state.{k3_name} : ndarray
        Third RK4 slope.
    """
    z_name = kwargs.get("z_name", "z")
    z_output_name = kwargs.get("z_output_name", "z_3")
    k3_name = kwargs.get("k3_name", "z_rk4_k3")
    classical_forces_name = kwargs.get("classical_forces_name", "classical_forces")
    quantum_classical_forces_name = kwargs.get(
        "quantum_classical_forces_name", "quantum_classical_forces"
    )
    classical_forces = getattr(state, classical_forces_name)
    quantum_classical_forces = getattr(state, quantum_classical_forces_name)
    dt_update = sim.settings.dt_update
    z_k = getattr(state, z_name)
    out, k3 = functions.update_z_rk4_k123_sum(
        z_k, classical_forces, quantum_classical_forces, dt_update
    )
    setattr(state, z_output_name, out)
    setattr(state, k3_name, k3)
    return state, parameters


def update_z_rk4_k4(sim, state, parameters, **kwargs):
    """
    Computes the final RK4 update for evolving the classical coordinates.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
        Name of input coordinates in state object.
    z_output_name : str, default: z_name
        Name of the output coordinates in the state object.
    k1_name : str, default: "z_rk4_k1"
        Name of the first RK4 slope in the state object.
    k2_name : str, default: "z_rk4_k2"
        Name of the second RK4 slope in the state object.
    k3_name : str, default: "z_rk4_k3"
        Name of the third RK4 slope in the state object.
    classical_forces_name : str, default: "classical_forces"
        Name of the classical forces in the state object.
    quantum_classical_forces_name : str, default: "quantum_classical_forces"
        Name of the quantum-classical forces in the state object.

    .. rubric:: Variable Modifications
    state.{z_output_name} : ndarray
        Output coordinates after a full step.
    """
    z_name = kwargs.get("z_name", "z")
    z_output_name = kwargs.get("z_output_name", z_name)
    k1_name = kwargs.get("k1_name", "z_rk4_k1")
    k2_name = kwargs.get("k2_name", "z_rk4_k2")
    k3_name = kwargs.get("k3_name", "z_rk4_k3")
    k1 = getattr(state, k1_name)
    k2 = getattr(state, k2_name)
    k3 = getattr(state, k3_name)
    classical_forces_name = kwargs.get("classical_forces_name", "classical_forces")
    quantum_classical_forces_name = kwargs.get(
        "quantum_classical_forces_name", "quantum_classical_forces"
    )
    classical_forces = getattr(state, classical_forces_name)
    quantum_classical_forces = getattr(state, quantum_classical_forces_name)
    dt_update = sim.settings.dt_update
    z_0 = getattr(state, z_name)
    out = functions.update_z_rk4_k4_sum(
        z_0,
        k1,
        k2,
        k3,
        classical_forces,
        quantum_classical_forces,
        dt_update,
    )
    setattr(state, z_output_name, out)
    return state, parameters


def update_dm_db_mf(sim, state, parameters, **kwargs):
    """
    Updates the diabatic density matrix based on the wavefunction.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    wf_db_name : str, default: "wf_db"
        Name of the diabatic wavefunction in the state object.
    dm_db_name : str, default: "dm_db"
        Name of the diabatic density matrix in the state object.

    .. rubric:: Variable Modifications
    state.{dm_db_name} : ndarray
        Diabatic density matrix.
    """
    wf_db_name = kwargs.get("wf_db_name", "wf_db")
    dm_db_name = kwargs.get("dm_db_name", "dm_db")
    wf_db = getattr(state, wf_db_name)
    setattr(
        state,
        dm_db_name,
        np.einsum("ti,tj->tij", wf_db, np.conj(wf_db), optimize="greedy"),
    )
    return state, parameters


def update_classical_energy(sim, state, parameters, **kwargs):
    """
    Updates the classical energy.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name : str, default: "z"
    classical_energy_name : str, default: "classical_energy"
        Name under which to store the classical energy in the state object.

    .. rubric:: Variable Modifications
    state.{classical_energy_name} : ndarray
        Energy of the classical subsystem.
    """
    z_name = kwargs.get("z_name", "z")
    classical_energy_name = kwargs.get("classical_energy_name", "classical_energy")
    z = getattr(state, z_name)
    h_c, _ = sim.model.get("h_c")
    setattr(state, classical_energy_name, np.real(h_c(sim.model, parameters, z=z)))
    return state, parameters


def update_classical_energy_fssh(sim, state, parameters, **kwargs):
    """
    Updates the classical energy for FSSH simulations. If deterministic, the energy in
    each branch is summed together with weights determined by the initial adiabatic
    populations. If not deterministic (and so there is only one branch), the energy is
    computed for the single branch.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    z_name  str, default: "z"
        Name of classical coordinates in state object.
    classical_energy_name : str, default: "classical_energy"
        Name under which to store the classical energy in the state object.
    dm_adb_0_name : str, default: "dm_adb_0"
        Name of the initial adiabatic density matrix in the state object.
    branch_ind_name : str, default: "branch_ind"
        Name of the branch indices in the state object.

    .. rubric:: Variable Modifications
    state.{classical_energy_name} : ndarray
        Energy of the classical subsystem.
    """
    z_name = kwargs.get("z_name", "z")
    classical_energy_name = kwargs.get("classical_energy_name", "classical_energy")
    dm_adb_0_name = kwargs.get("dm_adb_0_name", "dm_adb_0")
    branch_ind_name = kwargs.get("branch_ind_name", "branch_ind")
    z = getattr(state, z_name)
    dm_adb_0 = getattr(state, dm_adb_0_name)
    branch_ind = getattr(state, branch_ind_name)
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_states = sim.model.constants.num_quantum_states
    h_c, _ = sim.model.get("h_c")
    if sim.algorithm.settings.fssh_deterministic:
        classical_energy = 0.0
        branch_weights = num_branches * np.einsum(
            "tbbb->tb",
            dm_adb_0.reshape((batch_size, num_branches, num_states, num_states)),
        )
        for branch_num in range(num_branches):
            z_branch = z[branch_ind == branch_num]
            classical_energy += branch_weights[:, branch_num] * h_c(
                sim.model,
                parameters,
                z=z_branch,
            )
    else:
        classical_energy = 0.0
        z_branch = z[branch_ind == 0]
        classical_energy += h_c(
            sim.model,
            parameters,
            z=z_branch,
        )
    setattr(state, classical_energy_name, classical_energy)
    return state, parameters


def update_quantum_energy(sim, state, parameters, **kwargs):
    """
    Updates the quantum energy w.r.t. the wavefunction specified by ``state.{wf_db_name}`` by taking
    the expectation value of ``state.{h_quantum_name}``.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    wf_db_name : str, default: "wf_db"
        Name of the wavefunction in the state object.
    h_quantum_name : str, default: "h_quantum"
        Name of the total Hamiltonian of the quantum subsystem in the state object.
        (``h_q + h_qc``)
    quantum_energy_name : str, default: "quantum_energy"
        Name under which to store the quantum energy in the state object.

    .. rubric:: Variable Modifications
    state.{quantum_energy_name} : ndarray
        Quantum energy.
    """
    wf_db_name = kwargs.get("wf_db_name", "wf_db")
    h_quantum_name = kwargs.get("h_quantum_name", "h_quantum")
    quantum_energy_name = kwargs.get("quantum_energy_name", "quantum_energy")
    wf_db = getattr(state, wf_db_name)
    h_quantum = getattr(state, h_quantum_name)
    quantum_energy = np.real(
        np.einsum("ti,tij,tj->t", np.conj(wf_db), h_quantum, wf_db, optimize="greedy")
    )
    setattr(
        state,
        quantum_energy_name,
        quantum_energy,
    )
    return state, parameters


def update_quantum_energy_fssh(sim, state, parameters, **kwargs):
    """
    Updates the quantum energy w.r.t. the wavefunction specified by ``state.{wf_db_name}`` by taking
    the expectation value of ``state.{h_quantum_name}``.
    Accounts for both stochastic and deterministic FSSH modes.
    Typically, the wavefunction used is that of the active surface.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    wf_db_name : str, default: act_surf_wf
        Name of the wavefunction in the state object.
    h_quantum_name : str, default: "h_quantum"
        Name of the total Hamiltonian of the quantum subsystem in the state object.
        (``h_q + h_qc``)
    quantum_energy_name : str, default: "quantum_energy"
        Name under which to store the quantum energy in the state object.
    dm_adb_0_name : str, default: "dm_adb_0"
        Name of the initial adiabatic density matrix in the state object.

    .. rubric:: Variable Modifications
    state.{quantum_energy_name} : ndarray
        Quantum energy.
    """
    wf_db_name = kwargs.get("wf_db_name", "act_surf_wf")
    h_quantum_name = kwargs.get("h_quantum_name", "h_quantum")
    quantum_energy_name = kwargs.get("quantum_energy_name", "quantum_energy")
    dm_adb_0_name = kwargs.get("dm_adb_0_name", "dm_adb_0")
    h_quantum = getattr(state, h_quantum_name)
    wf_db = getattr(state, wf_db_name)
    dm_adb_0 = getattr(state, dm_adb_0_name)

    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
        batch_size = sim.settings.batch_size // num_branches
        num_states = sim.model.constants.num_quantum_states
        wf_db = wf_db * np.sqrt(
            num_branches
            * np.einsum(
                "tbbb->tb",
                dm_adb_0.reshape((batch_size, num_branches, num_states, num_states)),
                optimize="greedy",
            ).flatten()[:, np.newaxis]
        )
        quantum_energy = np.real(
            np.einsum(
                "ti,tij,tj->t", np.conj(wf_db), h_quantum, wf_db, optimize="greedy"
            )
        )
    else:
        quantum_energy = np.real(
            np.einsum(
                "ti,tij,tj->t", np.conj(wf_db), h_quantum, wf_db, optimize="greedy"
            )
        )
    setattr(state, quantum_energy_name, quantum_energy)
    return state, parameters


def update_dm_db_fssh(sim, state, parameters, **kwargs):
    """
    Updates the diabatic density matrix for FSSH. Accounts for both stochastic and
    deterministic FSSH modes.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    wf_adb_name : str, default: "wf_adb"
        Name of the adiabatic wavefunction in the state object.
    dm_adb_0_name : str, default: "dm_adb_0"
        Name of the initial adiabatic density matrix in the state object.
    act_surf_name : str, default: "act_surf"
        Name of the active surface wavefunction in the state object.
    dm_db_name : str, default: "dm_db"
        Name of the diabatic density matrix in the state object.
    dm_adb_name : str, default: "dm_adb"
        Name of the adiabatic density matrix in the state object.
    eigvecs_name : str, default: "eigvecs"
        Name of the eigenvectors in the state object.

    .. rubric:: Variable Modifications
    state.{dm_db_name} : ndarray
        Diabatic density matrix.
    state.{dm_adb_name} : ndarray
        Adiabatic density matrix.
    """
    wf_adb_name = kwargs.get("wf_adb_name", "wf_adb")
    dm_adb_0_name = kwargs.get("dm_adb_0_name", "dm_adb_0")
    act_surf_name = kwargs.get("act_surf_name", "act_surf")
    dm_db_name = kwargs.get("dm_db_name", "dm_db")
    dm_adb_name = kwargs.get("dm_adb_name", "dm_adb")
    eigvecs_name = kwargs.get("eigvecs_name", "eigvecs")
    wf_adb = getattr(state, wf_adb_name)
    dm_adb_0 = getattr(state, dm_adb_0_name)
    act_surf = getattr(state, act_surf_name)
    eigvecs = getattr(state, eigvecs_name)
    dm_adb = np.einsum(
        "ti,tj->tij",
        wf_adb,
        np.conj(wf_adb),
        optimize="greedy",
    )
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_quantum_states = sim.model.constants.num_quantum_states
    for nt, _ in enumerate(dm_adb):
        np.einsum("jj->j", dm_adb[nt])[...] = act_surf[nt]
    if sim.algorithm.settings.fssh_deterministic:
        # this reweighting by num_branches simplifies the subsequent averaging.s
        dm_adb = num_branches * (
            np.einsum(
                "tbbb->tb",
                dm_adb_0.reshape(
                    (batch_size, num_branches, num_quantum_states, num_quantum_states)
                ),
            ).flatten()[:, np.newaxis, np.newaxis]
            * dm_adb
        )
    dm_db = functions.transform_mat(dm_adb, eigvecs, adb_to_db=True)
    setattr(state, dm_adb_name, dm_adb)
    setattr(state, dm_db_name, dm_db)
    return state, parameters
