"""
This file contains ingredient functions for use in Model classes.
"""

import functools
import numpy as np
from numba import njit


def make_ingredient_sparse(ingredient):
    """
    Wrapper that converts a vectorized ingredient output to a sparse format
    consisting of the indices (inds), nonzero elements (mels), and shape.
    """

    @functools.wraps(ingredient)
    def sparse_ingredient(*args, **kwargs):
        (model, constants, parameters) = args
        out = ingredient(model, constants, parameters, **kwargs)
        inds = np.where(out != 0)
        mels = out[inds]
        shape = np.shape(out)
        return inds, mels, shape

    return sparse_ingredient


def vectorize_ingredient(ingredient):
    """
    Wrapper that vectorize an ingredient function.
    It assumes that any kwarg is an numpy.ndarray is vectorized over its first index.
    Other kwargs are assumed to not be vectorized.
    """

    @functools.wraps(ingredient)
    def vectorized_ingredient(*args, **kwargs):
        (model, constants, parameters) = args
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = len(parameters.seed)
        keys = kwargs.keys()
        kwargs_list = []
        for n in range(batch_size):
            kwargs_n = {}
            for key in keys:
                if isinstance(kwargs[key], np.ndarray):
                    kwargs_n[key] = kwargs[key][n]
                else:
                    kwargs_n[key] = kwargs[key]
            kwargs_list.append(kwargs_n)
        out = np.array(
            [
                ingredient(model, constants, parameters, **kwargs_list[n])
                for n in range(batch_size)
            ]
        )
        return out

    return vectorized_ingredient


def harmonic_oscillator_h_c(model, constants, parameters, **kwargs):
    """
    Harmonic oscillator classical Hamiltonian function.

    Required Constants:
        - `classical_coordinate_weight`: Array of weights for classical coordinates.
        - `harmonic_oscillator_frequency`: Array of harmonic oscillator frequencies.
        - `classical_coordinate_mass`: Array of masses for classical coordinates.

    Keyword Arguments:
        - `z`: complex-valued classical coordinates.
        - `batch_size`: (Optional) Number of batches for vectorized computation.
    """
    del model, parameters
    z = kwargs.get("z")
    if kwargs.get("batch_size") is not None:
        batch_size = kwargs.get("batch_size")
        assert len(z) == batch_size
    else:
        batch_size = len(z)
    h = constants.classical_coordinate_weight[np.newaxis, :]
    w = constants.harmonic_oscillator_frequency[np.newaxis, :]
    m = constants.classical_coordinate_mass[np.newaxis, :]
    q = np.sqrt(2 / (m * h)) * np.real(z)
    p = np.sqrt(2 * m * h) * np.imag(z)
    h_c = np.sum((1 / 2) * (((p**2) / m) + m * (w**2) * (q**2)), axis=-1)
    return h_c


@njit()
def harmonic_oscillator_dh_c_dzc_jit(z, h, w):
    """
    Numba accelerated calculation of the gradient of the Harmonic oscillator Hamiltonian.
    """
    a = (1 / 2) * (((w**2) / h) - h)
    b = (1 / 2) * (((w**2) / h) + h)
    out = b[..., :] * z + a[..., :] * np.conj(z)
    return out


def harmonic_oscillator_dh_c_dzc(model, constants, parameters, **kwargs):
    """
    Derivative of the classical harmonic oscillator Hamiltonian with respect to the `z` coordinate.

    Required Constants:
        - `classical_coordinate_weight`: Array of weights for classical coordinates.
        - `harmonic_oscillator_frequency`: Array of harmonic oscillator frequencies.

    Keyword Arguments:
        - `z`: complex-valued classical coordinates.
        - `batch_size`: (Optional) Number of batches for vectorized computation.
    """
    del model, parameters
    z = kwargs.get("z")
    if kwargs.get("batch_size") is not None:
        batch_size = kwargs.get("batch_size")
        assert len(z) == batch_size
    else:
        batch_size = len(z)
    h = constants.classical_coordinate_weight
    w = constants.harmonic_oscillator_frequency
    return harmonic_oscillator_dh_c_dzc_jit(z, h, w)


def two_level_system_h_q(model, constants, parameters, **kwargs):
    """
    Quantum Hamiltonian for a two-level system.

    Required Constants:
        - `two_level_system_a`: Energy of the first level.
        - `two_level_system_b`: Energy of the second level.
        - `two_level_system_c`: Real part of the coupling between levels.
        - `two_level_system_d`: Imaginary part of the coupling between levels.

    Keyword Arguments:
        - `batch_size`: (Optional) Number of batches for vectorized computation.
    """
    del model
    if kwargs.get("batch_size") is not None:
        batch_size = kwargs.get("batch_size")
    else:
        batch_size = len(parameters.seed)
    h_q = np.zeros((batch_size, 2, 2), dtype=complex)
    h_q[:, 0, 0] = constants.two_level_system_a
    h_q[:, 1, 1] = constants.two_level_system_b
    h_q[:, 0, 1] = constants.two_level_system_c + 1j * constants.two_level_system_d
    h_q[:, 1, 0] = constants.two_level_system_c - 1j * constants.two_level_system_d
    return h_q


@njit
def nearest_neighbor_lattice_h_q_jit(
    batch_size, num_sites, hopping_energy, periodic_boundary
):
    """
    Low level function to generate the nearest-neighbor lattice quantum Hamiltonian.
    """
    out = np.zeros((batch_size, num_sites, num_sites)) + 0.0j
    for b in range(batch_size):
        for n in range(num_sites - 1):
            out[b, n, n + 1] = -hopping_energy + 0.0j
            out[b, n + 1, n] = -np.conj(hopping_energy) + 0.0j
        if periodic_boundary:
            out[b, 0, num_sites - 1] = -hopping_energy + 0.0j
            out[b, num_sites - 1, 0] = -np.conj(hopping_energy) + 0.0j
    return out


def nearest_neighbor_lattice_h_q(model, constants, parameters, **kwargs):
    """
    Quantum Hamiltonian for a nearest-neighbor lattice.

    Required Constants:
        - `num_quantum_states`: Number of quantum states (sites).
        - `nearest_neighbor_lattice_hopping_energy`: Hopping energy between sites.
        - `nearest_neighbor_lattice_periodic_boundary`: Boolean indicating periodic boundary conditions.

    Keyword Arguments:
        - `batch_size`: (Optional) Number of batches for vectorized computation.
    """
    if kwargs.get("batch_size") is not None:
        batch_size = kwargs.get("batch_size")
    else:
        batch_size = len(parameters.seed)
    num_sites = constants.num_quantum_states
    hopping_energy = constants.nearest_neighbor_lattice_hopping_energy
    periodic_boundary = constants.nearest_neighbor_lattice_periodic_boundary
    if hasattr(model, "h_q_mat"):
        if model.h_q_mat is not None:
            if len(model.h_q_mat) == batch_size:
                return model.h_q_mat
    h_q = np.zeros((num_sites, num_sites), dtype=complex)
    # Fill the Hamiltonian matrix with hopping energies.
    for n in range(num_sites - 1):
        h_q[n, n + 1] += -hopping_energy
        h_q[n + 1, n] += np.conj(h_q[n, n + 1])
    # Apply periodic boundary conditions if specified.
    if periodic_boundary:
        h_q[0, num_sites - 1] += -hopping_energy
        h_q[num_sites - 1, 0] += np.conj(h_q[0, num_sites - 1])
    model.h_q_mat = h_q[np.newaxis, :, :] + np.zeros(
        (batch_size, num_sites, num_sites), dtype=complex
    )
    return model.h_q_mat


@njit()
def diagonal_linear_h_qc_jit(
    batch_size, num_sites, num_classical_coordinates, z, gamma
):
    """
    Low level function to generate the diagonal linear quantum-classical coupling Hamiltonian.
    """
    h_qc = np.zeros((batch_size, num_sites, num_sites)) + 0.0j
    for b in range(batch_size):
        for i in range(num_sites):
            for j in range(num_classical_coordinates):
                h_qc[b, i, i] = h_qc[b, i, i] + gamma[i, j] * (
                    z[b, j] + np.conj(z[b, j])
                )
    return h_qc


def diagonal_linear_h_qc(model, constants, parameters, **kwargs):
    """
    Diagonal linear quantum-classical coupling Hamiltonian.

    Diagonal elements are given by

    :math:`H_{ii} = \sum_{j} \gamma_{ij} (z_{j} + z_{j}^*)`

    Required Constants:
        - `num_quantum_states`: Number of quantum states (sites).
        - `num_classical_coordinates`: Number of classical coordinates.
        - `diagonal_linear_coupling`: Array of coupling constants (num_sites, num_classical_coordinates).

    Keyword Arguments:
        - `z`: complex-valued classical coordinates.
        - `batch_size`: (Optional) Number of batches for vectorized computation.
    """
    del model, parameters
    z = kwargs["z"]
    if kwargs.get("batch_size") is not None:
        batch_size = kwargs.get("batch_size")
        assert len(z) == batch_size
    else:
        batch_size = len(z)
    num_sites = constants.num_quantum_states
    num_classical_coordinates = constants.num_classical_coordinates
    gamma = constants.diagonal_linear_coupling
    return diagonal_linear_h_qc_jit(
        batch_size, num_sites, num_classical_coordinates, z, gamma
    )


def diagonal_linear_dh_qc_dzc(model, constants, parameters, **kwargs):
    """
    Gradient of the diagonal linear quantum-classical coupling Hamiltonian.

    Required Constants:
        - `num_quantum_states`: Number of quantum states (sites).
        - `num_classical_coordinates`: Number of classical coordinates
        - `diagonal_linear_coupling`: Array of coupling constants (num_sites, num_classical_coordinates).

    Keyword Arguments:
        - `z`: complex-valued classical coordinates.
        - `batch_size`: (Optional) Number of batches for vectorized computation.
    """
    z = kwargs["z"]
    if kwargs.get("batch_size") is not None:
        batch_size = kwargs.get("batch_size")
    else:
        batch_size = len(parameters.seed)
    recalculate = False
    if model.dh_qc_dzc_shape is not None:
        if model.dh_qc_dzc_shape[0] != batch_size:
            recalculate = True
    if (
        model.dh_qc_dzc_inds is None
        or model.dh_qc_dzc_mels is None
        or model.dh_qc_dzc_shape is None
        or recalculate
    ):

        num_states = constants.num_quantum_states
        num_classical_coordinates = constants.num_classical_coordinates
        gamma = constants.diagonal_linear_coupling
        dh_qc_dzc = np.zeros(
            (num_classical_coordinates, num_states, num_states), dtype=complex
        )
        for i in range(num_states):
            for j in range(num_classical_coordinates):
                dh_qc_dzc[j, i, i] = gamma[i, j]
        dh_qc_dzc = dh_qc_dzc[np.newaxis, :, :, :] + np.zeros(
            (batch_size, num_classical_coordinates, num_states, num_states),
            dtype=complex,
        )
        inds = np.where(dh_qc_dzc != 0)
        mels = dh_qc_dzc[inds]
        shape = np.shape(dh_qc_dzc)
        model.dh_qc_dzc_inds = inds
        model.dh_qc_dzc_mels = dh_qc_dzc[inds]
        model.dh_qc_dzc_shape = shape
        return inds, mels, shape
    return model.dh_qc_dzc_inds, model.dh_qc_dzc_mels, model.dh_qc_dzc_shape


def harmonic_oscillator_hop_function(model, constants, parameters, **kwargs):
    """
    Perform a hopping operation for the harmonic oscillator.

    Required Constants:
        - `harmonic_oscillator_frequency`: Array of harmonic oscillator frequencies.
        - `classical_coordinate_weight`: Array of weights for classical coordinates.

    Keyword Arguments:
        - `z`: complex-valued classical coordinates.
        - `delta_z`: Change in classical coordinates.
        - `ev_diff`: Energy difference for the hopping operation.
    """
    del model, parameters
    z = kwargs["z"]
    delta_z = kwargs["delta_z"]
    ev_diff = kwargs["ev_diff"]
    hopped = False
    delta_zc = np.conj(delta_z)
    zc = np.conj(z)
    a_const = (1 / 4) * (
        (
            (constants.harmonic_oscillator_frequency**2)
            / constants.classical_coordinate_weight
        )
        - constants.classical_coordinate_weight
    )
    b_const = (1 / 4) * (
        (
            (constants.harmonic_oscillator_frequency**2)
            / constants.classical_coordinate_weight
        )
        + constants.classical_coordinate_weight
    )
    akj_z = np.sum(
        2 * delta_zc * delta_z * b_const - a_const * (delta_z**2 + delta_zc**2)
    )
    bkj_z = 2j * np.sum(
        (z * delta_z - delta_zc * zc) * a_const
        + (delta_z * zc - delta_zc * z) * b_const
    )
    ckj_z = ev_diff
    disc = bkj_z**2 - 4 * akj_z * ckj_z
    if disc >= 0:
        if bkj_z < 0:
            gamma = bkj_z + np.sqrt(disc)
        else:
            gamma = bkj_z - np.sqrt(disc)
        if akj_z == 0:
            gamma = 0
        else:
            gamma = gamma / (2 * akj_z)
        # adjust classical coordinate
        z = z - 1.0j * np.real(gamma) * delta_z
        hopped = True
    return z, hopped


def harmonic_oscillator_boltzmann_init_classical(
    model, constants, parameters, **kwargs
):
    """
    Initialize classical coordinates according to Boltzmann statistics for the harmonic oscillator.

    Required Constants:
        - `kBT`: Thermal quantum.
        - `classical_coordinate_weight`: Array of weights for classical coordinates.
        - `harmonic_oscillator_frequency`: Array of harmonic oscillator frequencies.
        - `classical_coordinate_mass`: Array of masses for classical coordinates.

    Keyword Arguments:
        - `seed`: Array of random seeds for initialization.
    """
    del model, parameters
    seed = kwargs.get("seed", None)
    kBT = constants.kBT
    h = constants.classical_coordinate_weight
    w = constants.harmonic_oscillator_frequency
    m = constants.classical_coordinate_mass
    out = np.zeros((len(seed), constants.num_classical_coordinates), dtype=complex)
    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        # Calculate the standard deviations for q and p.
        std_q = np.sqrt(kBT / (m * (w**2)))
        std_p = np.sqrt(m * kBT)
        # Generate random q and p values.
        q = np.random.normal(
            loc=0, scale=std_q, size=constants.num_classical_coordinates
        )
        p = np.random.normal(
            loc=0, scale=std_p, size=constants.num_classical_coordinates
        )
        # Calculate the complex-valued classical coordinate.
        z = np.sqrt(h * m / 2) * (q + 1.0j * (p / (h * m)))
        out[s] = z
    return out


def harmonic_oscillator_wigner_init_classical(model, constants, parameters, **kwargs):
    """
    Initialize classical coordinates according to the Wigner distribution of the ground state of a harmonic oscillator.

    Required Constants:
        - `kBT`: Thermal quantum.
        - `classical_coordinate_weight`: Array of weights for classical coordinates.
        - `harmonic_oscillator_frequency`: Array of harmonic oscillator frequencies.
        - `classical_coordinate_mass`: Array of masses for classical coordinates.

    Keyword Arguments:
        - `seed`: Array of random seeds for initialization.
    """
    del model, parameters
    seed = kwargs.get("seed", None)
    m = constants.classical_coordinate_mass
    h = constants.classical_coordinate_weight
    w = constants.harmonic_oscillator_frequency
    kBT = constants.kBT
    out = np.zeros((len(seed), constants.num_classical_coordinates), dtype=complex)
    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        # Calculate the standard deviations for q and p.
        std_q = np.sqrt(1 / (2 * w * m * np.tanh(w / (2 * kBT))))
        std_p = np.sqrt((m * w) / (2 * np.tanh(w / (2 * kBT))))
        # Generate random q and p values.
        q = np.random.normal(
            loc=0, scale=std_q, size=constants.num_classical_coordinates
        )
        p = np.random.normal(
            loc=0, scale=std_p, size=constants.num_classical_coordinates
        )
        # Calculate the complex-valued classical coordinate.
        z = np.sqrt(h * m / 2) * (q + 1.0j * (p / (h * m)))
        out[s] = z
    return out
