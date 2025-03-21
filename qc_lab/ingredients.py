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


def harmonic_oscillator_dh_c_dzc(model, constants, parameters, **kwargs):
    """
    Calculate the derivative of the classical harmonic oscillator Hamiltonian
    with respect to the z coordinate.
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
    a = (1 / 4) * (((w**2) / h) - h)
    b = (1 / 4) * (((w**2) / h) + h)
    dh_c_dzc = 2 * b[..., :] * z + 2 * a[..., :] * np.conj(z)
    return dh_c_dzc


def two_level_system_h_q(model, constants, parameters, **kwargs):
    """
    Calculate the quantum Hamiltonian for a two-level system.
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
    Calculate the quantum Hamiltonian for a nearest-neighbor lattice.
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
def holstein_coupling_h_qc_jit(batch_size, num_sites, z, g, w, h):
    """
    Low level function to generate the Holstein coupling Hamiltonian.
    """
    h_qc = np.zeros((batch_size, num_sites, num_sites)) + 0.0j
    for b in range(batch_size):
        for i in range(num_sites):
            h_qc[b, i, i] = (g[i] * w[i] * np.sqrt(w[i] / h[i])) * (
                z[b, i] + np.conj(z[b, i])
            )
    return h_qc


def holstein_coupling_h_qc(model, constants, parameters, **kwargs):
    """
    Calculate the Holstein coupling Hamiltonian.
    """
    del model, parameters
    z = kwargs["z"]
    if kwargs.get("batch_size") is not None:
        batch_size = kwargs.get("batch_size")
        assert len(z) == batch_size
    else:
        batch_size = len(z)
    num_sites = constants.num_quantum_states
    w = constants.holstein_coupling_oscillator_frequency
    g = constants.holstein_coupling_dimensionless_coupling
    h = constants.classical_coordinate_weight
    return holstein_coupling_h_qc_jit(batch_size, num_sites, z, g, w, h)


def holstein_coupling_dh_qc_dzc(model, constants, parameters, **kwargs):
    """
    Calculate the derivative of the Holstein coupling Hamiltonian with
    respect to the z-coordinates.
    """
    # if there is not an explicitly specified batch_size,
    # use the length of the seed.
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
        num_sites = constants.num_quantum_states
        w = constants.holstein_coupling_oscillator_frequency
        g = constants.holstein_coupling_dimensionless_coupling
        h = constants.classical_coordinate_weight
        dh_qc_dzc = np.zeros(
            (batch_size, num_sites, num_sites, num_sites), dtype=complex
        )
        np.einsum("tiii->ti", dh_qc_dzc, optimize="greedy")[...] = (
            g * w * np.sqrt(w / h)
        )[..., :] * (np.ones_like(z, dtype=complex))
        inds = np.where(dh_qc_dzc != 0)
        mels = dh_qc_dzc[inds]
        shape = np.shape(dh_qc_dzc)
        model.dh_qc_dzc_inds = inds
        model.dh_qc_dzc_mels = dh_qc_dzc[inds]
        model.dh_qc_dzc_shape = shape
        return inds, mels, shape
    return model.dh_qc_dzc_inds, model.dh_qc_dzc_mels, model.dh_qc_dzc_shape

def spin_boson_h_qc(model, constants, parameters, **kwargs):
    z = kwargs.get("z")
    if kwargs.get("batch_size") is not None:
        batch_size = kwargs.get("batch_size")
        assert len(z) == batch_size
    else:
        batch_size = len(z)
    g = constants.spin_boson_coupling
    m = constants.classical_coordinate_mass
    h = constants.classical_coordinate_weight
    h_qc = np.zeros((batch_size, 2, 2), dtype=complex)
    h_qc[:, 0, 0] = np.sum(
        g * np.sqrt(1 / (2 * m * h))[np.newaxis, :] * (z + np.conj(z)), axis=-1
    )
    h_qc[:, 1, 1] = -h_qc[:, 0, 0]
    return h_qc


def spin_boson_dh_qc_dzc(model, constants, parameters, **kwargs):
    z = kwargs.get("z")
    if kwargs.get("batch_size") is not None:
        batch_size = kwargs.get("batch_size")
        assert len(z) == batch_size
    else:
        batch_size = len(z)

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

        m = constants.classical_coordinate_mass
        g = constants.spin_boson_coupling
        h = constants.classical_coordinate_weight
        assert constants.num_quantum_states == 2
        dh_qc_dzc = np.zeros((batch_size, constants.num_classical_coordinates, 2, 2), dtype=complex)
        dh_qc_dzc[:, :, 0, 0] = (g * np.sqrt(1 / (2 * m * h)))[..., :]
        dh_qc_dzc[:, :, 1, 1] = -dh_qc_dzc[..., :, 0, 0]
        inds = np.where(dh_qc_dzc != 0)
        mels = dh_qc_dzc[inds]
        shape = np.shape(dh_qc_dzc)
        model.dh_qc_dzc_inds = inds
        model.dh_qc_dzc_mels = dh_qc_dzc[inds]
        model.dh_qc_dzc_shape = shape
    else:
        inds = model.dh_qc_dzc_inds
        mels = model.dh_qc_dzc_mels
        shape = model.dh_qc_dzc_shape
    return inds, mels, shape


def harmonic_oscillator_hop_function(model, constants, parameters, **kwargs):
    """
    Perform a hopping operation for the harmonic oscillator.
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
    Initialize classical coordinates according to Boltzmann statistics for the Harmonic oscillator.
    """
    del model, parameters
    seed = kwargs.get("seed", None)
    kbt = constants.temp
    h = constants.classical_coordinate_weight
    w = constants.harmonic_oscillator_frequency
    m = constants.classical_coordinate_mass
    out = np.zeros((len(seed), constants.num_classical_coordinates), dtype=complex)
    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        # Calculate the standard deviations for q and p.
        std_q = np.sqrt(kbt / (m * (w**2)))
        std_p = np.sqrt(m * kbt)
        # Generate random q and p values.
        q = np.random.normal(
            loc=0, scale=std_q, size=constants.num_classical_coordinates
        )
        p = np.random.normal(
            loc=0, scale=std_p, size=constants.num_classical_coordinates
        )
        # Calculate the complex classical coordinate.
        z = np.sqrt(h * m / 2) * (q + 1.0j * (p / (h * m)))
        out[s] = z
    return out


def harmonic_oscillator_wigner_init_classical(model, constants, parameters, **kwargs):
    """
    Initialize classical coordinates according to the Wigner distribution
    of the ground state of a harmonic oscillator.
    """
    del model, parameters
    seed = kwargs.get("seed", None)
    m = constants.classical_coordinate_mass
    h = constants.classical_coordinate_weight
    w = constants.harmonic_oscillator_frequency
    kbt = constants.temp
    out = np.zeros((len(seed), constants.num_classical_coordinates), dtype=complex)
    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        # Calculate the standard deviations for q and p.
        std_q = np.sqrt(1 / (2 * w * m * np.tanh(w / (2 * kbt))))
        std_p = np.sqrt((m * w) / (2 * np.tanh(w / (2 * kbt))))
        # Generate random q and p values.
        q = np.random.normal(
            loc=0, scale=std_q, size=constants.num_classical_coordinates
        )
        p = np.random.normal(
            loc=0, scale=std_p, size=constants.num_classical_coordinates
        )
        # Calculate the complex classical coordinate.
        z = np.sqrt(h * m / 2) * (q + 1.0j * (p / (h * m)))
        out[s] = z
    return out
