import numpy as np


def _gen_sample_gaussian(constants, z0=None, seed=None, separable=True):
    """
    Generate a sample from a Gaussian distribution.

    Required constants:
        - num_classical_coordinates (int): Number of classical coordinates. Default: None.
        - mcmc_std (float): Standard deviation for sampling. Default: 1.
    """
    if seed is not None:
        np.random.seed(seed)
    num_classical_coordinates = constants.num_classical_coordinates
    if separable:
        rand = np.random.rand(num_classical_coordinates)
    else:
        rand = np.random.rand()
    std_re = constants.get("mcmc_std", 1)
    std_im = constants.get("mcmc_std", 1)
    # Generate random real and imaginary parts of z
    z_re = np.random.normal(loc=0, scale=std_re, size=num_classical_coordinates)
    z_im = np.random.normal(loc=0, scale=std_im, size=num_classical_coordinates)
    z = z_re + 1.0j * z_im
    if z0 is None:
        return (
            np.random.rand(num_classical_coordinates)
            + 1.0j * np.random.rand(num_classical_coordinates),
            rand,
        )
    return z0 + z, rand


def numerical_boltzmann_mcmc_init_classical(model, parameters, **kwargs):
    """
    Initialize classical coordinates according to Boltzmann statistics using Markov-Chain
    Monte Carlo with a Metropolis-Hastings algorithm.

    Required constants:
        - num_classical_coordinates (int): Number of classical coordinates. Default: None.
        - mcmc_burn_in_size (int): Number of burn-in steps. Default: 5000.
        - mcmc_std (float): Standard deviation for sampling. Default: 1.
        - mcmc_h_c_separable (bool): If the classical Hamiltonian is separable. Default: True.
        - mcmc_init_z (np.ndarray): Initial sample. Default: None.
        - kBT (float): Thermal quantum. Default: None.
    """
    seed = kwargs.get("seed", None)
    burn_in_size = model.constants.get("mcmc_burn_in_size", 10000)
    sample_size = model.constants.get("mcmc_sample_size", 100000)
    mcmc_h_c_separable = model.constants.get("mcmc_h_c_separable", True)
    burn_in_seeds = np.arange(burn_in_size)
    sample_seeds = np.arange(sample_size)
    save_inds = np.zeros(len(seed), dtype=int)
    out_tmp = np.zeros(
        (sample_size, model.constants.num_classical_coordinates), dtype=complex
    )
    for s, seed_s in enumerate(seed):
        np.random.seed(seed_s)
        save_inds[s] = np.random.randint(0, sample_size)
    mcmc_init_z, _ = _gen_sample_gaussian(
        model.constants, z0=None, seed=0, separable=False
    )
    sample = model.constants.get("mcmc_init_z", mcmc_init_z)
    h_c, _ = model.get("h_c")
    if mcmc_h_c_separable:
        for s, seed_s in enumerate(burn_in_seeds):
            last_sample = np.copy(sample)
            last_z = np.diag(last_sample)
            last_e = h_c(model, parameters, z=last_z, batch_size=len(last_z))
            proposed_sample, rand = _gen_sample_gaussian(
                model.constants, z0=last_sample, seed=seed_s, separable=True
            )
            new_z = np.diag(proposed_sample)
            new_e = h_c(model, parameters, z=new_z, batch_size=len(new_z))
            thresh = np.minimum(
                np.ones(model.constants.num_classical_coordinates),
                np.exp(-(new_e - last_e) / model.constants.kBT),
            )
            sample[rand < thresh] = proposed_sample[rand < thresh]
        for s, seed_s in enumerate(sample_seeds):
            last_sample = np.copy(sample)
            last_z = np.diag(last_sample)
            last_e = h_c(model, parameters, z=last_z, batch_size=len(last_z))
            proposed_sample, rand = _gen_sample_gaussian(
                model.constants, z0=last_sample, seed=seed_s, separable=True
            )
            new_z = np.diag(proposed_sample)
            new_e = h_c(model, parameters, z=new_z, batch_size=len(new_z))
            thresh = np.minimum(
                np.ones(model.constants.num_classical_coordinates),
                np.exp(-(new_e - last_e) / model.constants.kBT),
            )
            sample[rand < thresh] = proposed_sample[rand < thresh]
            out_tmp[s] = sample
        return out_tmp[save_inds]

    for s, seed_s in enumerate(burn_in_seeds):
        last_sample = np.copy(sample)
        last_e = h_c(model, parameters, z=last_sample, batch_size=len(last_sample))
        proposed_sample, rand = _gen_sample_gaussian(
            model.constants, z0=last_sample, seed=seed_s, separable=False
        )
        new_e = h_c(
            model,
            parameters,
            z=proposed_sample,
            batch_size=len(proposed_sample),
        )
        thresh = min(1, np.exp(-(new_e - last_e) / model.constants.kBT))
        if rand < thresh:
            sample = proposed_sample
    for s, seed_s in enumerate(sample_seeds):
        last_sample = np.copy(sample)
        last_e = h_c(model, parameters, z=last_sample, batch_size=len(last_sample))
        proposed_sample, rand = _gen_sample_gaussian(
            model.constants, z0=last_sample, seed=seed_s, separable=False
        )
        new_e = h_c(
            model,
            parameters,
            z=proposed_sample,
            batch_size=len(proposed_sample),
        )
        thresh = min(1, np.exp(-(new_e - last_e) / model.constants.kBT))
        if rand < thresh:
            sample = proposed_sample
        out_tmp[s] = sample
    return out_tmp[save_inds]



def dh_c_dzc_finite_differences(model, parameters, **kwargs):
    """
    Calculate the gradient of the classical Hamiltonian using finite differences.

    Required constants:
        - num_classical_coordinates (int): Number of classical coordinates. Default: None.
    """
    z = kwargs["z"]
    delta_z = model.constants.get("dh_c_dzc_finite_difference_delta", 1e-6)
    batch_size = len(parameters.seed)
    num_classical_coordinates = model.constants.num_classical_coordinates
    offset_z_re = (
        z[:, np.newaxis, :]
        + np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    offset_z_im = (
        z[:, np.newaxis, :]
        + 1.0j * np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    h_c, has_h_c = model.get("h_c")
    h_c_0 = h_c(model, parameters, z=z, batch_size=len(z))
    h_c_offset_re = h_c(
        model,
        parameters,
        z=offset_z_re,
        batch_size=batch_size * num_classical_coordinates,
    ).reshape(batch_size, num_classical_coordinates)
    h_c_offset_im = h_c(
        model,
        parameters,
        z=offset_z_im,
        batch_size=batch_size * num_classical_coordinates,
    ).reshape(batch_size, num_classical_coordinates)
    diff_re = (h_c_offset_re - h_c_0[:, np.newaxis]) / delta_z
    diff_im = (h_c_offset_im - h_c_0[:, np.newaxis]) / delta_z
    dh_c_dzc = 0.5 * (diff_re + 1.0j * diff_im)
    return dh_c_dzc


def dh_qc_dzc_finite_differences(model, parameters, **kwargs):
    """
    Calculate the gradient of the quantum-classical Hamiltonian using finite differences.

    Required constants:
        - num_classical_coordinates (int): Number of classical coordinates. Default: None.
        - num_quantum_states (int): Number of quantum states. Default: None.
        - finite_difference_dz (float): Step size for finite differences. Default: 1e-6.
    """
    z = kwargs["z"]
    delta_z = model.constants.get("dh_qc_dzc_finite_difference_delta", 1e-6)
    batch_size = len(parameters.seed)
    num_classical_coordinates = model.constants.num_classical_coordinates
    num_quantum_states = model.constants.num_quantum_states
    offset_z_re = (
        z[:, np.newaxis, :]
        + np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    offset_z_im = (
        z[:, np.newaxis, :]
        + 1.0j * np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    h_qc, has_h_qc = model.get("h_qc")
    h_qc_0 = h_qc(model, parameters, z=z)
    h_qc_offset_re = h_qc(
        model,
        parameters,
        z=offset_z_re,
        batch_size=batch_size * num_classical_coordinates,
    ).reshape(
        batch_size,
        num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    h_qc_offset_im = h_qc(
        model,
        parameters,
        z=offset_z_im,
        batch_size=batch_size * num_classical_coordinates,
    ).reshape(
        batch_size,
        num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    diff_re = (h_qc_offset_re - h_qc_0[:, np.newaxis, :, :]) / delta_z
    diff_im = (h_qc_offset_im - h_qc_0[:, np.newaxis, :, :]) / delta_z
    dh_qc_dzc = 0.5 * (diff_re + 1.0j * diff_im)
    inds = np.where(dh_qc_dzc != 0)
    mels = dh_qc_dzc[inds]
    shape = np.shape(dh_qc_dzc)
    return inds, mels, shape


def numerical_fssh_hop(model, parameters, **kwargs):
    """
    Determines the coordinate rescaling in FSSH numerically.

    Required constants:
        - numerical_fssh_hop_gamma_range (float): Range for gamma. Default: 5.
        - numerical_fssh_hop_num_iter (int): Number of iterations. Default: 10.
        - numerical_fssh_hop_num_points (int): Number of points. Default: 10.
    """
    z = kwargs["z"]
    delta_z = kwargs["delta_z"]
    ev_diff = kwargs["ev_diff"]
    gamma_range = model.constants.get("numerical_fssh_hop_gamma_range", 5)
    max_iter = model.constants.get("numerical_fssh_hop_max_iter", 20)
    num_points = model.constants.get("numerical_fssh_hop_num_points", 10)
    thresh = model.constants.get("numerical_fssh_hop_threshold", 1e-6)
    h_c, _ = model.get("h_c")
    init_energy = h_c(
        model, model.constants, parameters, z=np.array([z]), batch_size=1
    )[0]
    min_gamma = 0
    num_iter = 0
    min_energy = 1
    while min_energy > thresh and num_iter < max_iter:
        gamma_list = np.linspace(
            min_gamma - gamma_range, min_gamma + gamma_range, num_points
        )
        new_energies = np.abs(
            ev_diff
            - np.array(
                [
                    init_energy
                    - h_c(
                        model,
                        parameters,
                        z=np.array([z - 1.0j * gamma * delta_z]),
                        batch_size=1,
                    )[0]
                    for gamma in gamma_list
                ]
            )
        )
        min_gamma = gamma_list[np.argmin(new_energies)]
        min_energy = np.min(new_energies)
        gamma_range = gamma_range / 2
        num_iter += 1
    if min_energy > thresh:
        return z, False
    return z - 1.0j * min_gamma * delta_z, True