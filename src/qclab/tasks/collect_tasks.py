"""
This module contains tasks that are used to collect data from the state or
parameters objects into the output dictionary of the state object.
"""

from qclab.simulation import Simulation


def collect_t(
    sim: Simulation,
    state: dict,
    parameters: dict,
    t_var_name: str = "t",
    t_output_name: str = "t",
):
    """
    Collects the time into the output dictionary of the state object.

    Parameters
    ----------
    sim : Simulation
        The simulation object.
    state : dict
        The state object.
    parameters : dict
        The parameters object.

    Other Parameters
    ----------------
    t_var_name
        Name of the time variable in the state object.
    t_output_name
        Name of the output variable for the time.

    Reads
    -----
    state[t_var_name] : ndarray, (B), float64
        The time in each trajectory.

    Writes
    ------
    state["output_dict"][t_output_name] : ndarray, (B), float64
        stores the current time in each trajectory.

    Shapes and dtypes
    -------------------
    B = sim.settings.batch_size

    Requires
    --------
    None

    Returns
    -------
    (state, parameters) : tuple(dict, dict)
        The updated state and parameters objects.

    See Also
    --------
    qclab.tasks.update_tasks.update_t
    """
    state["output_dict"][t_output_name] = state[t_var_name]
    return state, parameters


def collect_dm_db(
    sim: Simulation,
    state: dict,
    parameters: dict,
    dm_db_var_name: str = "dm_db",
    dm_db_output_name: str = "dm_db",
):
    """
    Collects the diabatic density matrix into the output dictionary of the state object.

    Parameters
    ----------
    sim : Simulation
        The simulation object.
    state : dict
        The state object.
    parameters : dict
        The parameters object.

    Other Parameters
    ----------------
    dm_db_var_name
        Name of the diabatic density matrix in the state object.
    dm_db_output_name
        Name of the diabatic density matrix in the output dictionary.

    Reads
    -----
    state[dm_db_var_name] : ndarray, (B, N, N), complex128
        The diabatic density matrix in each trajectory.

    Writes
    ------
    state["output_dict"][dm_db_output_name] : ndarray, (B, N, N), complex128
        The diabatic density matrix in each trajectory.

    Shapes and dtypes
    -------------------
    B = sim.settings.batch_size
    N = sim.model.constants.num_quantum_states

    Requires
    --------
    sim.model.constants.num_quantum_states : int
        Number of quantum states.
    sim.settings.batch_size : int
        Number of trajectories in the batch.

    Returns
    -------
    (state, parameters) : tuple(dict, dict)
        The updated state and parameters objects.

    See Also
    --------
    qclab.tasks.update_tasks.update_dm_db
    qclab.tasks.update_tasks.update_dm_db_fssh
    """
    state["output_dict"][dm_db_output_name] = state[dm_db_var_name]
    return state, parameters


def collect_classical_energy(
    sim: Simulation,
    state: dict,
    parameters: dict,
    classical_energy_var_name: str = "classical_energy",
    classical_energy_output_name: str = "classical_energy",
):
    """
    Collects the classical energy into the output dictionary of the state object.

    Parameters
    ----------
    sim : Simulation
        The simulation object.
    state : dict
        The state object.
    parameters : dict
        The parameters object.

    Other Parameters
    ----------------
    classical_energy_var_name
        Name of the classical energy variable in the state object.
    classical_energy_output_name
        Name of the output variable for the classical energy.

    Reads
    -----
    state[classical_energy_var_name] : ndarray, (B), float64
        The classical energy in each trajectory.

    Writes
    ------
    state["output_dict"][classical_energy_output_name] : ndarray, (B), float64
        stores the classical energy.

    Shapes and dtypes
    -------------------
    B = sim.settings.batch_size

    Requires
    --------
    None

    Returns
    -------
    (state, parameters) : tuple(dict, dict)
        The updated state and parameters objects.

    See Also
    --------
    qclab.tasks.update_tasks.update_classical_energy
    qclab.tasks.update_tasks.update_classical_energy_fssh
    """
    state["output_dict"][classical_energy_output_name] = state[
        classical_energy_var_name
    ]
    return state, parameters


def collect_quantum_energy(sim: Simulation, state: dict, parameters: dict, quantum_energy_var_name: str = "quantum_energy", quantum_energy_output_name: str = "quantum_energy"):
    """
    Collects the quantum energy into the output dictionary of the state object.

    Parameters
    ----------
    sim : Simulation
        The simulation object.
    state : dict
        The state object.
    parameters : dict
        The parameters object.

    Other Parameters
    ----------------
    quantum_energy_var_name
        Name of the quantum energy variable in the state object.
    quantum_energy_output_name
        Name of the output variable for the quantum energy.

    Reads
    -----
    state[quantum_energy_var_name] : ndarray, (B), float64
        The quantum energy in each trajectory.
    
    Writes
    ------
    state["output_dict"][quantum_energy_output_name] : ndarray, (B), float64
        The quantum energy in each trajectory.

    Shapes and dtypes
    -------------------
    B = sim.settings.batch_size

    Requires
    --------
    None

    Returns
    -------
    (state, parameters) : tuple(dict, dict)
        The updated state and parameters objects.

    See Also
    --------
    qclab.tasks.update_tasks.update_quantum_energy
    qclab.tasks.update_tasks.update_quantum_energy_fssh
    """
    state["output_dict"][quantum_energy_output_name] = state[quantum_energy_var_name]
    return state, parameters
