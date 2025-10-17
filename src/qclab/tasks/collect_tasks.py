"""
This module contains tasks that are used to collect data from the state or
parameters objects into the output dictionary of the state object.
"""


def collect_t(sim, state, parameters, **kwargs):
    """
    Collects the time from the state object and stores it in the output
    dictionary.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    t_name : str, default : "t"
        Name of the time in the state object.
    t_output_name : str, default : "t"
        Name of the time in the output dictionary.

    .. rubric:: Modifications
    state["output_dict"][t_output_name] : ndarray
        stores the current time in each trajectory.
    """
    t_name = kwargs.get("t_name", "t")
    t_output_name = kwargs.get("t_output_name", "t")
    state["output_dict"][t_output_name] = state[t_name]
    return state, parameters


def collect_dm_db(sim, state, parameters, **kwargs):
    """
    Collects the diabatic density matrix from the state object and stores it
    in the output dictionary.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    dm_db_name : str, default : "dm_db"
        Name of the diabatic density matrix in the state object.
    dm_db_output_name : str, default : "dm_db"
        Name of the diabatic density matrix in the output dictionary.

    .. rubric:: Modifications
    state["output_dict"][dm_db_output_name] : ndarray
        Stores the diabatic density matrix.
    """
    dm_db_name = kwargs.get("dm_db_name", "dm_db")
    dm_db_output_name = kwargs.get("dm_db_output_name", "dm_db")
    state["output_dict"][dm_db_output_name] = state[dm_db_name]
    return state, parameters


def collect_classical_energy(sim, state, parameters, **kwargs):
    """
    Collects the classical energy from the state object and stores it in the
    output dictionary.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    classical_energy_name : str, default : "classical_energy"
        Name of the classical energy in the state object.
    classical_energy_output_name : str, default : "classical_energy"
        Name of the classical energy in the output dictionary.

    .. rubric:: Modifications
    state["output_dict"][classical_energy_output_name] : ndarray
        stores the classical energy.
    """
    classical_energy_name = kwargs.get("classical_energy_name", "classical_energy")
    classical_energy_output_name = kwargs.get(
        "classical_energy_output_name", "classical_energy"
    )
    state["output_dict"][classical_energy_output_name] = state[classical_energy_name]
    return state, parameters


def collect_quantum_energy(sim, state, parameters, **kwargs):
    """
    Collects the quantum energy from the state object and stores it in the
    output dictionary.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    quantum_energy_name : str, default : "quantum_energy"
        Name of the quantum energy in the state object.
    quantum_energy_output_name : str, default : "quantum_energy"
        Name of the quantum energy in the output dictionary.

    .. rubric:: Modifications
    state["output_dict"][quantum_energy_output_name] : ndarray
        Stores the quantum energy.
    """
    quantum_energy_name = kwargs.get("quantum_energy_name", "quantum_energy")
    quantum_energy_output_name = kwargs.get(
        "quantum_energy_output_name", "quantum_energy"
    )
    state["output_dict"][quantum_energy_output_name] = state[quantum_energy_name]
    return state, parameters
