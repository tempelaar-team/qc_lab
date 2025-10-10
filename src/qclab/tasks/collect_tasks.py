"""
This module contains tasks that are used to collect data from the state or
parameters objects into the output dictionary of the state object.
"""


def collect_t(sim, state, parameters, **kwargs):
    """
    Collects the time in the state object.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    t_var_name : str, default : "t"
        Name of the time variable in the state object.
    t_output_name : str, default : "t"

    .. rubric:: Variable Modifications
    state.output_dict[t_output_name] : ndarray
        stores the current time in each trajectory.
    """
    t_var_name = kwargs.get("t_var_name", "t")
    t_output_name = kwargs.get("t_output_name", "t")
    state.output_dict[t_output_name] = getattr(state, t_var_name)
    return state, parameters


def collect_dm_db(sim, state, parameters, **kwargs):
    """
    Collects the diabatic density matrix in the state object.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    dm_db_var_name : str, default : "dm_db"
        Name of the diabatic density matrix variable in the state object.
    dm_db_output_name : str, default : "dm_db"
        Name of the output variable for the diabatic density matrix.

    .. rubric:: Variable Modifications
    state.output_dict[dm_db_output_name] : ndarray
        stores the diabatic density matrix.
    """
    dm_db_var_name = kwargs.get("dm_db_var_name", "dm_db")
    dm_db_output_name = kwargs.get("dm_db_output_name", "dm_db")
    state.output_dict[dm_db_output_name] = getattr(state, dm_db_var_name)
    return state, parameters


def collect_classical_energy(sim, state, parameters, **kwargs):
    """
    Collects the classical energy in the state object.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    classical_energy_var_name : str, default : "classical_energy"
        Name of the classical energy variable in the state object.
    classical_energy_output_name : str, default : "classical_energy"
        Name of the output variable for the classical energy.

    .. rubric:: Variable Modifications
    state.output_dict[classical_energy_output_name] : ndarray
        stores the classical energy.
    """
    classical_energy_var_name = kwargs.get(
        "classical_energy_var_name", "classical_energy"
    )
    classical_energy_output_name = kwargs.get(
        "classical_energy_output_name", "classical_energy"
    )
    state.output_dict[classical_energy_output_name] = getattr(
        state, classical_energy_var_name
    )
    return state, parameters


def collect_quantum_energy(sim, state, parameters, **kwargs):
    """
    Collects the quantum energy in the state object.

    .. rubric:: Required Constants
    None

    .. rubric:: Keyword Arguments
    quantum_energy_var_name : str, default : "quantum_energy"
        Name of the quantum energy variable in the state object.
    quantum_energy_output_name : str, default : "quantum_energy"
        Name of the output variable for the quantum energy.

    .. rubric:: Variable Modifications
    state.output_dict[quantum_energy_output_name] : ndarray
        stores the quantum energy.
    """
    quantum_energy_var_name = kwargs.get("quantum_energy_var_name", "quantum_energy")
    quantum_energy_output_name = kwargs.get(
        "quantum_energy_output_name", "quantum_energy"
    )
    state.output_dict[quantum_energy_output_name] = getattr(
        state, quantum_energy_var_name
    )
    return state, parameters
