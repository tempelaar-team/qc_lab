"""
This module runs the spin-boson model using the sparse and vectorization
decorators and compares the result against a reference calculation.
"""

import pytest
import numpy as np
import os
import time
from qclab import Simulation, Data
import qclab.ingredients as ingredients
from qclab.algorithms import MeanField
from qclab.functions import vectorize_ingredient, make_ingredient_sparse
from qclab.dynamics import serial_driver
from qclab.models import SpinBoson
try: 
    from tests.reference_settings import model_sim_settings, model_settings
except ImportError:
    from reference_settings import model_sim_settings, model_settings


def test_vectorize_decorator():
    """
    This test checks the ``vectorize_ingredient`` decorator.

    It runs a spin-boson model simulation using the ``MeanField`` algorithm
    where the quantum-classical and classical Hamiltonians are vectorized using
    the decorator. The results are compared against a reference calculation.
    """
    print('Testing vectorization decorator')
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    sim = Simulation(model_sim_settings["SpinBoson"])
    sim.model = SpinBoson(model_settings["SpinBoson"])
    sim.model.initialize_constants()
    sim.algorithm = MeanField()

    @vectorize_ingredient
    def h_q(model, parameters, **kwargs):
        E = model.constants.get("E")
        V = model.constants.get("V")
        return np.array([[E, V], [V, -E]])

    @vectorize_ingredient
    def h_qc(model, parameters, **kwargs):
        z = kwargs["z"]
        A = model.constants.get("A")
        diag_coupling = model.constants.diagonal_linear_coupling
        out = np.zeros((2, 2), dtype=complex)
        for i in range(A):
            out[0, 0] += diag_coupling[0, i] * (z[i] + np.conj(z[i]))
            out[1, 1] += diag_coupling[1, i] * (z[i] + np.conj(z[i]))
        return out

    sim.model.ingredients.append(("h_q", h_q))
    sim.model.ingredients.append(("h_qc", h_qc))
    sim.initial_state.wf_db = np.zeros(
        sim.model.constants.num_quantum_states, dtype=complex
    )
    sim.initial_state.wf_db[0] = 1j
    st = time.time()
    data = serial_driver(sim)
    et = time.time()
    print(f"Finished in {et - st:.2f} seconds.")
    data_correct = Data().load(
        os.path.join(reference_folder, "SpinBoson_MeanField.h5")
    )
    for key, val in data.data_dict.items():
        np.testing.assert_allclose(
            val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
        )


def test_make_sparse_decorator():
    """
    This test checks the ``make_ingredient_sparse`` decorator.

    It runs a spin-boson model simulation using the ``MeanField`` algorithm
    where the gradient of the quantum-classical Hamiltonian is made sparse using
    the decorator. The results are compared against a reference calculation.
    """
    print('Testing sparsity decorator')
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    sim = Simulation(model_sim_settings["SpinBoson"])
    sim.model = SpinBoson(model_settings["SpinBoson"])
    sim.model.initialize_constants()
    sim.algorithm = MeanField()

    @make_ingredient_sparse
    def dh_qc_dzc(model, parameters, **kwargs):
        inds, mels, shape = ingredients.dh_qc_dzc_diagonal_linear(model, parameters, **kwargs)
        out = np.zeros(shape, dtype=complex)
        np.add.at(out, inds, mels)
        return out

    sim.model.ingredients.append(("dh_qc_dzc", dh_qc_dzc))
    sim.initial_state.wf_db = np.zeros(
        sim.model.constants.num_quantum_states, dtype=complex
    )
    sim.initial_state.wf_db[0] = 1j
    st = time.time()
    data = serial_driver(sim)
    et = time.time()
    print(f"Finished in {et - st:.2f} seconds.")
    data_correct = Data().load(
        os.path.join(reference_folder, "SpinBoson_MeanField.h5")
    )
    for key, val in data.data_dict.items():
        np.testing.assert_allclose(
            val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
        )

def test_sparse_and_vectorize_decorators():
    """
    This test checks the combination of the ``make_ingredient_sparse`` and ``vectorize_ingredient`` decorators.

    It runs a spin-boson model simulation using the ``MeanField`` algorithm
    where the gradient of the quantum-classical Hamiltonian is made sparse using
    the ``make_ingredient_sparse`` decorator and vectorized using the
    ``vectorize_ingredient`` decorator. The results are compared against a reference calculation.
    """
    print('Testing sparsity and vectorization decorators')
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    sim = Simulation(model_sim_settings["SpinBoson"])
    sim.model = SpinBoson(model_settings["SpinBoson"])
    sim.model.initialize_constants()
    sim.algorithm = MeanField()

    @make_ingredient_sparse
    @vectorize_ingredient
    def dh_qc_dzc(model, parameters, **kwargs):
        A = model.constants.get("A")
        diag_coupling = model.constants.diagonal_linear_coupling
        out = np.zeros((A, 2, 2), dtype=complex)
        for i in range(A):
            out[i, 0, 0] = diag_coupling[0, i]
            out[i, 1, 1] = diag_coupling[1, i]
        return out

    sim.model.ingredients.append(("dh_qc_dzc", dh_qc_dzc))
    sim.initial_state.wf_db = np.zeros(
        sim.model.constants.num_quantum_states, dtype=complex
    )
    sim.initial_state.wf_db[0] = 1j
    st = time.time()
    data = serial_driver(sim)
    et = time.time()
    print(f"Finished in {et - st:.2f} seconds.")
    data_correct = Data().load(
        os.path.join(reference_folder, "SpinBoson_MeanField.h5")
    )
    for key, val in data.data_dict.items():
        np.testing.assert_allclose(
            val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
        ) 


if __name__ == "__main__":
    st = time.time()
    test_vectorize_decorator()
    et1 = time.time()
    print(f"Vectorization tests completed in {et1 - st:.2f} seconds.")
    test_make_sparse_decorator()
    et2 = time.time()
    print(f"Sparsity tests completed in {et2 - et1:.2f} seconds.")
    test_sparse_and_vectorize_decorators()
    et3 = time.time()
    print(f"Sparsity and vectorization tests completed in {et3 - et2:.2f} seconds.")
    print(f"All tests completed in {et3 - st:.2f} seconds.")