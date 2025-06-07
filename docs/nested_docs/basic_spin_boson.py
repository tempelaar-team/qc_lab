import numpy as np
import matplotlib.pyplot as plt
from qc_lab import Simulation
from qc_lab.models import SpinBoson
from qc_lab.algorithms import MeanField
from qc_lab.dynamics import serial_driver


# Initialize the simulation object.
sim = Simulation()
# Equip it with a SpinBoson model object.
sim.model = SpinBoson()
# Attach the MeanField algorithm.
sim.algorithm = MeanField()
# Initialize the diabatic wavefunction.
sim.state.wf_db = np.array([1, 0], dtype=complex)
# Run the simulation.
data = serial_driver(sim)

t = data.data_dict["t"]
populations = np.real(np.einsum("tii->ti", data.data_dict["dm_db"]))
classical_energy = data.data_dict["classical_energy"]
quantum_energy = data.data_dict["quantum_energy"]

plt.plot(t, populations[:, 0], label="upper state")
plt.plot(t, populations[:, 1], label="lower state")
plt.xlabel("time")
plt.ylabel("population")
plt.legend()
plt.savefig("populations.png", dpi=150)
plt.close()

plt.plot(t, classical_energy - classical_energy[0], label="classical energy")
plt.plot(t, quantum_energy - quantum_energy[0], label="quantum energy")
plt.plot(
    t,
    classical_energy + quantum_energy - classical_energy[0] - quantum_energy[0],
    label="total energy",
)
plt.xlabel("time")
plt.ylabel("energy")
plt.legend()
plt.savefig("energies.png", dpi=150)
plt.close()
