.. _data_object:

Data Object
-----------

The data object in QC Lab is used to store the results of a simulation, it can be imported from `qc_lab.Data`. Typically, it is created by the dynamics driver
and contains the output variables in the dictionary `data.data_dict`. 

Assuming we have a data object or have initialized one as:

.. code-block:: python

    from qc_lab import Data
    data = Data()


Methods
~~~~~~~

The data object also contains the method `data.add_data` which can be used to combine two data objects. 
Additionally, it contains methods that allwo you to write the data to an `h5` archive and also to read a written archive back into a data object.

.. function:: data.add_data(new_data)

    Adds the data from new_data to the data object. Joins the seeds together and sums any 
    data with the same keys.

    :param new_data: An input data object. 

.. function:: data.save(filename)

    Saves the data in the data object as an HDF5 file.

    :param filename: A string providing the name of the file to save the data to.

.. function:: data.load(filename)

    Loads data into the data object from an HDF5 file.

    :param filename: A string providing the name of the file to load the data from.
    :returns: The Data object with the loaded data.


Attributes
~~~~~~~~~~

The `data.data_dict` dictionary also contains the time array `t` giving the time points at which the output variables were 
calculated, as well as the `norm_factor` which is used to normalize the output variables, and the `seed` array which contains
the random seeds used in the simulation.

.. attribute:: data.data_dic

    A dictionary containing the data stored in the Data object. By default, this contains the seeds 
    (`data.data_dict["seed"]`), normalization factor (`data.data_dict["norm_factor"]`), output timesteps 
    (`data.data_dict["t"]`), and any other collect variables that were collected during the simulation (specified in `algorithm.collect_variables`).


Example Usage
~~~~~~~~~~~~~


.. code-block:: python

    import numpy as np
    from qc_lab import Simulation, Data
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
    # Run it again on the last data object
    data = serial_driver(sim, data=data)

    # Print the keys of the data dictionary.
    print(data.data_dict.keys())

    # Write the data to an h5 archive.
    data.save("spin_boson_data.h5")

    # Read the data back into a data object.
    data_read = Data().load("spin_boson_data.h5")