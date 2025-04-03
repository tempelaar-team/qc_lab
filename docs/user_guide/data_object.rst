.. _data_object:

Data Object
-----------

The Data object in QC Lab is used to store the results of simulations. Data objects come with a variety of methods that are intended as 
quality-of-life features to make using QC Lab easier. Here we review those methods.

Assuming we have a data object or have initialized one as:

.. code-block:: python

    from qc_lab import Data
    data = Data()


Methods
~~~~~~~

.. function:: data.add_data(new_data)

    Adds the data from new_data to the data object. Joins the seeds together and sums any 
    data with the same keys.

    :param new_data: An input data object. 

.. function:: data.save_as_h5(filename)

    Saves the data in the data object as an HDF5 file.

    :param filename: A string providing the name of the file to save the data to.

.. function:: data.load_from_h5(filename)

    Loads data into the data object from an HDF5 file.

    :param filename: A string providing the name of the file to load the data from.
    :returns: The Data object with the loaded data.


Attributes
~~~~~~~~~~

.. attribute:: data.data_dic

    A dictionary containing the data stored in the Data object. By default, this contains the seeds 
    (`data.data_dict["seed"]`)
    used in the simulation. In a new data object, this list is empty.