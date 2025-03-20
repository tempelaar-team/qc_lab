.. _algorithms:

Algorithms
----------

Algorithms in QC Lab are classes that can be paired with models to carry out a quantum-classical dynamics simulation.
Like models, each algorithm in QC Lab has a set of parameters that allow users to control particular attributes of the 
algorithm. The contents and structure of an algorithm class are described in the Developer Guide.

The algorithms currently available in QC Lab can be imported from the algorithms module:

.. code-block:: python

    from qclab.algorithms import MeanField

After which they can be instantiated with a set of parameters:

.. code-block:: python
    
    algorithm = MeanField(parameters=None)

Here there are no parameters passed to the `MeanField` algorithm.

.. toctree::
    :maxdepth: 1
    :caption: Algorithms

    mf_algorithm
    fssh_algorithm