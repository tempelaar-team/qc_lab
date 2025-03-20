.. _models:

Models
------

Models in QC Lab are classes that define the physical properties of the system. Each model in QC Lab 
has a set of parameters that can be accessed by the user when the model is instantiated. The contents 
and structure of a model class are described in the Developer Guide. 

The models currently available in QC Lab can be imported from the models module:

::

    from qclab.models.spin_boson import SpinBosonModel

After which they can be instantiated with a set of parameters:

::
    
    model = SpinBosonModel(parameters=dict(temp=1, V=0.5, E=0.5, A=100, W=0.1, l_reorg=0.005, boson_mass=1))


See the subsequent pages for more information on the models available in QC Lab including their default 
parameters. 

.. toctree::
    :maxdepth: 1
    :caption: Models

    spin_boson_model
    holstein_model
