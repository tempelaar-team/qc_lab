.. _mf-algorithm:
Mean-Field Dynamics 
~~~~~~~~~~~~~~~~~~~

The `qclab.algorithms.MeanField` class implements the mean-field (Ehrenfest) dynamics algorithm according to `Tully 1998 <https://doi.org/10.1039/A801824C>`_.
For a more recent reference we suggest `Krotz et al. 2021 <https://doi.org/10.1063/5.0053177>`_.

Settings
--------

The mean-field algorithm has no default settings.




Output Variables
----------------

The following table lists the default output variables for the `MeanField` class.

.. list-table:: MeanField Output Variables
   :header-rows: 1

   * - Variable name
     - Description
   * - `classical_energy`
     - Energy in the classical subsystem
   * - `quantum_energy`
     - Energy in the quantum subsystem
   * - `dm_db`
     - Diabatic density matrix