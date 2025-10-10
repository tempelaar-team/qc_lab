.. _install:

====================
Installing QC Lab
====================

This guide walks you through installing QC Lab from source or from PyPI, using pip.

Requirements
------------
- Python 3.8 or newer
- pip (Python package installer)
- git (optional, for cloning the repository directly)


Installing from PyPI
--------------------
QC Lab can be installed from the Python Package Index (PyPI) by executing


.. code-block:: bash

      pip install qclab


To install QC Lab without h5py or numba support, execute


.. code-block:: bash

      pip install qclab --no-deps
      pip install numpy tqdm


Installing from source
----------------------

You can install QC Lab from source by downloading the repository and executing


.. code-block:: bash

      pip install ./

from inside its topmost directory (where the `pyproject.toml` file is located).


That’s it! QC Lab should now be installed and ready for us
