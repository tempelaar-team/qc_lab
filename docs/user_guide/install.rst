.. _install:

====================
Installing QC Lab
====================

This guide walks you through installing QC Lab by downloading its source from GitHub, unpacking it, and installing it with pip.

Requirements
------------
- Python 3.8 or newer
- pip (Python package installer)
- git (optional, for cloning the repository directly)

Downloading the source
----------------------
You can obtain the QC Lab source code in one of two ways:

1. **Clone via git** (recommended if you plan to pull updates):
   
   .. code-block:: bash

      git clone https://github.com/tempelaar-team/qc_lab.git

2. **Download a ZIP archive** of the latest `main` branch:
   
   .. code-block:: bash

      wget https://github.com/tempelaar-team/qc_lab/archive/refs/heads/main.zip -O qc_lab-main.zip

   Then unpack the archive:

   .. code-block:: bash
      unzip qc_lab-main.zip

   If you do not have ``wget`` or ``unzip``, you can alternatively navigate to the repository page in your browser:
   https://github.com/tempelaar-team/qc_lab and click **Code → Download ZIP**, then extract it with your preferred archive manager.

Unpacking (if using ZIP or TAR)
-------------------------------
If you downloaded a ZIP file, make sure it is unpacked:

.. code-block:: bash

   unzip qc_lab-main.zip       # Produces folder `qc_lab-main/`

If you downloaded a tarball (for example, if GitHub provides a `.tar.gz`), unpack with:

.. code-block:: bash

   tar -xzf qc_lab-main.tar.gz # Produces folder `qc_lab-main/`

Ensure your working directory now contains a folder named `qc_lab/` (if you cloned via git) or `qc_lab-main/` (if you downloaded and unpacked).

Installing with pip
-------------------
Navigate into the QC Lab source directory and install via pip:

.. code-block:: bash

   # If you used git:
   cd qc_lab/

   # If you downloaded and unpacked ZIP/TAR:
   cd qc_lab-main/

   # Install QC Lab (and its Python dependencies)
   pip install .

If you want to install it without using numba precompilation (which 
may improve compatibility with some systems but be much slower), you can use:

.. code-block:: bash

   QC_LAB_DISABLE_NUMBA=1 pip install .

You can also install QC Lab without `h5py` support by using:

.. code-block:: bash

   QC_LAB_DISABLE_H5PY=1 pip install .

If you are interesting in building the documentation, you can install the optional dependencies with:

.. code-block:: bash

   pip install .[docs]



That’s it! QC Lab should now be installed and ready for us
