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

Downloading the Source
----------------------
You can obtain the QC Lab source code in one of two ways:

1. **Clone via git** (recommended if you plan to pull updates):
   
   .. code-block:: bash

      git clone https://github.com/tempelaar-team/qc_lab.git

2. **Download a ZIP archive** of the latest `main` branch:
   
   .. code-block:: bash

      # Download the ZIP archive
      wget https://github.com/tempelaar-team/qc_lab/archive/refs/heads/main.zip -O qc_lab-main.zip

      # Unzip the archive
      unzip qc_lab-main.zip

   If you do not have `wget` or `unzip`, you can alternatively navigate to the repository page in your browser:
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

This command will:

- Read the `setup.py` in the source directory.
- Install QC Lab and any required dependencies into your active Python environment.

If you wish to install in “editable” (development) mode so that changes in the source tree are immediately available, run:

.. code-block:: bash

   pip install -e .

Verifying the Installation
--------------------------
After installation, verify that QC Lab is available:

.. code-block:: bash

   python -c "import qc_lab; print(qc_lab.__version__)"

You should see the QC Lab version printed without errors. 

That’s it! QC Lab should now be installed and ready for us
