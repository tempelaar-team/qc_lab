"""
setup.py
"""

from setuptools import setup, find_packages

VERSION = "0.1.0a3"
DESCRIPTION = "QC Lab: a python package for quantum-classical modeling."
LONG_DESCRIPTION = "QC Lab is a python package for quantum-classical modeling. It was developed in 2025 by the Tempelaar Team at Northwestern University."
setup(
    name="qc_lab",
    version=VERSION,
    author="Tempelaar Team",
    author_email="roel.tempelaar@northwestern.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    python_requires=">3.9,<3.14",
    install_requires=["numpy<=1.26,>=1.22", "tqdm","h5py","numba<=0.60,>=0.58","matplotlib"],
    extras_requires={"test":["pytest", "mpi4py"]},
    keywords=["surface hopping", "mixed quantum-classical dynamics",
              "theoretical chemistry", "ehrenfest", "python", "quantum-classical"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Linux :: Linux",
        "License :: Apache License 2.0",
    ]
)
