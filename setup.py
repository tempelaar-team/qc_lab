"""Setup script for QC Lab."""
from pathlib import Path
from setuptools import setup, find_packages

VERSION = "0.1.0a3"
DESCRIPTION = "QC Lab: a python package for quantum-classical modeling."
README = Path(__file__).with_name("README.rst")
LONG_DESCRIPTION = README.read_text(encoding="utf-8")

setup(
    name="qc_lab",
    version=VERSION,
    author="Tempelaar Team",
    author_email="roel.tempelaar@northwestern.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9,<3.14",
    install_requires=[
        "numpy>=1.22,<=1.26",
        "tqdm",
        "h5py",
        "numba>=0.58,<=0.60",
        "matplotlib",
    ],
    extras_require={"test": ["pytest", "mpi4py"]},
    keywords=[
        "surface hopping",
        "mixed quantum-classical dynamics",
        "theoretical chemistry",
        "ehrenfest",
        "python",
        "quantum-classical",
    ],
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Linux :: Linux",
        "License :: OSI Approved :: Apache Software License",
    ],
)
