QC Lab: A package for quantum-classical modeling
================================================


**QC Lab** is a Python package designed for implementing and executing quantum-classical (QC) dynamics simulations.
It offers an environment for developing physical models and QC algorithms which enables algorithms and models to be combined arbitrarily.
QC Lab comes with a variety of already implemented models and algorithms which we hope will encourage new researchers to explore the field of quantum-classical dynamics. Users that implement their own models and algorithms will have the opportunity to contribute them to QC Lab to form a growing library of quantum-classical dynamics tools.


**QC Lab** is developed and maintained by the Tempelaar Team in the Chemistry Department of Northwestern University in Evanston, Illinois, USA.


The documentation for QC Lab can be found at https://tempelaar-team.github.io/qclab/.


Capabilities
------------

Dynamics Algorithms
```````````````````

The following algorithms are implemented making use of the complex-classical coordinate formalism established in [1].


* Mean-field (Ehrenfest) dynamics [2]
* Fewest-switches surface hopping (FSSH) dynamics [3]

Model Systems
`````````````

* Spin-boson model [4]
* Holstein lattice model [5]
* Fenna-Matthews-Olson (FMO) complex [6, 7]


Installing qclab
-----------------

QC Lab can be installed from the Python Package Index (PyPI) by executing::

   pip install qclab

To install QC Lab without h5py or numba support, execute::

   pip install qclab --no-deps
   pip install numpy tqdm

QC Lab can be installed from source by downloading the repository and executing::

   pip install ./

from inside its topmost directory (where the `pyproject.toml` file is located).


Bibliography
------------

1. Miyazaki, K.; Krotz, A.; Tempelaar, R. Mixed Quantum-Classical Dynamics under Arbitrary Unitary Basis Transformations. J. Chem. Theory Comput. 2024, 20 (15), 6500-6509. https://doi.org/10.1021/acs.jctc.4c00555
2. Tully, J. C. Mixed Quantum–Classical Dynamics. Faraday Discuss. 1998, 110 (0), 407–419. https://doi.org/10.1039/A801824C.
3. Hammes‐Schiffer, S.; Tully, J. C. Proton Transfer in Solution: Molecular Dynamics with Quantum Transitions. J. Chem. Phys. 1994, 101 (6), 4657–4667. https://doi.org/10.1063/1.467455.
4. Tempelaar, R.; Reichman, D. R. Generalization of Fewest-Switches Surface Hopping for Coherences. The Journal of Chemical Physics 2018, 148 (10), 102309. https://doi.org/10.1063/1.5000843.
5. Krotz, A.; Provazza, J.; Tempelaar, R. A Reciprocal-Space Formulation of Mixed Quantum–Classical Dynamics. J. Chem. Phys. 2021, 154 (22), 224101. https://doi.org/10.1063/5.0053177.
6. Fenna, R. E. & Matthews, B. W. Chlorophyll arrangement in a bacteriochlorophyll protein from Chlorobium limicola. Nature 258, 573–577 (1975). https://doi.org/10.1038/258573a0.
7. Mulvihill, E.; Lenn, K. M.; Gao, X.; Schubert, A.; Dunietz, B. D.; Geva, E. Simulating Energy Transfer Dynamics in the Fenna–Matthews–Olson Complex via the Modified Generalized Quantum Master Equation. The Journal of Chemical Physics 2021, 154 (20), 204109. https://doi.org/10.1063/5.0051101.
