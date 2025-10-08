.. _tully_problems:


The Tully Problems
------------------

The Tully problems are a set of model problems used to evaluate the performance of the Fewest-Switches Surface Hopping (FSSH) algorithm in 
the paper `Tully 1990 <https://doi.org/10.1063/1.459170>`_. Three problems were originally proposed by Tully, the simple avoided crossing, 
the dual avoided crossing, and the extended coupling problem. In each problem the trajectories are initialized at the same position and momentum.

In all problems we take the quantum Hamiltonian to be zero and the classical Hamiltonian to be a free particle. In the following sections we 
detail the quantum-classical Hamiltonian for each problem and tabulate the constants used in each model class.

In each problem, the quantum-classical Hamiltonian is given by

.. math::

    \hat{H}_{\mathrm{q-c}}(q) = \begin{pmatrix}
    V_{11}(q) & V_{12}(q) \\
    V_{12}^{*}(q) & V_{22}(q)
    \end{pmatrix}


Tully 1: Simple Avoided Crossing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simple avoided crossing model is `qc_lab.models.TullyProblemOne` and the quantum-classical Hamiltonian is given by

.. math::

    V_{11} = -V_{22} = \mathrm{sign}(q)A(1-\exp(-B\,\mathrm{sign}(q) q))

and

.. math::

    V_{12} = C\exp(-D q^{2})

The model constants are:

.. list-table:: HolsteinLatticeModel constants
   :header-rows: 1

   * - Parameter (symbol)
     - Description
     - Default Value
   * - `mass` :math:`(m)`
     - Mass of the classical particle
     - 2000
   * - `init_position`
     - Initial position of the classical particle
     - -25
   * - `init_momentum`
     - Initial momentum of the classical particle
     - 10
   * - `A` :math:`(A)`
     - Hamiltonian parameter
     - 0.01
   * - `B` :math:`(B)`
     - Hamiltonian parameter
     - 1.6
   * - `C` :math:`(C)`
     - Hamiltonian parameter
     - 0.005
   * - `D` :math:`(D)`
     - Hamiltonian parameter
     - 1.0


Tully 2: Dual Avoided Crossing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The dual avoided crossing model is `qc_lab.models.TullyProblemTwo` and the quantum-classical Hamiltonian is given by

.. math::

    V_{11} = 0

and

.. math::

    V_{22} = -A\exp(-B q^{2}) + E_{0}


.. math::

    V_{12} = C\exp(-D q^{2})

The model constants are:

.. list-table:: HolsteinLatticeModel constants
   :header-rows: 1

   * - Parameter (symbol)
     - Description
     - Default Value
   * - `mass` :math:`(m)`
     - Mass of the classical particle
     - 2000
   * - `init_position`
     - Initial position of the classical particle
     - -25
   * - `init_momentum`
     - Initial momentum of the classical particle
     - 10
   * - `A` :math:`(A)`
     - Hamiltonian parameter
     - 0.01
   * - `B` :math:`(B)`
     - Hamiltonian parameter
     - 0.28
   * - `C` :math:`(C)`
     - Hamiltonian parameter
     - 0.015
   * - `D` :math:`(D)`
     - Hamiltonian parameter
     - 0.06
   * - `E_0` :math:`(E_{0})`
     - Hamiltonian parameter
     - 0.05



Tully 3: Extended Coupling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The extended coupling model is `qc_lab.models.TullyProblemThree` and the quantum-classical Hamiltonian is given by

.. math::

    V_{11} = -V_{22}=A

and for :math:`q < 0`

.. math::

    V_{12} = B\exp(C q)

and for :math:`q > 0`

.. math::

    V_{12} = B(2-\exp(-C q))

The model constants are:

.. list-table:: HolsteinLatticeModel constants
   :header-rows: 1

   * - Parameter (symbol)
     - Description
     - Default Value
   * - `mass` :math:`(m)`
     - Mass of the classical particle
     - 2000
   * - `init_position`
     - Initial position of the classical particle
     - -25
   * - `init_momentum`
     - Initial momentum of the classical particle
     - 10
   * - `A` :math:`(A)`
     - Hamiltonian parameter
     - 0.0006
   * - `B` :math:`(B)`
     - Hamiltonian parameter
     - 0.1
   * - `C` :math:`(C)`
     - Hamiltonian parameter
     - 0.9
