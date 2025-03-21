.. model_dev:

Model Development
================

This page will guide you in the construction of a new Model Class for use in QC Lab.

The model class describes a physical model as a set of functions referred to as "ingredients". 
QC Lab is designed to accommodate a minimal model that consists of only a quantum-classical Hamiltonian. 
However, by incorporating additional ingredients, such as analytic gradients, the performance of QC Lab can be greatly improved. 
We will first describe the construction of a minimal model class, and then discuss how to incorporate additional ingredients.

.. contents:: Table of Contents
   :local:

Minimal Model Class
-------------------

A physical model in QC Lab is assumed to consist of a Hamiltonian of the form:

.. math::

    \hat{H}(\boldsymbol{z}) = \hat{H}_{\mathrm{q}} + \hat{H}_{\mathrm{q-c}}(\boldsymbol{z}) + H_{\mathrm{c}}(\boldsymbol{z})

where :math:`\hat{H}_{\mathrm{q}}` is the quantum Hamiltonian, :math:`\hat{H}_{\mathrm{q-c}}` is the quantum-classical coupling Hamiltonian,
and :math:`H_{\mathrm{c}}` is the classical Hamiltonian. :math:`\boldsymbol{z}` is a complex-valued classical coordinate that defines the
classical degrees of freedom.

Before describing how the model ingredients should be structured in the model class, we will first describe the `__init__` method of the model class 
which is responsible for initializing the default model constants and any input constants into the set of constants needed for QC Lab and all 
of the model ingredients to run. 

::

    from qclab import Model  # import the model class

    # create a minimal spin-boson model subclass
    class MinimalSpinBoson(Model):
        def __init__(self, constants=None):
            if constants is None:
                constants = {}
            self.default_constants = {
                'temp': 1, 'V': 0.5, 'E': 0.5, 'A': 100, 'W': 0.1,
                'l_reorg': 0.02 / 4, 'boson_mass': 1
            }
            super().__init__(self.default_constants, constants)

In the above example, the `__init__` method takes an optional `constants` dictionary which is added to the `default_constants` dictionary by 
`super().__init__`. The `default_constants` dictionary contains the default input constants for the model. These constants are independent from the 
internal constants required by QC Lab to function and are instead drawn from the analytic formulation of the spin-boson model. 

.. math::
    
    \hat{H}_{\mathrm{q}} = \left(\begin{array}{cc} E & V \\ V & -E \end{array}\right)

.. math::

    \hat{H}_{\mathrm{q-c}} = \sigma_{z} \sum_{\alpha}^{A}  \frac{g_{\alpha}}{\sqrt{2mh_{\alpha}}} \left(z^{*}_{\alpha} + z_{\alpha}\right)

.. math::

    H_{\mathrm{c}} = \sum_{\alpha}^{A} \omega_{\alpha} z^{*}_{\alpha} z_{\alpha}

Here :math:`\sigma_{z}` is the Pauli-z matrix (:math:`\sigma_{z}=\vert0\rangle\langle 0\vert - \vert 1\rangle\langle 1\vert`), :math:`g_{\alpha}` is the coupling strength,
:math:`m` is the boson mass, :math:`h_{\alpha}` is the z-coordinate parameter (which here we may take to correspond to the frequencies: :math:`h_{\alpha}=\omega_{\alpha}`),
 and :math:`A` is the number of bosons. We sample the frequencies and coupling strengths from a Debye spectral density which is discretized to obtain

.. math::

    \omega_{\alpha} = \Omega\tan\left(\frac{\alpha - 1/2}{2A}\pi\right)

.. math::

    g_{\alpha} = \omega_{\alpha}\sqrt{\frac{2\lambda}{A}}

Where :math:`\Omega` is the characteristic frequency and :math:`\lambda` is the reorganization energy. 

In a spin-boson model, the number of bosons :math:`A` can be quite large (e.g. 100). Rather than specifying every value of :math:`\omega_{\alpha}` 
and :math:`g_{\alpha}` in the input constants, we can instead specify the characteristic frequency :math:`\Omega` and the reorganization energy :math:`\lambda`.
We can then use an internal function to generate the remaining constants needed by the model and any constants needed by QC Lab. 


Initialization functions
~~~~~~~~~~~~~~~~~~~~~~~~


This is accomplished by specifying a list of function called `initialization_functions` as an attribute of the model class. These functions will 
be executed by the Model superclass when the model is instantiated or whenever a constant is changed. Because it is a list of functions, it is executed 
from start to finish, so the order of the functions can sometimes be important. The first function we will specify initializes the model constants and make 
use of the `get` method of the Constants object (`self.constants`) to obtain the input constants. We use the get method of the 
`default_constants` dictionary (`self.default_constants`) to obtain the default constants in the event that the input constant has not been specified.

Here, we initialize the constants needed by QC Lab which are the number of classical coordinates (`sim.model.constants.num_classical_coordinates`),
the number of quantum states (`sim.model.constants.num_quantum_states`), the classical coordinate weight (`sim.model.constants.classical_coordinate_weight`),
and the classical coordinate mass (`sim.model.constants.classical_coordinate_mass`). Because the classical Hamiltonian is a harmonic oscillator,
we set the classical coordinate weight to the oscillator frequencies (`sim.model.constant.w`) even though these frequencies are not strictly speaking a 
constant needed by QC Lab (they would otherwise be specified in the initialization function for the classical Hamiltonian).

::

    def initialize_constants_model(self):
        num_bosons = self.constants.get("A", self.default_constants.get("A"))
        char_freq = self.constants.get("W", self.default_constants.get("W"))
        boson_mass = self.constants.get(
            "boson_mass", self.default_constants.get("boson_mass")
        )
        self.constants.w = char_freq * np.tan(
            ((np.arange(num_bosons) + 1) - 0.5) * np.pi / (2 * num_bosons)
        )
        # The following constants are required by QC Lab
        self.constants.num_classical_coordinates = num_bosons
        self.constants.num_quantum_states = 2
        self.constants.classical_coordinate_weight = self.constants.w
        self.constants.classical_coordinate_mass = boson_mass * np.ones(num_bosons)


Next we define a function which initializes the constants needed by the classical Hamiltonian, quantum Hamiltonian, and quantum-classical Hamiltonian. Be aware that the 
constants we define in the functions are dictated by the requirements of the ingredients (these are defined in the :ref:`ingredients` section).


::

    def initialize_constants_h_c(self):
        """
        Initialize the constants for the classical Hamiltonian.
        """
        w = self.constants.get("w", self.default_constants.get("w"))
        self.constants.harmonic_oscillator_frequency = w


    def initialize_constants_h_qc(self):
        """
        Initialize the constants for the quantum-classical coupling Hamiltonian.
        """
        num_bosons = self.constants.get("A", self.default_constants.get("A"))
        w = self.constants.get("w", self.default_constants.get("w"))
        l_reorg = self.constants.get("l_reorg", self.default_constants.get("l_reorg"))
        self.constants.spin_boson_coupling = w * np.sqrt(2 * l_reorg / num_bosons)

    def initialize_constants_h_q(self):
        """
        Initialize the constants for the quantum Hamiltonian. None are required in this case.
        """

These are all placed into the `initialization_functions` list in the model class.

::

    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_c,
        initialize_constants_h_qc,
        initialize_constants_h_q,
    ]


Now you can check that the updating of model constants is functioning properly by changing one of the input constants (A for example) and then checking that
the coupling strengths are updated appropriately:

::

    model = MinimalSpinBoson()
    model.constants.A = 10
    print('coupling strengths: ', model.constants.spin_boson_coupling)  # should be a list of length 10
    model.constants.A = 5
    print('coupling strengths: ', model.constants.spin_boson_coupling)  # should be a list of length 5


Ingredients
~~~~~~~~~~~

Now we can add the minimal set of ingredients to the model class. The ingredients are the quantum Hamiltonian, 
the quantum-classical coupling Hamiltonian, and the classical Hamiltonian. The ingredients in a model class 
take a standard form which is required by QC Lab. 


A generic ingredients has as arguments the model class itself, the constants object containing time independent quantities (stored in sim.model.constants), and 
the parameters object which contain potentially time-dependent quantities (stored in sim.model.parameters). The ingredients can also take additional keyword arguments
which are passed to the ingredient when it is called. The ingredients return the result of the calculation directly. Typically, users will never call ingredients as they 
are internal functions used by QC Lab to define the model.

As an example we will use the quantum Hamiltonian. Importantly, QC Lab is a vectorized code capable of calculating multiple quantum-classical trajectories simultaneously. 
As a result, the ingredients must also be vectorized, meaning that they accept as input quantities with an additional dimension corresponding to the number of trajectories 
(this is taken to be the first dimension as a convention). The quantum Hamiltonian is a 2x2 matrix and so the vectorized quantum Hamiltonian is a 3D array with shape
(len(parameters.seed), 2, 2) where the number of trajectories is given by the number of seeds in the parameters object. 

Rather than writing a vectorized ingredient (which will be discussed later) we can invoke a decorator (`ingredients.vectorize`) which will automatically vectorize the ingredient
at the cost of some performance (it is strongly recommended to write vectorized ingredients as a first pass for performance optimization).

.. code-block:: python

    import qc_lab.ingredients as ingredients

    @ingredients.vectorize_ingredient
    def h_q(self, constants, parameters, **kwargs):
        E = constants.E
        V = constants.V
        return np.array([[E, V], [V, -E]], dtype=complex)

The rest of the model ingredients can likewise be written:

.. code-block:: python 

    @ingredients.vectorize_ingredient
    def h_q(self, constants, parameters, **kwargs):
        E = constants.E
        V = constants.V
        return np.array([[E, V], [V, -E]], dtype=complex)

    @ingredients.vectorize_ingredient
    def h_qc(self, constants, parameters, **kwargs):
        z = kwargs['z']
        g = constants.spin_boson_coupling
        m = constants.classical_coordinate_mass
        h = constants.classical_coordinate_weight
        h_qc = np.zeros((2, 2), dtype=complex)
        h_qc[0, 0] = np.sum((g * np.sqrt(1 / (2 * m * h))) * (z + np.conj(z)))
        h_qc[1, 1] = -h_qc[0, 0]
        return h_qc

    @ingredients.vectorize_ingredient
    def h_c(self, constants, parameters, **kwargs):
        z = kwargs['z']
        w = constants.harmonic_oscillator_frequency
        return np.sum(w * np.conj(z) * z)


Now you have a working model class which you can instantiate and use following the instructions in the Quickstart Guide! 

.. note::
    
    Please be aware that the performance is going to be significantly worse than what can be achieved by implementing the 
    upgrades below. 

Prior to implementing upgrades, a minimal model will be subject to a number of defaults which are discussed in the `.. default_behavior`


The full minimal model looks like this:

.. code-block:: python

    class MinimalSpinBoson(Model):
        def __init__(self, constants=None):
            if constants is None:
                constants = {}
            self.default_constants = {
                'temp': 1, 'V': 0.5, 'E': 0.5, 'A': 100, 'W': 0.1,
                'l_reorg': 0.02 / 4, 'boson_mass': 1
            }
            super().__init__(self.default_constants, constants)

        def initialize_constants_model(self):
            num_bosons = self.constants.get("A", self.default_constants.get("A"))
            char_freq = self.constants.get("W", self.default_constants.get("W"))
            boson_mass = self.constants.get(
                "boson_mass", self.default_constants.get("boson_mass")
            )
            self.constants.w = char_freq * np.tan(
                ((np.arange(num_bosons) + 1) - 0.5) * np.pi / (2 * num_bosons)
            )
            # The following constants are required by QC Lab
            self.constants.num_classical_coordinates = num_bosons
            self.constants.num_quantum_states = 2
            self.constants.classical_coordinate_weight = self.constants.w
            self.constants.classical_coordinate_mass = boson_mass * np.ones(num_bosons)

        def initialize_constants_h_c(self):
            """
            Initialize the constants for the classical Hamiltonian.
            """
            w = self.constants.get("w", self.default_constants.get("w"))
            self.constants.harmonic_oscillator_frequency = w


        def initialize_constants_h_qc(self):
            """
            Initialize the constants for the quantum-classical coupling Hamiltonian.
            """
            num_bosons = self.constants.get("A", self.default_constants.get("A"))
            w = self.constants.get("w", self.default_constants.get("w"))
            l_reorg = self.constants.get("l_reorg", self.default_constants.get("l_reorg"))
            self.constants.spin_boson_coupling = w * np.sqrt(2 * l_reorg / num_bosons)

        def initialize_constants_h_q(self):
            """
            Initialize the constants for the quantum Hamiltonian. None are required in this case.
            """

        initialization_functions = [
            initialize_constants_model,
            initialize_constants_h_c,
            initialize_constants_h_qc,
            initialize_constants_h_q,
        ]

        @ingredients.vectorize_ingredient
        def h_q(self, constants, parameters, **kwargs):
            E = constants.E
            V = constants.V
            return np.array([[E, V], [V, -E]], dtype=complex)

        @ingredients.vectorize_ingredient
        def h_qc(self, constants, parameters, **kwargs):
            z = kwargs['z']
            g = constants.spin_boson_coupling
            m = constants.classical_coordinate_mass
            h = constants.classical_coordinate_weight
            h_qc = np.zeros((2, 2), dtype=complex)
            h_qc[0, 0] = np.sum((g * np.sqrt(1 / (2 * m * h))) * (z + np.conj(z)))
            h_qc[1, 1] = -h_qc[0, 0]
            return h_qc

        @ingredients.vectorize_ingredient
        def h_c(self, constants, parameters, **kwargs):
            z = kwargs['z']
            w = constants.harmonic_oscillator_frequency
            return np.sum(w * np.conj(z) * z)


Upgrading the Model Class
-------------------------


Vectorized Ingredients
~~~~~~~~~~~~~~~~~~~~~~~

The first upgrade we recommend is to include vectorized ingredients. Vectorized ingredients are ingredients that can be computed for a batch of
trajectories simultaneously. If implemented making use of broadcasting and vectorized numpy functions, vectorized ingredients can greatly improve
the performance of QC Lab.

Here we show vectorized versions of the ingredients used in the minimal model. Since they are vectorized, they do not need to use the `@ingredients.vectorize_ingredient`
decorator. An important feature of vectorized ingredients is how they determine the number of trajectories being calculated. In ingredients that depend on the classical coordinate
this is done by comparing the shape of the first index of the classical coordinate to the provided `batch_size` parameter. In others where the classical coordinate is not 
provided, the `batch_size` is compared to the number of seeds in the simulation.

.. code-block:: python

    def h_q(self, constants, parameters, **kwargs):
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = len(parameters.seed)
        E = constants.E
        V = constants.V
        h_q = np.zeros((batch_size, 2, 2), dtype=complex)
        h_q[:, 0, 0] = E
        h_q[:, 1, 1] = -E
        h_q[:, 0, 1] = V
        h_q[:, 1, 0] = V
        return h_q


    def h_qc(self, constants, parameters, **kwargs):
        z = kwargs.get("z")
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
            assert len(z) == batch_size
        else:
            batch_size = len(z)
        g = constants.spin_boson_coupling
        m = constants.classical_coordinate_mass
        h = constants.classical_coordinate_weight
        h_qc = np.zeros((batch_size, 2, 2), dtype=complex)
        h_qc[:, 0, 0] = np.sum(
            g * np.sqrt(1 / (2 * m * h))[np.newaxis, :] * (z + np.conj(z)), axis=-1
        )
        h_qc[:, 1, 1] = -h_qc[:, 0, 0]
        return h_qc

    def h_c(self, constants, parameters, **kwargs):
        z = kwargs.get("z")
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
            assert len(z) == batch_size
        else:
            batch_size = len(z)

        h = constants.classical_coordinate_weight[np.newaxis, :]
        w = constants.harmonic_oscillator_frequency[np.newaxis, :]
        m = constants.classical_coordinate_mass[np.newaxis, :]
        q = np.sqrt(2 / (m * h)) * np.real(z)
        p = np.sqrt(2 * m * h) * np.imag(z)
        h_c = np.sum((1 / 2) * (((p**2) / m) + m * (w**2) * (q**2)), axis=-1)
        return h_c



Analytic Gradients
~~~~~~~~~~~~~~~~~~


By Default, QC Lab calculates gradients numerically with finite differences. This can in many cases be avoided by providing ingredients
that return the gradients based on analytic formulas. The gradient of the classical Hamiltonian in the spin-boson model is given by 

.. math::

    \frac{\partial H_{\mathrm{c}}}{\partial z^{*}_{\alpha}} = \frac{1}{2}\left(\frac{\omega^{2}_{\alpha}}{h_{\alpha}} + h_{\alpha}\right)z_{\alpha} + 
            \frac{1}{2}\left(\frac{\omega^{2}_{\alpha}}{h_{\alpha}} - h_{\alpha}\right)z^{*}_{\alpha}

which can be implemented in a vectorized fashion as:

.. code-block:: python

    def dh_c_dzc(self, constants, parameters, **kwargs):
        z = kwargs.get("z")
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
            assert len(z) == batch_size
        else:
            batch_size = len(z)
        h = constants.classical_coordinate_weight
        w = constants.harmonic_oscillator_frequency
        a = (1 / 4) * (
            ((w**2) / h) - h
        )
        b = (1 / 4) * (
            ((w**2) / h) + h
        )
        dh_c_dzc = 2 * b[..., :] * z + 2 *a[..., :] * np.conj(z)
        return dh_c_dzc

Likewise we can construct an ingredient to generate the gradient of the quantum-classical Hamiltonian with respect to the conjugate z coordinate.
In many cases this requires the calculation of a sparse tensor and so QC Lab assumes that it is in terms of indices, nonzero elements, and a shape.

.. math::

    \left\langle i\left\vert \frac{\partial \hat{H}_{\mathrm{q-c}}}{\partial z^{*}_{\alpha}}\right\vert j \right\rangle = (-1)^{i}\frac{g_{\alpha}}{\sqrt{2mh_{\alpha}}}\delta_{ij}


Which can be implemented as:

.. code-block:: python

    def dh_qc_dzc(self, constants, parameters, **kwargs):
        z = kwargs.get("z")
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
            assert len(z) == batch_size
        else:
            batch_size = len(z)

        recalculate = False
        if self.dh_qc_dzc_shape is not None:
            if self.dh_qc_dzc_shape[0] != batch_size:
                recalculate = True

        if (
            self.dh_qc_dzc_inds is None
            or self.dh_qc_dzc_mels is None
            or self.dh_qc_dzc_shape is None
            or recalculate
        ):

            m = constants.classical_coordinate_mass
            g = constants.spin_boson_coupling
            h = constants.classical_coordinate_weight
            dh_qc_dzc = np.zeros((batch_size, constants.A, 2, 2), dtype=complex)
            dh_qc_dzc[:, :, 0, 0] = (g * np.sqrt(1 / (2 * m * h)))[..., :]
            dh_qc_dzc[:, :, 1, 1] = -dh_qc_dzc[..., :, 0, 0]
            inds = np.where(dh_qc_dzc != 0)
            mels = dh_qc_dzc[inds]
            shape = np.shape(dh_qc_dzc)
            self.dh_qc_dzc_inds = inds
            self.dh_qc_dzc_mels = dh_qc_dzc[inds]
            self.dh_qc_dzc_shape = shape
        else:
            inds = self.dh_qc_dzc_inds
            mels = self.dh_qc_dzc_mels
            shape = self.dh_qc_dzc_shape
        return inds, mels, shape

An important feature of the above implementation is that it checks if the gradient has already been calculated, this is convenient because the gradient is a constant
and so does not need to be recalculated every time the ingredient is called. As a consequence, however, we need to initialize the gradient to None in the model class.

.. code-block:: python

    def __init__(self, constants=None):
        # Include initialization of the model as done above.
        self.dh_qc_dzc_inds = None
        self.dh_qc_dzc_mels = None
        self.dh_qc_dzc_shape = None



Note that a flag can be included to prevent the RK4 solver in QC Lab from recalculating the quantum-classical forces (ie the expectation value of `dh_qc_dzc`):
`sim.model.linear_h_qc = True`.



Classical Initialization
~~~~~~~~~~~~~~~~~~~~~~~~


By default QC Lab assumes that a model's initial z coordinate is sampled from a Boltzmann distribution at temperature "temp" and attempts to sample a 
Boltzmann distribution given the classical Hamiltonian. This is in practice making a number of assumptions, notably that all the z coordinates are uncoupled from 
one another in the classical Hamiltonian. 


This is accomplished by defining an ingredient called `init_classical` which has the following form:

.. code-block:: python

    def init_classical(model, constants, parameters, **kwargs):
        del model, parameters
        seed = kwargs.get("seed", None)
        kbt = constants.temp
        h = constants.classical_coordinate_weight
        w = constants.harmonic_oscillator_frequency
        m = constants.classical_coordinate_mass
        out = np.zeros((len(seed), constants.num_classical_coordinates), dtype=complex)
        for s, seed_value in enumerate(seed):
            np.random.seed(seed_value)
            # Calculate the standard deviations for q and p.
            std_q = np.sqrt(kbt / (m * (w**2)))
            std_p = np.sqrt(m * kbt)
            # Generate random q and p values.
            q = np.random.normal(
                loc=0, scale=std_q, size=constants.num_classical_coordinates
            )
            p = np.random.normal(
                loc=0, scale=std_p, size=constants.num_classical_coordinates
            )
            # Calculate the complex classical coordinate.
            z = np.sqrt(h * m / 2) * (q + 1.0j * (p / (h * m)))
            out[s] = z
        return out



.. note::
    The upgraded model still relies on numerical hopping for FSSH simulations. Below we will see how to use the analytic hopping ingredient for the harmonic oscillator.

The full code for the upgraded model is:

.. code-block:: python

    class UpgradedSpinBoson(Model):
        def __init__(self, constants=None):
            if constants is None:
                constants = {}
            self.default_constants = {
                'temp': 1, 'V': 0.5, 'E': 0.5, 'A': 100, 'W': 0.1,
                'l_reorg': 0.02 / 4, 'boson_mass': 1
            }
            self.dh_qc_dzc_inds = None
            self.dh_qc_dzc_mels = None
            self.dh_qc_dzc_shape = None
            self.linear_h_qc = True
            super().__init__(self.default_constants, constants)

        def initialize_constants_model(self):
            num_bosons = self.constants.get("A", self.default_constants.get("A"))
            char_freq = self.constants.get("W", self.default_constants.get("W"))
            boson_mass = self.constants.get(
                "boson_mass", self.default_constants.get("boson_mass")
            )
            self.constants.w = char_freq * np.tan(
                ((np.arange(num_bosons) + 1) - 0.5) * np.pi / (2 * num_bosons)
            )
            # The following constants are required by QC Lab
            self.constants.num_classical_coordinates = num_bosons
            self.constants.num_quantum_states = 2
            self.constants.classical_coordinate_weight = self.constants.w
            self.constants.classical_coordinate_mass = boson_mass * np.ones(num_bosons)

        def initialize_constants_h_c(self):
            """
            Initialize the constants for the classical Hamiltonian.
            """
            w = self.constants.get("w", self.default_constants.get("w"))
            self.constants.harmonic_oscillator_frequency = w


        def initialize_constants_h_qc(self):
            """
            Initialize the constants for the quantum-classical coupling Hamiltonian.
            """
            num_bosons = self.constants.get("A", self.default_constants.get("A"))
            w = self.constants.get("w", self.default_constants.get("w"))
            l_reorg = self.constants.get("l_reorg", self.default_constants.get("l_reorg"))
            self.constants.g = w * np.sqrt(2 * l_reorg / num_bosons)

        def initialize_constants_h_q(self):
            """
            Initialize the constants for the quantum Hamiltonian. None are required in this case.
            """

        initialization_functions = [
            initialize_constants_model,
            initialize_constants_h_c,
            initialize_constants_h_qc,
            initialize_constants_h_q,
        ]

        def h_q(self, constants, parameters, **kwargs):
            if kwargs.get("batch_size") is not None:
                batch_size = kwargs.get("batch_size")
            else:
                batch_size = len(parameters.seed)
            E = constants.E
            V = constants.V
            h_q = np.zeros((batch_size, 2, 2), dtype=complex)
            h_q[:, 0, 0] = E
            h_q[:, 1, 1] = -E
            h_q[:, 0, 1] = V
            h_q[:, 1, 0] = V
            return h_q


        def h_qc(self, constants, parameters, **kwargs):
            z = kwargs.get("z")
            if kwargs.get("batch_size") is not None:
                batch_size = kwargs.get("batch_size")
                assert len(z) == batch_size
            else:
                batch_size = len(z)
            g = constants.spin_boson_coupling
            m = constants.classical_coordinate_mass
            h = constants.classical_coordinate_weight
            h_qc = np.zeros((batch_size, 2, 2), dtype=complex)
            h_qc[:, 0, 0] = np.sum(
                g * np.sqrt(1 / (2 * m * h))[np.newaxis, :] * (z + np.conj(z)), axis=-1
            )
            h_qc[:, 1, 1] = -h_qc[:, 0, 0]
            return h_qc

        def h_c(self, constants, parameters, **kwargs):
            z = kwargs.get("z")
            if kwargs.get("batch_size") is not None:
                batch_size = kwargs.get("batch_size")
                assert len(z) == batch_size
            else:
                batch_size = len(z)

            h = constants.classical_coordinate_weight[np.newaxis, :]
            w = constants.harmonic_oscillator_frequency[np.newaxis, :]
            m = constants.classical_coordinate_mass[np.newaxis, :]
            q = np.sqrt(2 / (m * h)) * np.real(z)
            p = np.sqrt(2 * m * h) * np.imag(z)
            h_c = np.sum((1 / 2) * (((p**2) / m) + m * (w**2) * (q**2)), axis=-1)
            return h_c
        
        def dh_c_dzc(self, constants, parameters, **kwargs):
            z = kwargs.get("z")
            if kwargs.get("batch_size") is not None:
                batch_size = kwargs.get("batch_size")
                assert len(z) == batch_size
            else:
                batch_size = len(z)
            h = constants.classical_coordinate_weight
            w = constants.harmonic_oscillator_frequency
            a = (1 / 4) * (
                ((w**2) / h) - h
            )
            b = (1 / 4) * (
                ((w**2) / h) + h
            )
            dh_c_dzc = 2 * b[..., :] * z + 2 *a[..., :] * np.conj(z)
            return dh_c_dzc
        
        def dh_qc_dzc(self, constants, parameters, **kwargs):
            z = kwargs.get("z")
            if kwargs.get("batch_size") is not None:
                batch_size = kwargs.get("batch_size")
                assert len(z) == batch_size
            else:
                batch_size = len(z)

            recalculate = False
            if self.dh_qc_dzc_shape is not None:
                if self.dh_qc_dzc_shape[0] != batch_size:
                    recalculate = True

            if (
                self.dh_qc_dzc_inds is None
                or self.dh_qc_dzc_mels is None
                or self.dh_qc_dzc_shape is None
                or recalculate
            ):

                m = constants.classical_coordinate_mass
                g = constants.spin_boson_coupling
                h = constants.classical_coordinate_weight
                dh_qc_dzc = np.zeros((batch_size, constants.A, 2, 2), dtype=complex)
                dh_qc_dzc[:, :, 0, 0] = (g * np.sqrt(1 / (2 * m * h)))[..., :]
                dh_qc_dzc[:, :, 1, 1] = -dh_qc_dzc[..., :, 0, 0]
                inds = np.where(dh_qc_dzc != 0)
                mels = dh_qc_dzc[inds]
                shape = np.shape(dh_qc_dzc)
                self.dh_qc_dzc_inds = inds
                self.dh_qc_dzc_mels = dh_qc_dzc[inds]
                self.dh_qc_dzc_shape = shape
            else:
                inds = self.dh_qc_dzc_inds
                mels = self.dh_qc_dzc_mels
                shape = self.dh_qc_dzc_shape
            return inds, mels, shape
        
        def init_classical(model, constants, parameters, **kwargs):
            del model, parameters
            seed = kwargs.get("seed", None)
            kbt = constants.temp
            h = constants.classical_coordinate_weight
            w = constants.harmonic_oscillator_frequency
            m = constants.classical_coordinate_mass
            out = np.zeros((len(seed), constants.num_classical_coordinates), dtype=complex)
            for s, seed_value in enumerate(seed):
                np.random.seed(seed_value)
                # Calculate the standard deviations for q and p.
                std_q = np.sqrt(kbt / (m * (w**2)))
                std_p = np.sqrt(m * kbt)
                # Generate random q and p values.
                q = np.random.normal(
                    loc=0, scale=std_q, size=constants.num_classical_coordinates
                )
                p = np.random.normal(
                    loc=0, scale=std_p, size=constants.num_classical_coordinates
                )
                # Calculate the complex classical coordinate.
                z = np.sqrt(h * m / 2) * (q + 1.0j * (p / (h * m)))
                out[s] = z
            return out


Using Built-in Ingredients
~~~~~~~~~~~~~~~~~~~~~~~~~~~

QC Lab comes with a number of built-in ingredients that can be used to construct a model rather than writing ingredients from scratch like above. These ingredients can be imported 
from `qc_lab.ingredients` and are documented in the :ref:`ingredients` section.

For the present example, we can avoid writing our own optimized ingredients and simply add the available built-in ingredients to the model. 

First let's load the quantum Hamiltonian as the built-in two-level system Hamiltonian:

.. code-block:: python

    import qc_lab.ingredients as ingredients
    model = MinimalSpinBoson 

    model.h_q = ingredients.two_level_system_h_q

We will also need a new initialization function to interface with the constants used by the built-in ingredient:

.. code-block:: python

    def initialize_constants_h_q(model):
        """
        Initialize the constants for the quantum Hamiltonian.
        """
        model.constants.two_level_system_a = model.constants.get(
            "E", model.default_constants.get("E")
        )
        model.constants.two_level_system_b = -model.constants.get(
            "E", model.default_constants.get("E")
        )
        model.constants.two_level_system_c = model.constants.get(
            "V", model.default_constants.get("V")
        )
        model.constants.two_level_system_d = 0

Let's then add the ingredient and initialization function to the model class

.. code-block:: python

    model.h_q = ingredients.two_level_system_h_q
    model.initialize_constants_h_q = initialize_constants_h_q
    # also update the list of initialization functions
    model.initialization_functions[3] = initialize_constants_h_q


Next we can load the classical Hamiltonian as the built-in harmonic oscillator Hamiltonian and update the initialization function like before:

.. code-block:: python


    def initialize_constants_h_c(model):
        """
        Initialize the constants for the classical Hamiltonian.
        """
        w = model.constants.get("w", model.default_constants.get("w"))
        model.constants.harmonic_oscillator_frequency = w

    model.h_c = ingredients.harmonic_oscillator_h_c
    model.initialize_constants_h_c = initialize_constants_h_c
    model.initialization_functions[1] = initialize_constants_h_c
    
We can also load analytic gradients for the classical Hamiltonian (which relies
on the same constants has the classical Hamiltonian).

.. code-block:: python

    model.dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc

Next we can load the quantum-classical Hamiltonian and its gradient, no additional initialization is needed.

.. code-block:: python

    model.h_qc = ingredients.spin_boson_h_qc
    model.dh_qc_dzc = ingredients.spin_boson_dh_qc_dzc
    model.dh_qc_dzc_inds = None
    model.dh_qc_dzc_mels = None
    model.dh_qc_dzc_shape = None


Then we can load in the built-in classical initialization ingredient which samples the Boltzmann distribution for the harmonic oscillator Hamiltonian.

.. code-block:: python

    model.init_classical = ingredients.harmonic_oscillator_init_classical

Next, we can load an ingredient that executes the hopping procedure of the FSSH algorithm according to a harmonic oscillator. This will improve the 
performance of the FSSH algorithm. 

.. code-block:: python

    model.hop_function = ingredients.harmonic_oscillator_boltzmann_init_classical

Lastly, we can add a flag to the model class that enables the RK4 solver in QC Lab to avoid recalculating gradients of the quantum-classical Hamiltonian 
(which is a constant if the quantum-classical Hamiltonian is linear in z).

.. code-block:: python

    model.linear_h_qc = True


The resulting model class is now fully upgraded and can be used to simulate the spin-boson model with significantly improved performance.

The full code for upgrading the `MinimalSpinBoson` using built-in ingredients is:

.. code-block:: python 

    model = MinimalSpinBoson

    model.h_q = ingredients.two_level_system_h_q

    def initialize_constants_h_q(model):
        """
        Initialize the constants for the quantum Hamiltonian.
        """
        model.constants.two_level_system_a = model.constants.get(
            "E", model.default_constants.get("E")
        )
        model.constants.two_level_system_b = -model.constants.get(
            "E", model.default_constants.get("E")
        )
        model.constants.two_level_system_c = model.constants.get(
            "V", model.default_constants.get("V")
        )
        model.constants.two_level_system_d = 0

    model.initialize_constants_h_q = initialize_constants_h_q
    # also update the list of initialization functions
    model.initialization_functions[3] = initialize_constants_h_q

    def initialize_constants_h_c(model):
        """
        Initialize the constants for the classical Hamiltonian.
        """
        w = model.constants.get("w", model.default_constants.get("w"))
        model.constants.harmonic_oscillator_frequency = w

    model.h_c = ingredients.harmonic_oscillator_h_c
    model.initialize_constants_h_c = initialize_constants_h_c
    model.initialization_functions[1] = initialize_constants_h_c

    model.dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc

    model.h_qc = ingredients.spin_boson_h_qc
    model.dh_qc_dzc = ingredients.spin_boson_dh_qc_dzc
    model.dh_qc_dzc_inds = None
    model.dh_qc_dzc_mels = None
    model.dh_qc_dzc_shape = None

    model.init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical

    model.hop_function = ingredients.harmonic_oscillator_hop_function

    model.linear_h_qc = True
