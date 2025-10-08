import os

from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop


DEPENDENCIES = [
    "numpy",
    "tqdm",
    "h5py",
    "matplotlib",
    "numba",
]


DISABLE_NUMBA = os.environ.get("QC_LAB_DISABLE_NUMBA", "0") == "1"
DISABLE_H5PY = os.environ.get("QC_LAB_DISABLE_H5PY", "0") == "1"

if DISABLE_NUMBA:
    print("Numba is disabled. QC Lab will run without Numba optimizations.")
    DEPENDENCIES.remove("numba")
if DISABLE_H5PY:
    print("H5PY is disabled. QC Lab will run without HDF5 support.")
    DEPENDENCIES.remove("h5py")


def _write_config(target: str) -> None:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as f:
        f.write(
            "\n".join(
                [
                    "import os as _os",
                    "",
                    "def _env_flag(name: str, default: bool) -> bool:",
                    "    value = _os.environ.get(name)",
                    "    if value is None:",
                    "        return default",
                    '    return value not in {"0", "false", "False"}',
                    "",
                    f'DISABLE_NUMBA = _env_flag("QC_LAB_DISABLE_NUMBA", {DISABLE_NUMBA})',
                    f'DISABLE_H5PY = _env_flag("QC_LAB_DISABLE_H5PY", {DISABLE_H5PY})',
                    "",
                ]
            )
        )


class develop(_develop):
    def run(self) -> None:
        self.distribution.install_requires = DEPENDENCIES
        target = os.path.join(self.install_lib, "qc_lab", "_config.py")
        _write_config(target)
        super().run()


class build_py(_build_py):
    def run(self) -> None:
        self.distribution.install_requires = DEPENDENCIES
        target = os.path.join(self.build_lib, "qc_lab", "_config.py")
        _write_config(target)
        super().run()
