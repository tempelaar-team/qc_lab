import os as _os


def _env_flag(name: str, default: bool) -> bool:
    value = _os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False"}


DISABLE_NUMBA = _env_flag("QC_LAB_DISABLE_NUMBA", False)
DISABLE_H5PY = _env_flag("QC_LAB_DISABLE_H5PY", False)
