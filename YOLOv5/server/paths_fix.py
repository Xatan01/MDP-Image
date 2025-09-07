# Runs on import: allow loading Linux-saved checkpoints on Windows (Py 3.12/3.13)
import sys, importlib, pathlib

if sys.platform.startswith("win"):
    try:
        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass
    try:
        _local = importlib.import_module("pathlib._local")  # Py 3.12+
        _local.PosixPath = pathlib.WindowsPath
    except Exception:
        pass
