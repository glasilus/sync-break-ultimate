"""Supplemental PyInstaller hook for numpy (inner-.libs layout).

Why this exists
---------------
NumPy 1.24 on Windows uses the **auditwheel/cibuildwheel inner-.libs
convention**: native DLLs (libopenblas...) live at
`<site-packages>/numpy/.libs/`. PyInstaller's standard collection
machinery (`--collect-all numpy`, `collect_dynamic_libs("numpy")`)
does NOT copy this directory into the bundle. The directory name
starts with a dot and is skipped by the underlying file-walk used
to gather binary files — empirically verified by inspecting the
bundle layout via a runtime diagnostic hook.

Without `numpy/.libs/` in the bundle, `_distributor_init.py` runs
during `import numpy` and tries to register the OpenBLAS path via
`os.add_dll_directory()`, but the directory does not exist. Then
`_multiarray_umath.cp310-win_amd64.pyd` fails to load OpenBLAS and
import dies with the generic "DLL load failed: module not found".

Fix
---
Enumerate `<numpy>/.libs/` at build time and append each file to
`binaries` with the destination `numpy/.libs` (relative to MEIPASS).
PyInstaller copies them into `_internal/numpy/.libs/` in the bundle.
At runtime, our `pyi_rth_inner_dll_libs.py` registers that path with
`os.add_dll_directory()` so Windows can resolve OpenBLAS from inside
`_multiarray_umath`.

This hook supplements (does not replace) PyInstaller's built-in
numpy hook — both run and their `binaries` lists are merged.

For numpy 1.26+ (delvewheel sibling `numpy.libs/`) this hook is a
no-op: the inner directory does not exist there. The built-in hook
or `collect_delvewheel_libs_directory("numpy")` handles that case.
"""
import os

import numpy as _numpy

_NUMPY_ROOT = os.path.dirname(_numpy.__file__)
_INNER_LIBS = os.path.join(_NUMPY_ROOT, ".libs")

binaries = []
if os.path.isdir(_INNER_LIBS):
    for _fname in os.listdir(_INNER_LIBS):
        _src = os.path.join(_INNER_LIBS, _fname)
        if os.path.isfile(_src):
            # (source absolute path, destination dir relative to MEIPASS).
            binaries.append((_src, os.path.join("numpy", ".libs")))
