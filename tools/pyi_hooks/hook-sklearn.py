"""Supplemental PyInstaller hook for scikit-learn (inner-.libs layout).

Same problem and same fix as hook-numpy.py: sklearn's Windows wheel
ships its native DLLs in `<site-packages>/sklearn/.libs/` (inner
auditwheel convention), and PyInstaller's standard collectors skip
the dot-prefixed directory. We pick those DLLs up explicitly and
emit (src, dest) entries; the runtime hook registers the destination
with `os.add_dll_directory()` so the .pyd extensions can find them.

scikit-learn is not a direct dependency of this project but is
pulled transitively via librosa, so the failure mode is the same
opaque "DLL load failed" message during `import librosa`.
"""
import os

try:
    import sklearn as _sklearn
except ImportError:
    binaries = []
else:
    _SK_ROOT = os.path.dirname(_sklearn.__file__)
    _INNER_LIBS = os.path.join(_SK_ROOT, ".libs")

    binaries = []
    if os.path.isdir(_INNER_LIBS):
        for _fname in os.listdir(_INNER_LIBS):
            _src = os.path.join(_INNER_LIBS, _fname)
            if os.path.isfile(_src):
                binaries.append((_src, os.path.join("sklearn", ".libs")))
