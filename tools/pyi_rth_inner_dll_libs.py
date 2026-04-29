"""PyInstaller runtime hook: register inner `.libs/` directories on Windows.

Wheel layout conventions
------------------------
Python wheels for the scientific stack ship native DLL dependencies
in two different ways:

  (a) **delvewheel** convention (numpy 1.26+, scipy, av, ...):
      `<pkg>.libs/` lives as a *sibling* of the package directory.
      PyInstaller has a purpose-built utility for this:
      `PyInstaller.utils.hooks.collect_delvewheel_libs_directory()`,
      and runtime DLL search is wired up by the bootloader.

  (b) **auditwheel-cibuildwheel** convention (numpy 1.24, sklearn,
      Pillow legacy, ...): `<pkg>/.libs/` lives *inside* the package
      directory. `--collect-all <pkg>` already bundles those DLLs —
      they ride along with the package — but Windows' default DLL
      search path does not recurse into subdirectories. So the .pyd
      that depends on them (e.g. `numpy/core/_multiarray_umath.pyd`
      → libopenblas) cannot find them at runtime, and import dies
      with the generic "DLL load failed" message.

This hook handles convention (b) at runtime: it walks `_MEIPASS`,
finds every `<pkg>/.libs/` subdirectory, and registers each via
`os.add_dll_directory()` (Python 3.8+ on Windows). For convention
(a) the per-package hook files in `tools/pyi_hooks/` (hook-scipy,
hook-av, ...) handle bundling, and PyInstaller's bootloader
already registers `_MEIPASS` on the search path so the bundled
sibling `<pkg>.libs/` directories are reachable from there.

Why this is the right tool, not a workaround
--------------------------------------------
PyInstaller's `collect_delvewheel_libs_directory` API only handles
the sibling layout (a). There is no equivalent built-in for the
inner-`.libs/` layout (b) because PyInstaller's static collection
already places those DLLs in the right relative location — what's
missing is the *runtime* search-path registration. `--runtime-hook`
is PyInstaller's documented mechanism for exactly that:
"Custom runtime hook code that runs before the bundled python
imports, intended for situations like setting environment
variables or extending the DLL search path."

Refs:
- https://pyinstaller.org/en/stable/spec-files.html#using-the-spec-file-built-in-runtime-hooks
- https://pyinstaller.org/en/stable/hooks.html#PyInstaller.utils.hooks.collect_delvewheel_libs_directory
"""
import os
import sys

# add_dll_directory is Windows-only and Python 3.8+. On macOS/Linux
# this hook is a no-op — the dynamic linker uses RPATH/RUNPATH and
# PyInstaller already sets those correctly.
if hasattr(os, "add_dll_directory") and getattr(sys, "_MEIPASS", None):
    _base = sys._MEIPASS
    try:
        for _entry in os.listdir(_base):
            _inner = os.path.join(_base, _entry, ".libs")
            if os.path.isdir(_inner):
                try:
                    os.add_dll_directory(_inner)
                except OSError:
                    # Path was deleted between listdir and the call,
                    # or the OS rejected the path. Best-effort hook —
                    # skip and let the import fail loudly downstream.
                    pass
    except OSError:
        pass
