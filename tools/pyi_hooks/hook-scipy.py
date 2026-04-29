"""Supplemental PyInstaller hook for scipy.

Same rationale as hook-numpy.py: scipy's Windows wheel uses
delvewheel-style `scipy.libs/` for native deps (OpenBLAS,
libgfortran). `--collect-all scipy` misses it because it's a
sibling of the scipy/ package, not part of it. Without these
DLLs scipy.signal / scipy.fft / etc. fail to import in `--onedir`
bundles with the same opaque "DLL load failed" message numpy
shows.
"""
from PyInstaller.utils.hooks import collect_delvewheel_libs_directory

datas, binaries = collect_delvewheel_libs_directory("scipy")
