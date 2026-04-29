"""Supplemental PyInstaller hook for PyAV.

PyAV's Windows wheel ships ~30 native DLLs (FFmpeg's libavcodec /
libavformat / etc.) in a sibling `av.libs/` directory via delvewheel.
Same problem as scipy/numpy in `--onedir`: `--collect-all av` walks
the package itself but cannot reach the sibling `.libs/` dir.
Without these, importing av.* raises a generic "DLL load failed".

PyAV is pulled transitively by some video-decode paths (scenedetect,
imageio); we don't import it directly, but if anything in the import
chain does, this hook ensures the DLLs travel with the bundle.
"""
from PyInstaller.utils.hooks import collect_delvewheel_libs_directory

datas, binaries = collect_delvewheel_libs_directory("av")
