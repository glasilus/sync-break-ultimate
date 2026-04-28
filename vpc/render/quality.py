"""Quality presets — convenience layer over manual CRF/preset/tune.

These are *suggestions*. The GUI keeps the manual CRF / ffmpeg-preset /
tune controls fully editable; picking a preset just fills those fields in
one click. Selecting any preset value writes the three keys; touching any
of the three by hand flips the preset selector to 'Custom' so the user
sees that the dropdown no longer matches what's set.

`tune` is x264/x265-only. We restrict the menu to values supported by
both encoders (`film`, `grain`, `animation`, `stillimage`) so a preset
swap can't break a render when the user later switches codec. `none`
means: don't pass `-tune` at all.
"""
from __future__ import annotations

from typing import Optional


CUSTOM = 'Custom'

QUALITY_PRESETS = {
    # Marker — picking 'Custom' does NOT mutate any field. It exists so
    # the dropdown has a stable label when manual values diverge from any
    # preset.
    CUSTOM:    None,
    # Heavy: archival-grade, generous bitrate, grain-tuned for
    # noise/datamosh material that "smooth" tunes would smear.
    'Archive': {'crf': 17, 'export_preset': 'slow',   'tune': 'grain'},
    # Default. Visually lossless for most material at sensible speed.
    'High':    {'crf': 20, 'export_preset': 'medium', 'tune': 'none'},
    # Web-safe size, fast encode.
    'Web':     {'crf': 23, 'export_preset': 'fast',   'tune': 'none'},
    # Smallest reasonable file, still watchable.
    'Compact': {'crf': 26, 'export_preset': 'fast',   'tune': 'none'},
}

# Keys a preset writes — used by the GUI's 'is current state == preset?' check.
PRESET_KEYS = ('crf', 'export_preset', 'tune')

# tune values exposed in the GUI dropdown.
TUNE_VALUES = ('none', 'film', 'grain', 'animation', 'stillimage')


def preset_names() -> list[str]:
    """Order shown in the dropdown."""
    return [CUSTOM, 'Archive', 'High', 'Web', 'Compact']


def matches(name: str, *, crf: int, export_preset: str,
            tune: str) -> bool:
    """True if (crf, preset, tune) exactly equal the named preset.
    `Custom` never matches — it's the fallback label."""
    spec = QUALITY_PRESETS.get(name)
    if not spec:
        return False
    return (int(crf) == int(spec['crf'])
            and str(export_preset) == str(spec['export_preset'])
            and str(tune or 'none') == str(spec['tune']))


def detect_preset(*, crf: int, export_preset: str, tune: str) -> str:
    """Return the preset name whose (crf, preset, tune) match, or 'Custom'."""
    for name in QUALITY_PRESETS:
        if name == CUSTOM:
            continue
        if matches(name, crf=crf, export_preset=export_preset, tune=tune):
            return name
    return CUSTOM


def tune_supported(vcodec: str) -> bool:
    """`-tune` is only meaningful for libx264/libx265."""
    return vcodec in ('libx264', 'libx265')


def normalize_tune(value: Optional[str]) -> str:
    """Coerce arbitrary cfg input to a legal TUNE_VALUES entry."""
    if value is None:
        return 'none'
    v = str(value).strip().lower()
    return v if v in TUNE_VALUES else 'none'
