"""Encoder catalogue + capability detection + rate-control mapping.

This module is the single source of truth about which video encoders we
ship, which ones the local ffmpeg build *actually* has, and how the
user-facing Quality controls (CRF / ffmpeg preset / tune) translate into
the encoder-specific flags each family wants.

Why a separate module:
  * `EXPORT_FORMATS` in `sink.py` was hard-coded to the soft-codec set.
    Adding NVENC/QSV/AMF flags inside that dict mixed two concerns
    (container + encoder + rate control).
  * Hardware encoders are *optional* — they're only listed in the GUI if
    the local ffmpeg build was compiled with them AND the runtime
    succeeds (we verify the runtime side in engine.py via a fallback
    on first-write failure, not here).

Design contract:
  * `available_specs()` returns a list of `EncoderSpec` filtered by what
    `ffmpeg -encoders` reports. The result is cached for the process.
  * `build_rate_control_args(spec, crf, preset, tune)` returns a list
    of CLI flags appended after `-vcodec <vcodec> -pix_fmt <pix_fmt>`.
    The function is the only place that knows about per-family quirks.
  * Soft codecs (libx264/libx265/libvpx-vp9/prores_ks) are always
    listed — they're part of the pure-software fallback path.

Per-family rate-control mapping (input is the user's CRF 0-51 + ffmpeg
preset + tune):

    libx264 / libx265
        -preset <p> -crf <n> [-tune <t>]
        Direct passthrough; the canonical interpretation.

    h264_nvenc / hevc_nvenc
        -preset p1..p7 -rc vbr -cq <n> -b:v 0
        NVENC's `-preset` was renamed in recent ffmpeg from the legacy
        slow/medium/fast names to p1..p7. We map x264 names → p-indices.
        `-cq` accepts the same 0-51 scale as CRF; `-b:v 0` keeps it in
        constant-quality mode. `-tune` is silently ignored (NVENC has
        its own `-tune ll/ull/hq` but we don't expose it here).

    h264_qsv / hevc_qsv
        -preset <veryfast..veryslow> -global_quality <n>
        QSV preset names match x264's, so we passthrough. ICQ rate
        control is implicit when -global_quality is set.

    h264_amf / hevc_amf
        -quality <speed|balanced|quality> -rc cqp -qp_i <n> -qp_p <n>
        AMF doesn't have a CRF-equivalent; constant QP is the closest.

    h264_videotoolbox / hevc_videotoolbox
        -q:v <map(crf)>
        VideoToolbox uses a 1-100 quality scale (higher = better),
        opposite direction from CRF. We map crf=18 → q=64 etc.

    libvpx-vp9
        -crf <n> -b:v 0 -deadline good -cpu-used 4
        Same CRF scale, different rate-control plumbing.

    prores_ks
        -profile:v 3
        ProRes 422 HQ; CRF/preset/tune are all ignored — ProRes is
        intra-frame with fixed quality per profile.

Adding a new encoder:
  1. Append an EncoderSpec to ENCODER_TABLE.
  2. Add a branch in build_rate_control_args() if the family's flags
     don't match an existing one.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

from .sink import ffmpeg_bin


# ----- spec -----

@dataclass(frozen=True)
class EncoderSpec:
    label: str            # user-facing, e.g. 'H.264 NVENC (MP4)'
    container_ext: str    # 'mp4'
    vcodec: str           # 'h264_nvenc'
    acodec: str           # 'aac'
    pix_fmt: str          # 'yuv420p'
    family: str           # 'x264' | 'nvenc_h264' | 'qsv_h264' | ...
    is_hw: bool
    extra_v: List[str] = field(default_factory=list)


# ----- catalogue -----
#
# Order matters: this is the order specs appear in the GUI dropdown, so
# soft-codecs come first (the safe default), HW variants below.

ENCODER_TABLE: List[EncoderSpec] = [
    # ----- software (always available, part of every ffmpeg build) -----
    EncoderSpec('H.264 (MP4)',  'mp4',  'libx264',    'aac',
                'yuv420p', 'x264', False),
    EncoderSpec('H.265 (MP4)',  'mp4',  'libx265',    'aac',
                'yuv420p', 'x265', False, ['-tag:v', 'hvc1']),
    EncoderSpec('H.264 (MKV)',  'mkv',  'libx264',    'aac',
                'yuv420p', 'x264', False),
    EncoderSpec('H.265 (MKV)',  'mkv',  'libx265',    'aac',
                'yuv420p', 'x265', False),
    EncoderSpec('H.264 (MOV)',  'mov',  'libx264',    'aac',
                'yuv420p', 'x264', False),
    EncoderSpec('ProRes (MOV)', 'mov',  'prores_ks',  'pcm_s16le',
                'yuv422p10le', 'prores', False, ['-profile:v', '3']),
    EncoderSpec('VP9 (WebM)',   'webm', 'libvpx-vp9', 'libopus',
                'yuv420p', 'vp9', False, ['-row-mt', '1', '-b:v', '0']),

    # ----- NVIDIA NVENC -----
    EncoderSpec('H.264 NVENC (MP4)', 'mp4', 'h264_nvenc', 'aac',
                'yuv420p', 'nvenc_h264', True),
    EncoderSpec('H.265 NVENC (MP4)', 'mp4', 'hevc_nvenc', 'aac',
                'yuv420p', 'nvenc_hevc', True, ['-tag:v', 'hvc1']),

    # ----- Intel Quick Sync -----
    EncoderSpec('H.264 QSV (MP4)', 'mp4', 'h264_qsv', 'aac',
                'yuv420p', 'qsv_h264', True),
    EncoderSpec('H.265 QSV (MP4)', 'mp4', 'hevc_qsv', 'aac',
                'yuv420p', 'qsv_hevc', True, ['-tag:v', 'hvc1']),

    # ----- AMD AMF -----
    EncoderSpec('H.264 AMF (MP4)', 'mp4', 'h264_amf', 'aac',
                'yuv420p', 'amf_h264', True),
    EncoderSpec('H.265 AMF (MP4)', 'mp4', 'hevc_amf', 'aac',
                'yuv420p', 'amf_hevc', True, ['-tag:v', 'hvc1']),

    # ----- Apple VideoToolbox (macOS) -----
    EncoderSpec('H.264 VideoToolbox (MP4)', 'mp4', 'h264_videotoolbox',
                'aac', 'yuv420p', 'vt_h264', True),
    EncoderSpec('H.265 VideoToolbox (MP4)', 'mp4', 'hevc_videotoolbox',
                'aac', 'yuv420p', 'vt_hevc', True, ['-tag:v', 'hvc1']),
]


# ----- detection -----

_AVAILABLE_VCODECS_CACHE: Optional[set] = None


def _probe_vcodecs() -> set:
    """Run `ffmpeg -encoders` once and return the set of vcodec names
    listed. On failure (ffmpeg missing, parse error) returns the
    soft-codec floor so the GUI is at least functional."""
    soft_floor = {'libx264', 'libx265', 'libvpx-vp9', 'prores_ks'}
    try:
        r = subprocess.run([ffmpeg_bin(), '-hide_banner', '-encoders'],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return soft_floor
        names = set()
        # ffmpeg's table has columns; the encoder name is the second
        # whitespace-separated token after the flag block, e.g.
        #   "V....D h264_nvenc           NVIDIA NVENC ..."
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith('V'):
                names.add(parts[1])
        return names | soft_floor  # always include the floor
    except (FileNotFoundError, subprocess.SubprocessError):
        return soft_floor


def available_specs() -> List[EncoderSpec]:
    """Cached list of EncoderSpec entries this ffmpeg build supports."""
    global _AVAILABLE_VCODECS_CACHE
    if _AVAILABLE_VCODECS_CACHE is None:
        _AVAILABLE_VCODECS_CACHE = _probe_vcodecs()
    avail = _AVAILABLE_VCODECS_CACHE
    return [s for s in ENCODER_TABLE if s.vcodec in avail]


def find_spec(label: str) -> Optional[EncoderSpec]:
    """Lookup by user-facing label, e.g. 'H.264 NVENC (MP4)'."""
    for s in ENCODER_TABLE:
        if s.label == label:
            return s
    return None


def fallback_spec() -> EncoderSpec:
    """The encoder we drop back to when a HW encoder fails at runtime."""
    s = find_spec('H.264 (MP4)')
    assert s is not None  # part of the static table
    return s


# ----- rate-control mapping -----

# x264 preset names → NVENC p-presets. NVENC's p1=fastest..p7=slowest.
_NVENC_PRESET_MAP = {
    'ultrafast': 'p1', 'superfast': 'p2', 'veryfast': 'p2',
    'faster': 'p3',    'fast': 'p3',
    'medium': 'p4',
    'slow': 'p6',      'slower': 'p7',     'veryslow': 'p7',
}

# x264 preset → AMF -quality. AMF only has 3 levels.
_AMF_QUALITY_MAP = {
    'ultrafast': 'speed', 'superfast': 'speed', 'veryfast': 'speed',
    'faster': 'speed',    'fast': 'speed',
    'medium': 'balanced',
    'slow': 'quality',    'slower': 'quality', 'veryslow': 'quality',
}


def _vt_quality_from_crf(crf: int) -> int:
    """Map our 0-51 CRF scale onto VideoToolbox's 1-100 quality scale.
    Higher q = better quality (opposite direction from CRF)."""
    crf = max(0, min(51, int(crf)))
    return max(1, 100 - 2 * crf)


def build_rate_control_args(spec: EncoderSpec, *, crf: int, preset: str,
                            tune: Optional[str]) -> List[str]:
    """Return per-encoder rate-control flags for the given Quality inputs.

    Always emits *something* sensible (constant-quality mode by default)
    so a render never proceeds with random encoder defaults.
    """
    fam = spec.family
    crf = int(crf)
    preset = str(preset or 'medium')
    tune_clean = (str(tune).lower() if tune else 'none')

    if fam in ('x264', 'x265'):
        args = ['-preset', preset, '-crf', str(crf)]
        if tune_clean and tune_clean != 'none':
            args += ['-tune', tune_clean]
        return args

    if fam.startswith('nvenc'):
        p = _NVENC_PRESET_MAP.get(preset, 'p4')
        # -rc vbr + -cq <n> + -b:v 0 = constant-quality VBR. Equivalent
        # to libx264's CRF mode for this encoder family.
        return ['-preset', p, '-rc', 'vbr', '-cq', str(crf), '-b:v', '0']

    if fam.startswith('qsv'):
        # QSV's preset names overlap with x264's; pass through and let
        # ffmpeg reject anything unknown (none of the values we expose
        # are unknown).
        return ['-preset', preset, '-global_quality', str(crf)]

    if fam.startswith('amf'):
        q = _AMF_QUALITY_MAP.get(preset, 'balanced')
        return ['-quality', q, '-rc', 'cqp',
                '-qp_i', str(crf), '-qp_p', str(crf)]

    if fam.startswith('vt'):  # videotoolbox
        return ['-q:v', str(_vt_quality_from_crf(crf))]

    if fam == 'vp9':
        return ['-crf', str(crf), '-deadline', 'good', '-cpu-used', '4']

    if fam == 'prores':
        # Quality is locked by -profile:v in extra_v; nothing else to add.
        return []

    # Future families: emit empty so the render still runs on encoder
    # defaults rather than hard-crashing on a missing branch.
    return []
