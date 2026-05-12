"""Self-contained ffmpeg setup helpers used by BreakcoreEngine.

Extracted from `engine.py` so the orchestrator stays focused on the render
pipeline. Each function here is a pure helper — it does not touch engine
state, only the filesystem + ffmpeg subprocess. The engine wraps each call
in a thin method that supplies its `log` callback.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Callable, Optional

import cv2

from .sink import ffmpeg_bin


LogFn = Callable[[str], None]


def extract_audio_track(video_path: str, log: LogFn) -> Optional[str]:
    """Demux the audio of `video_path` into a temp WAV; return its path.

    Returns None if the video has no audio stream or extraction fails — in
    that case the engine still renders, but with no segments and no audio
    in the output.

    Stereo 44.1 kHz s16 is preserved deliberately: this WAV is BOTH analysed
    AND muxed back into the rendered video as the audio track. Downsampling
    here would be audible (mono panorama collapse, lost high-end). The
    analyzer does its own downsample on the in-memory waveform.
    """
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    # Probe duration cheaply so we can scale the ffmpeg timeout. The old
    # hard 120 s ceiling killed extraction on 30+ minute passthrough sources
    # mid-write and left an empty/broken WAV behind.
    try:
        _cap = cv2.VideoCapture(video_path)
        _fps = float(_cap.get(cv2.CAP_PROP_FPS) or 24.0)
        _n = float(_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        _cap.release()
        src_dur = (_n / _fps) if _fps > 0 else 0.0
    except Exception:
        src_dur = 0.0
    # ~1 s of wall-clock per 60 s of audio is conservative; clamp to a
    # 60 s floor and a 30 min ceiling to avoid runaway hangs.
    extract_timeout = int(max(60.0, min(1800.0, 60.0 + src_dur)))
    cmd = [
        ffmpeg_bin(), '-y', '-i', video_path,
        '-vn', '-ac', '2', '-ar', '44100', '-sample_fmt', 's16',
        tmp.name,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True,
                                timeout=extract_timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        log(f'Audio extraction failed: {exc}')
        try: os.remove(tmp.name)
        except OSError: pass
        return None
    if result.returncode != 0 or os.path.getsize(tmp.name) == 0:
        err = (result.stderr or b'')[:200].decode(errors='replace').strip()
        log(f'Audio extraction: no track / ffmpeg error ({err}).')
        try: os.remove(tmp.name)
        except OSError: pass
        return None
    return tmp.name


def prepare_datamosh_source(video_path: str, output_path: str,
                            log: LogFn, *,
                            mode: str = 'strip') -> bool:
    """Re-encode `video_path` for "true" datamosh in two flavours.

    Common flags (both modes):
      • ``-bf 0`` — kill B-frames. B-frames decode in non-display order
        and reference both directions; they'd reset the smear chain and
        ruin the effect.
      • ``-sc_threshold 0`` — forbid the encoder from inserting its own
        scene-cut I-frames. Without this libx264 silently sprinkles I's
        wherever motion changes a lot, breaking the long P-chain that
        the datamosh look depends on.
      • ``-g 99999 -keyint_min 99999`` — force the longest possible GOP
        so essentially everything is a P-frame.
      • ``-refs 1`` — each P-frame references only its immediate
        predecessor; produces the long, drifting motion-vector chain
        characteristic of "real" datamosh.
      • ``-preset slow`` — far better motion estimation than ultrafast.
        With ultrafast the encoder gives up on hard-to-track regions
        and emits intra blocks INSIDE P-frames, which look like static
        bricks instead of smear.

    Modes:
      • ``mode='strip'`` (cut-mode default) — additionally drops every
        source I-frame via ``select=not(eq(pict_type,I))``. Frame count
        SHRINKS, so this is only safe in cut mode where audio sync
        comes from random sampling, not from 1:1 frame alignment.
      • ``mode='longgop'`` (passthrough mode) — keeps every source
        frame; only the encode side enforces long-GOP P-only output.
        Frame count is preserved 1:1, so audio sync survives, but the
        decoder still produces the characteristic motion-vector smear
        on scene cuts (since the encoder isn't allowed to insert new
        I-frames where the source content jumps).
    """
    cmd = [ffmpeg_bin(), '-y', '-i', video_path]
    if mode == 'strip':
        cmd += ['-vf', 'select=not(eq(pict_type\\,I))', '-vsync', 'vfr']
    elif mode == 'longgop':
        # No filter: keep frame count 1:1 with source so the passthrough
        # loop can still align frames to audio. The encoder flags below
        # are what produce the datamosh look.
        pass
    else:
        log(f'Datamosh prebake: unknown mode {mode!r}, aborting.')
        return False
    cmd += [
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-bf', '0',
        '-g', '99999',
        '-keyint_min', '99999',
        '-sc_threshold', '0',
        '-refs', '1',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-an',
        output_path,
    ]
    # Scale the timeout to the source duration. `-preset slow` plus a
    # 30-min input would blow past any fixed ceiling; without a timeout
    # ffmpeg occasionally hangs on bad streams indefinitely.
    try:
        _cap = cv2.VideoCapture(video_path)
        _fps = float(_cap.get(cv2.CAP_PROP_FPS) or 24.0)
        _n = float(_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        _cap.release()
        src_dur = (_n / _fps) if _fps > 0 else 0.0
    except Exception:
        src_dur = 0.0
    # `slow` preset is roughly 1× realtime on modern hardware; allow 4×
    # headroom and clamp to a 5-min floor / 60-min ceiling.
    timeout = int(max(300.0, min(3600.0, 60.0 + src_dur * 4.0)))
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        log(f'Datamosh prebake failed: {exc}')
        try: os.remove(output_path)
        except OSError: pass
        return False
    if result.returncode != 0:
        log(f'Datamosh ffmpeg error: '
            f'{result.stderr[:200].decode(errors="replace")}')
        return False
    return True
