"""Unit tests for FFmpegSink argv composition.

These run without any real video files — we only inspect the assembled
ffmpeg command to confirm flags land where they should. This is the cheap
regression net for the encoder/quality/tune work in steps 1 and 2.
"""
from __future__ import annotations

import pytest

from vpc.render.sink import FFmpegSink, EXPORT_FORMATS


def _make(**kw) -> list[str]:
    """Construct a sink and return its assembled argv (without spawning)."""
    defaults = dict(
        width=640, height=360, fps=24,
        audio_path='audio.wav', output_path='out.mp4',
        vcodec='libx264', acodec='aac', pix_fmt='yuv420p',
        preset='medium', crf=18, target_duration=2.0,
    )
    defaults.update(kw)
    return FFmpegSink(**defaults)._cmd


def test_x264_carries_preset_and_crf():
    cmd = _make(vcodec='libx264', preset='medium', crf=18)
    assert '-preset' in cmd and 'medium' in cmd
    assert '-crf' in cmd and '18' in cmd


def test_x265_carries_preset_and_crf():
    cmd = _make(vcodec='libx265', preset='slow', crf=20,
                output_path='out.mp4',
                extra_v_flags=['-tag:v', 'hvc1'])
    assert '-preset' in cmd and 'slow' in cmd
    assert '-crf' in cmd and '20' in cmd
    assert '-tag:v' in cmd and 'hvc1' in cmd


def test_vp9_uses_deadline_not_preset():
    cmd = _make(vcodec='libvpx-vp9', acodec='libopus', preset='medium', crf=32,
                output_path='out.webm', extra_v_flags=['-row-mt', '1', '-b:v', '0'])
    assert '-preset' not in cmd
    assert '-deadline' in cmd and 'good' in cmd
    assert '-cpu-used' in cmd
    assert '-crf' in cmd and '32' in cmd


def test_prores_skips_preset_and_crf():
    cmd = _make(vcodec='prores_ks', acodec='pcm_s16le',
                pix_fmt='yuv422p10le', preset='medium', crf=18,
                output_path='out.mov',
                extra_v_flags=['-profile:v', '3'])
    assert '-preset' not in cmd
    assert '-crf' not in cmd
    assert '-profile:v' in cmd and '3' in cmd


def test_target_duration_present_but_no_shortest():
    """-t bounds output, but -shortest must NOT be set (truncation bug)."""
    cmd = _make(target_duration=12.345)
    assert '-t' in cmd
    assert '-shortest' not in cmd
    # check value formatting: 3 decimals
    idx = cmd.index('-t')
    assert cmd[idx + 1] == '12.345'


def test_faststart_only_for_mp4_and_mov():
    mp4 = _make(output_path='clip.mp4')
    mov = _make(output_path='clip.mov', vcodec='prores_ks',
                acodec='pcm_s16le', pix_fmt='yuv422p10le')
    mkv = _make(output_path='clip.mkv')
    webm = _make(output_path='clip.webm', vcodec='libvpx-vp9',
                 acodec='libopus')
    assert '+faststart' in mp4
    assert '+faststart' in mov
    assert '+faststart' not in mkv
    assert '+faststart' not in webm


def test_export_formats_are_consistent():
    """Every entry has the required keys; pix_fmt is non-empty."""
    required = {'ext', 'vcodec', 'acodec', 'pix_fmt', 'extra_v'}
    for label, spec in EXPORT_FORMATS.items():
        missing = required - spec.keys()
        assert not missing, f'{label} missing keys: {missing}'
        assert spec['pix_fmt'], f'{label} has empty pix_fmt'
        assert isinstance(spec['extra_v'], list), f'{label} extra_v not list'


def _input_pix_fmt(cmd: list[str]) -> str:
    """Extract the rawvideo pipe's pixel format (the -pix_fmt that appears
    before -i pipe:0)."""
    pipe_idx = cmd.index('pipe:0')
    pre = cmd[:pipe_idx]
    return pre[pre.index('-pix_fmt') + 1]


def test_input_pix_fmt_yuv420p_when_output_is_yuv420p():
    """yuv420p output → pipe yuv420p (1.5 bytes/pixel, half the bandwidth
    of rgb24). This is the Step 2 optimization."""
    cmd = _make(vcodec='libx264', pix_fmt='yuv420p')
    assert _input_pix_fmt(cmd) == 'yuv420p'


def test_input_pix_fmt_rgb24_for_prores_10bit():
    """ProRes 4:2:2 10-bit must stay on rgb24 input — converting through
    I420 would discard chroma detail before ffmpeg sees the frame."""
    cmd = _make(vcodec='prores_ks', acodec='pcm_s16le',
                pix_fmt='yuv422p10le', output_path='out.mov',
                extra_v_flags=['-profile:v', '3'])
    assert _input_pix_fmt(cmd) == 'rgb24'


def test_input_pix_fmt_explicit_override():
    """Explicit input_pix_fmt wins over the auto-pick (escape hatch)."""
    cmd = _make(vcodec='libx264', pix_fmt='yuv420p', input_pix_fmt='rgb24')
    assert _input_pix_fmt(cmd) == 'rgb24'


def test_pack_frame_rgb24_passthrough():
    """_pack_frame on 'rgb24' is identity over the bytes."""
    import numpy as np
    from vpc.render.engine import BreakcoreEngine
    rgb = np.arange(360 * 480 * 3, dtype=np.uint8).reshape(360, 480, 3)
    out = BreakcoreEngine._pack_frame(rgb, 'rgb24')
    assert out == rgb.tobytes()
    assert len(out) == 360 * 480 * 3


def test_pack_frame_yuv420p_size():
    """yuv420p packs to exactly 1.5 bytes per pixel (planar I420)."""
    import numpy as np
    from vpc.render.engine import BreakcoreEngine
    rgb = np.full((360, 480, 3), 128, dtype=np.uint8)
    out = BreakcoreEngine._pack_frame(rgb, 'yuv420p')
    assert len(out) == 360 * 480 * 3 // 2


def test_pack_frame_unknown_falls_back_to_rgb24():
    """Defensive: unknown format must not silently produce wrong bytes."""
    import numpy as np
    from vpc.render.engine import BreakcoreEngine
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    assert BreakcoreEngine._pack_frame(rgb, 'nv12_made_up') == rgb.tobytes()
