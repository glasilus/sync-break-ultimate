"""HW-encoder smoke test.

Activated by `RUN_HW_SMOKE=1`. Synthesises a 3s testsrc + sine fixture
and renders it once with each available HW encoder. We assert the
resulting file exists and has the expected frame count — the same
contract as the regression test, just sweeping codecs.

The point isn't to benchmark (run `tools/bench_render.py` for that),
it's to confirm the integration didn't break: argv composition, sink
fallback path, ETA loop. If a HW encoder dies at init, the engine's
fallback branch should take over and the file should still be produced
via libx264.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from vpc.render import BreakcoreEngine, RENDER_FINAL
from vpc.render.encoders import available_specs
from vpc.render.sink import ffmpeg_bin


RUN = os.environ.get('RUN_HW_SMOKE') == '1'
ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tools' / '.bench_fixtures'


def _ensure(duration: float = 3.0) -> tuple[str, str]:
    FIX.mkdir(parents=True, exist_ok=True)
    v = FIX / f'testsrc_{int(duration)}s.mp4'
    a = FIX / f'sine_{int(duration)}s.wav'
    ff = ffmpeg_bin()
    if not v.exists():
        subprocess.run([ff, '-y', '-f', 'lavfi',
                        '-i', f'testsrc=duration={duration}:size=1280x720:rate=24',
                        '-pix_fmt', 'yuv420p', '-c:v', 'libx264',
                        '-preset', 'ultrafast', str(v)],
                       check=True, capture_output=True)
    if not a.exists():
        subprocess.run([ff, '-y', '-f', 'lavfi',
                        '-i', f'sine=frequency=220:duration={duration}',
                        '-ar', '44100', str(a)],
                       check=True, capture_output=True)
    return str(v), str(a)


def _ffprobe_frames(path: str) -> int:
    ff = ffmpeg_bin()
    probe = Path(ff).with_name('ffprobe.exe' if os.name == 'nt' else 'ffprobe')
    cmd = [str(probe) if probe.exists() else 'ffprobe',
           '-v', 'error', '-select_streams', 'v:0',
           '-count_frames', '-show_entries', 'stream=nb_read_frames',
           '-of', 'json', path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    s = (json.loads(r.stdout or '{}').get('streams') or [{}])[0]
    return int(s.get('nb_read_frames') or 0)


HW_LABELS = [s.label for s in available_specs() if s.is_hw and s.label.startswith('H.264')]


@pytest.mark.skipif(not RUN, reason='Set RUN_HW_SMOKE=1.')
@pytest.mark.skipif(not HW_LABELS, reason='No HW encoders on this build.')
@pytest.mark.parametrize('codec_label', HW_LABELS)
def test_hw_encoder_produces_valid_output(tmp_path, codec_label):
    video, audio = _ensure(3.0)
    out = tmp_path / 'hw.mp4'
    cfg = {
        'video_paths': [video], 'audio_path': audio,
        'output_path': str(out),
        'fps': 24, 'crf': 22, 'export_preset': 'fast',
        'video_codec': codec_label,
        'tune': 'none', 'quality_preset': 'Custom',
        'resolution': '480p', 'resolution_mode': 'preset',
        'silence_mode': 'none',
        'chaos_level': 0.5, 'threshold': 1.2, 'transient_thresh': 0.5,
        'min_cut_duration': 0.05,
        'snap_to_beat': False, 'snap_tolerance': 0.05,
        'use_scene_detect': False,
    }
    eng = BreakcoreEngine(cfg)
    ok = eng.run(render_mode=RENDER_FINAL, max_output_duration=3.0)
    assert ok, f'engine.run returned False for {codec_label}'
    assert out.exists() and out.stat().st_size > 1000, codec_label
    # Frame-count contract holds whether we used HW or the fallback.
    frames = _ffprobe_frames(str(out))
    expected = round(3.0 * 24)
    assert abs(frames - expected) <= 2, (
        f'{codec_label}: got {frames} frames, expected ~{expected}')
