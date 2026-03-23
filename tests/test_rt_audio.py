# tests/test_rt_audio.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from dataclasses import dataclass


def make_stats(rms=0.05, beat=False, bass=0.1, mid=0.05, treble=0.02,
               flatness=0.3, trend_slope=0.0, rms_mean=0.04, is_noisy=False):
    """Helper: build RTAudioStats-like dict for testing type selection."""
    return dict(rms=rms, beat=beat, bass=bass, mid=mid, treble=treble,
                flatness=flatness, trend_slope=trend_slope,
                rms_mean=rms_mean, is_noisy=is_noisy)


def test_band_rms_pure_sine():
    """band_rms should return non-zero only in the correct frequency band."""
    rate = 44100
    chunk = 1024
    # 100 Hz sine → should appear in bass (20–300 Hz), not mid or treble
    t = np.arange(chunk) / rate
    audio = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    fft = np.abs(np.fft.rfft(audio * np.hanning(chunk)))
    freqs = np.fft.rfftfreq(chunk, 1.0 / rate)

    def band_rms(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sqrt(np.mean(fft[mask] ** 2))) if mask.any() else 0.0

    bass = band_rms(20, 300)
    mid = band_rms(300, 3000)
    treble = band_rms(3000, 16000)
    assert bass > mid * 5, f"Expected bass to dominate, got bass={bass:.3f} mid={mid:.3f}"
    assert bass > treble * 5


# Import after rt.py is updated
import importlib, types

def _make_seg(stats_dict):
    """Call make_segment_from_stats() from rt module."""
    import rt
    s = rt.RTAudioStats(**stats_dict)
    return rt.make_segment_from_stats(s, prev_rms=stats_dict['rms'] * 0.9, t=0.0)

def test_seg_type_build():
    from analyzer import SegmentType
    stats = make_stats(trend_slope=0.01, rms_mean=0.04)  # slope > 0.04*0.07=0.0028
    seg = _make_seg(stats)
    assert seg.type == SegmentType.BUILD

def test_seg_type_drop():
    from analyzer import SegmentType
    stats = make_stats(trend_slope=-0.01, rms=0.06, rms_mean=0.04)  # rms > mean, slope < -thresh
    seg = _make_seg(stats)
    assert seg.type == SegmentType.DROP

def test_seg_type_noise():
    from analyzer import SegmentType
    stats = make_stats(is_noisy=True, trend_slope=0.0)
    seg = _make_seg(stats)
    assert seg.type == SegmentType.NOISE

def test_seg_type_impact():
    from analyzer import SegmentType
    # beat=True, bass > mid and treble
    stats = make_stats(beat=True, bass=0.5, mid=0.1, treble=0.05, trend_slope=0.0, is_noisy=False)
    seg = _make_seg(stats)
    assert seg.type == SegmentType.IMPACT

def test_seg_type_sustain():
    from analyzer import SegmentType
    stats = make_stats(rms=0.06, rms_mean=0.04)  # rms > mean * 1.2 = 0.048
    seg = _make_seg(stats)
    assert seg.type == SegmentType.SUSTAIN

def test_seg_type_silence():
    from analyzer import SegmentType
    stats = make_stats(rms=0.01, rms_mean=0.04)
    seg = _make_seg(stats)
    assert seg.type == SegmentType.SILENCE

def test_seg_intensity_clamped():
    stats = make_stats(rms=99.0, rms_mean=0.04)
    seg = _make_seg(stats)
    assert 0.0 <= seg.intensity <= 1.0


def test_video_source_cache_populated(tmp_path):
    """VideoSource pre-cache should have frames after precache() call."""
    import cv2
    from rt import VideoSource
    # Create a tiny synthetic video
    out_path = str(tmp_path / "test.avi")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (64, 36))
    for _ in range(60):
        out.write(np.zeros((36, 64, 3), dtype=np.uint8))
    out.release()
    vs = VideoSource(out_path)
    vs.precache(n=10, width=64, height=36)
    assert len(vs._cache) == 10
    frame = vs.get_random_frame(64, 36)
    assert frame.shape == (36, 64, 3)
    vs.close()
