"""Source pre-decode cache (CachedCap + VideoPool eligibility).

Doesn't require any real video files: we monkey-patch cv2.VideoCapture
so the decode loop reads from a synthetic frame stream. This isolates
the cache logic from codec quirks and keeps the test fast (<100ms).
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pytest

import cv2

from vpc.render import source as src_mod


# ----- eligibility -----

def test_eligible_within_budget():
    # 100 frames × 320×240 × 3 = 23 MB → easily fits in default 1.5 GB
    assert src_mod._eligible_for_cache(100, 320, 240, 1_500_000_000)


def test_not_eligible_when_over_budget():
    # 4000 frames × 1920×1080 × 3 ≈ 24 GB → exceeds default
    assert not src_mod._eligible_for_cache(4000, 1920, 1080, 1_500_000_000)


def test_not_eligible_with_zero_budget():
    """Setting VPC_SOURCE_CACHE_MAX_BYTES=0 disables the cache entirely."""
    assert not src_mod._eligible_for_cache(10, 320, 240, 0)


def test_not_eligible_when_total_unknown():
    """Streams with no reported frame count (live captures, broken
    headers) bypass caching to avoid an unbounded decode."""
    assert not src_mod._eligible_for_cache(0, 320, 240, 1_000_000_000)


def test_budget_env_override(monkeypatch):
    monkeypatch.setenv('VPC_SOURCE_CACHE_MAX_BYTES', '0')
    assert src_mod._cache_budget() == 0
    monkeypatch.setenv('VPC_SOURCE_CACHE_MAX_BYTES', '5000000')
    assert src_mod._cache_budget() == 5_000_000
    monkeypatch.setenv('VPC_SOURCE_CACHE_MAX_BYTES', 'garbage')
    assert src_mod._cache_budget() == src_mod.DEFAULT_CACHE_MAX_BYTES


# ----- CachedCap behavior -----

class _FakeCap:
    """Minimal cv2.VideoCapture stand-in driven by a list of frames."""
    _registry: dict = {}

    def __init__(self, path: str):
        self._frames: List[np.ndarray] = list(_FakeCap._registry.get(path, []))
        self._idx = 0

    def isOpened(self): return True
    def release(self): pass
    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        fr = self._frames[self._idx]
        self._idx += 1
        return True, fr
    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._idx = max(0, min(int(val), len(self._frames) - 1))
            return True
        return False
    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT: return len(self._frames)
        return 0


@pytest.fixture
def fake_video(monkeypatch):
    frames = [np.full((4, 6, 3), i, dtype=np.uint8) for i in range(8)]
    _FakeCap._registry = {'/fake/clip.mp4': frames}
    monkeypatch.setattr(cv2, 'VideoCapture', _FakeCap)
    return '/fake/clip.mp4', frames


def test_cached_cap_decodes_lazily(fake_video):
    path, frames = fake_video
    cc = src_mod.CachedCap(path, fps=24.0, total_frames=8, w=6, h=4)
    assert cc._cache is None  # not decoded until first read
    ok, fr = cc.read()
    assert ok
    assert cc._cache is not None
    assert fr.tolist() == frames[0].tolist()


def test_cached_cap_seek_and_read(fake_video):
    path, frames = fake_video
    cc = src_mod.CachedCap(path, fps=24.0, total_frames=8, w=6, h=4)
    cc.read()  # force decode
    cc.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ok, fr = cc.read()
    assert ok and int(fr[0, 0, 0]) == 5
    # Reading past the end returns (False, None).
    cc.set(cv2.CAP_PROP_POS_FRAMES, 99)
    ok2, _ = cc.read()
    # After clamping: pos lands at last frame, one read succeeds, next fails.
    cc.set(cv2.CAP_PROP_POS_FRAMES, 7)
    cc.read()
    ok3, _ = cc.read()
    assert ok3 is False


def test_cached_cap_get_returns_metadata(fake_video):
    path, _ = fake_video
    cc = src_mod.CachedCap(path, fps=30.0, total_frames=8, w=6, h=4)
    assert cc.get(cv2.CAP_PROP_FPS) == 30.0
    assert cc.get(cv2.CAP_PROP_FRAME_COUNT) == 8
    assert cc.get(cv2.CAP_PROP_FRAME_WIDTH) == 6
    assert cc.get(cv2.CAP_PROP_FRAME_HEIGHT) == 4


def test_cached_cap_falls_back_when_decode_yields_zero_frames(monkeypatch):
    """Corrupt clip → decode raises → CachedCap silently switches to a
    real VideoCapture, so the engine doesn't crash."""
    _FakeCap._registry = {'/fake/empty.mp4': []}
    monkeypatch.setattr(cv2, 'VideoCapture', _FakeCap)

    cc = src_mod.CachedCap('/fake/empty.mp4', fps=24.0,
                           total_frames=0, w=6, h=4)
    ok, _ = cc.read()
    assert ok is False
    # _fallback_cap is set; cache stays None.
    assert cc._fallback_cap is not None
    assert cc._cache is None


# ----- VideoPool eligibility -----

def test_video_pool_uses_cache_for_small_clip(monkeypatch):
    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(8)]

    class _ProbeCap(_FakeCap):
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS: return 24.0
            if prop == cv2.CAP_PROP_FRAME_COUNT: return len(self._frames)
            if prop == cv2.CAP_PROP_FRAME_WIDTH: return 6.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT: return 4.0
            return 0.0

    _FakeCap._registry = {'/fake/short.mp4': frames}
    monkeypatch.setattr(cv2, 'VideoCapture', _ProbeCap)

    pool = src_mod.VideoPool(['/fake/short.mp4'])
    assert pool.any_cached
    assert isinstance(pool.caps[0], src_mod.CachedCap)


def test_video_pool_skips_cache_when_budget_zero(monkeypatch):
    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(8)]

    class _ProbeCap(_FakeCap):
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS: return 24.0
            if prop == cv2.CAP_PROP_FRAME_COUNT: return len(self._frames)
            if prop == cv2.CAP_PROP_FRAME_WIDTH: return 6.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT: return 4.0
            return 0.0

    _FakeCap._registry = {'/fake/short.mp4': frames}
    monkeypatch.setattr(cv2, 'VideoCapture', _ProbeCap)

    pool = src_mod.VideoPool(['/fake/short.mp4'], cache_max_bytes=0)
    assert not pool.any_cached
    assert not isinstance(pool.caps[0], src_mod.CachedCap)
