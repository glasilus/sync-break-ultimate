"""Frame source: VideoPool over multiple cv2.VideoCapture handles.

Optional in-memory cache: short source clips are decoded once and held
as a single contiguous (T, H, W, 3) uint8 ndarray, after which `read()`
and `set(POS_FRAMES)` become O(1) array indexing. This is a big win
for sessions with many short segments — the engine seeks ~once per
segment, and OpenCV's seek does a real keyframe-rewind + re-decode.

Eligibility is purely byte-budgeted (default 1.5 GB) and lazy: the
cache is filled on first `read()`, so opening the project doesn't pay
any decode cost up front. Long source clips (or multi-GB ones) bypass
the cache transparently and use the original VideoCapture.
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Default cap on per-clip cache size. 1.5 GB ≈ 1080p × 24fps × ~150s
# in BGR uint8. Override with env `VPC_SOURCE_CACHE_MAX_BYTES=0` to
# disable, or any byte count to widen.
DEFAULT_CACHE_MAX_BYTES = 1_500_000_000


def _cache_budget() -> int:
    raw = os.environ.get('VPC_SOURCE_CACHE_MAX_BYTES')
    if raw is None:
        return DEFAULT_CACHE_MAX_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_CACHE_MAX_BYTES


class CachedCap:
    """`cv2.VideoCapture`-shaped in-memory frame array.

    Implements the subset of the VideoCapture API engine.py actually
    uses: `read()`, `set(POS_FRAMES, n)`, `get(prop)`, `release()`,
    `isOpened()`. Frames are stored in BGR exactly as `VideoCapture.read`
    returns them, so the engine's downstream `cv2.cvtColor(BGR2RGB)`
    keeps working unchanged.

    Decode is lazy and one-shot: the first `read()` rewinds the source,
    pulls every frame into a stacked ndarray, and from then on all
    operations are pointer arithmetic. If the decode throws (corrupt
    file, codec mismatch) we fall back to a real VideoCapture
    transparently — the engine sees no difference.
    """

    def __init__(self, path: str, fps: float, total_frames: int,
                 w: int, h: int):
        self.path = path
        self.fps = fps
        self._w = w
        self._h = h
        self._declared_total = total_frames
        self._cache: Optional[np.ndarray] = None
        self._total_frames = total_frames
        self._pos = 0
        self._fallback_cap: Optional[cv2.VideoCapture] = None

    # ---- decode ----
    def _decode(self) -> None:
        """Materialize the whole file into one ndarray. On any error,
        flip into fallback mode and let `read` go through a normal
        VideoCapture."""
        try:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise RuntimeError('cv2.VideoCapture failed to open')
            frames: List[np.ndarray] = []
            while True:
                ret, fr = cap.read()
                if not ret:
                    break
                frames.append(fr)
            cap.release()
            if not frames:
                raise RuntimeError('zero frames decoded')
            arr = np.stack(frames)
            self._cache = arr
            self._total_frames = arr.shape[0]
        except Exception:
            # Cache build failed — fall through to a real cap so the
            # render keeps going. CachedCap will delegate from now on.
            self._fallback_cap = cv2.VideoCapture(self.path)

    # ---- VideoCapture-shaped API ----
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cache is None and self._fallback_cap is None:
            self._decode()
        if self._fallback_cap is not None:
            return self._fallback_cap.read()
        if self._cache is None or self._pos >= self._total_frames:
            return False, None
        fr = self._cache[self._pos]
        self._pos += 1
        # Return a view; engine immediately copies via cvtColor/resize
        # so writes won't bleed back into the cache.
        return True, fr

    def set(self, prop: int, value) -> bool:
        if self._fallback_cap is not None:
            return bool(self._fallback_cap.set(prop, value))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            n = int(value)
            if self._total_frames > 0:
                n = max(0, min(n, self._total_frames - 1))
            self._pos = max(0, n)
            return True
        return False

    def get(self, prop: int) -> float:
        if self._fallback_cap is not None:
            return self._fallback_cap.get(prop)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._total_frames)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        self._cache = None
        if self._fallback_cap is not None:
            try:
                self._fallback_cap.release()
            except Exception:
                pass
            self._fallback_cap = None


def _eligible_for_cache(total: int, w: int, h: int, budget: int) -> bool:
    """Cheap pre-decode probe — would the file fit in the byte budget?

    Uses BGR uint8 (3 bytes/pixel) which is what VideoCapture returns.
    Files larger than the budget are skipped and use the original
    VideoCapture path."""
    if budget <= 0 or total <= 0 or w <= 0 or h <= 0:
        return False
    return total * w * h * 3 <= budget


class VideoPool:
    """Manages multiple VideoCapture handles; selects randomly per segment."""

    def __init__(self, paths: List[str],
                 cache_max_bytes: Optional[int] = None):
        if not paths:
            raise ValueError('VideoPool requires at least one path')
        self.paths = paths
        self.caps: List = []  # cv2.VideoCapture | CachedCap
        self.fps_list: List[float] = []
        self.total_frames_list: List[int] = []
        self.durations: List[float] = []
        self.sizes: List[Tuple[int, int]] = []
        self._cached_flags: List[bool] = []

        budget = (cache_max_bytes if cache_max_bytes is not None
                  else _cache_budget())

        for path in paths:
            probe = cv2.VideoCapture(path)
            if not probe.isOpened():
                raise RuntimeError(f'Cannot open video: {path}')
            fps = probe.get(cv2.CAP_PROP_FPS) or 24.0
            total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if _eligible_for_cache(total, w, h, budget):
                # Hand off to lazy CachedCap — close the probe handle
                # so we don't hold two ffmpeg sessions on the same file.
                probe.release()
                cap = CachedCap(path, fps, total, w, h)
                self._cached_flags.append(True)
            else:
                cap = probe
                self._cached_flags.append(False)

            self.caps.append(cap)
            self.fps_list.append(fps)
            self.total_frames_list.append(total)
            self.durations.append(total / fps if fps else 0.0)
            self.sizes.append((w, h))

        self.vid_fps = self.fps_list[0]
        self.vid_total_frames = self.total_frames_list[0]
        self.vid_duration = max(self.durations) if self.durations else 0.0
        self.primary_size = self.sizes[0] if self.sizes else (0, 0)

    @property
    def any_cached(self) -> bool:
        """True if at least one source uses the in-memory cache. Engine
        can use this to log a one-line note ('Source(s) cached, X MB')."""
        return any(self._cached_flags)

    def random_cap(self):
        i = random.randrange(len(self.caps))
        return self.caps[i], self.fps_list[i], self.total_frames_list[i], self.durations[i]

    def primary_cap(self):
        return self.caps[0], self.fps_list[0], self.total_frames_list[0], self.durations[0]

    def release_all(self):
        for cap in self.caps:
            try:
                cap.release()
            except Exception:
                pass
