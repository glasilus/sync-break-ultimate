"""Video source, overlay manager, and realtime processing engine."""
import os
import random
import threading
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageSequence

from rt_audio import AudioAnalyzer, RTAudioStats, make_segment_from_stats
from effects import RGBShiftEffect, PixelSortEffect, DatamoshEffect, FlashEffect
from analyzer import SegmentType

# Segment types that trigger a video cut (jump to random frame)
_CUT_TYPES = {SegmentType.IMPACT, SegmentType.BUILD, SegmentType.DROP}


class VideoSource:
    """Источник видео"""
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30

        self._cache = []
        self._cap_lock = threading.Lock()

    def precache(self, n=30, width=640, height=360):
        """Pre-read N random frames into memory. Call from a background thread."""
        frames = []
        for _ in range(n):
            if self.total_frames == 0:
                break
            frame_num = random.randint(0, max(0, self.total_frames - 10))
            with self._cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (width, height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        self._cache = frames

    def get_random_frame(self, width=640, height=360):
        if self._cache:
            return random.choice(self._cache).copy()
        if self.total_frames == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        frame_num = random.randint(0, max(0, self.total_frames - 10))
        with self._cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_sequential_frame(self, width=640, height=360):
        with self._cap_lock:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.zeros((height, width, 3), dtype=np.uint8)

    def close(self):
        with self._cap_lock:
            if self.cap:
                self.cap.release()


class VideoPool:
    """Wraps multiple VideoSource objects; distributes frame requests across them."""

    def __init__(self, width: int = 640, height: int = 360):
        self.width = width
        self.height = height
        self._sources: list = []
        self._seq_index: int = 0

    def load(self, paths: list):
        """Load a list of video paths, replacing any existing sources."""
        for src in self._sources:
            src.close()
        self._sources = []
        self._seq_index = 0
        for path in paths:
            src = VideoSource(path)
            self._sources.append(src)

    def precache(self, n: int = 30):
        """Pre-cache frames from all sources in background threads."""
        import threading
        for src in self._sources:
            t = threading.Thread(
                target=src.precache,
                args=(n, self.width, self.height),
                daemon=True,
            )
            t.start()

    def get_random_frame(self):
        if not self._sources:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        src = random.choice(self._sources)
        return src.get_random_frame(self.width, self.height)

    def get_sequential_frame(self):
        if not self._sources:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        src = self._sources[self._seq_index % len(self._sources)]
        frame = src.get_sequential_frame(self.width, self.height)
        # Advance to next source when current one wraps (detected by frame being zeros)
        return frame

    def close(self):
        for src in self._sources:
            src.close()
        self._sources = []

    @property
    def loaded(self) -> bool:
        return len(self._sources) > 0


class OverlayManager:
    """Менеджер оверлеев"""
    def __init__(self):
        self.overlays = []

    def load_overlays(self, folder_path):
        self.overlays = []
        if not os.path.exists(folder_path):
            return
        supported_ext = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        for file in os.listdir(folder_path):
            if file.lower().endswith(supported_ext):
                path = os.path.join(folder_path, file)
                try:
                    if file.lower().endswith('.gif'):
                        gif = Image.open(path)
                        frame = next(ImageSequence.Iterator(gif))
                        if frame.mode != 'RGBA':
                            frame = frame.convert('RGBA')
                        img_np = np.array(frame)
                    else:
                        img = Image.open(path)
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        img_np = np.array(img)
                    self.overlays.append({'image': img_np, 'name': file})
                except Exception as e:
                    print(f"Error loading overlay {file}: {e}")

    def get_random_overlay(self, max_width, max_height):
        if not self.overlays:
            return None, None, None
        overlay = random.choice(self.overlays)
        img = overlay['image']
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0) * random.uniform(0.3, 0.8)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w > 0 and new_h > 0:
            return cv2.resize(img, (new_w, new_h)), new_w, new_h
        return None, None, None


class RealtimeEngine:
    """Движок реального времени"""
    def __init__(self, width=640, height=360):
        self.width = width
        self.height = height

        self.audio = AudioAnalyzer()
        self.video_pool = VideoPool(width, height)
        self.overlay_mgr = OverlayManager()

        self.running = False
        self.frame_buffer = deque(maxlen=5)
        self.audio_stats = {
            'rms': 0.0, 'beat': False, 'noisy': False,
            'bass': 0.0, 'mid': 0.0, 'treble': 0.0,
            'flatness': 0.3, 'trend_slope': 0.0, 'rms_mean': 0.001, 'is_noisy': False,
        }
        self.settings = {
            'chaos': 0.6,
            'fx_rgb': True, 'fx_stutter': True, 'fx_flash': True,
            'fx_pixel_sort': True, 'fx_overlays': True, 'fx_datamosh': True,
            'sequential_mode': False, 'overlay_intensity': 0.5,
        }
        # All effect types that should fire when audio is active
        _active_types = [
            SegmentType.IMPACT, SegmentType.BUILD, SegmentType.DROP,
            SegmentType.SUSTAIN, SegmentType.NOISE,
        ]
        self._effects = [
            FlashEffect(enabled=True,    chance=0.5),
            RGBShiftEffect(enabled=True, chance=0.6),
            PixelSortEffect(enabled=True, chance=0.5),
            DatamoshEffect(enabled=True,  chance=0.4),
        ]
        for _fx in self._effects:
            _fx.trigger_types = _active_types
        self._prev_rms = 0.0
        self._frame_t  = 0.0

    def set_video_source(self, video_path: str):
        return self.set_video_sources([video_path])

    def set_video_sources(self, video_paths: list):
        try:
            self.video_pool.load(video_paths)
            self.video_pool.precache(30)
            return True
        except Exception as e:
            print(f"Error loading video(s): {e}")
            return False

    def load_overlays(self, folder_path):
        self.overlay_mgr.load_overlays(folder_path)

    def start(self, audio_device_idx=None):
        if not self.video_pool.loaded:
            return False, "No video source loaded"
        if not self.audio.start(audio_device_idx):
            return False, "Failed to start audio capture"
        self.running = True
        return True, "Engine started"

    def stop(self):
        self.running = False
        self.audio.stop()
        # Don't close video_pool — user may restart without reloading video

    def process_frame(self):
        if not self.running or not self.video_pool.loaded:
            return None
        try:
            audio_data = self.audio.analyze_chunk()
            if audio_data:
                self.audio_stats = {
                    'rms':         audio_data.rms,
                    'beat':        audio_data.beat,
                    'bass':        audio_data.bass,
                    'mid':         audio_data.mid,
                    'treble':      audio_data.treble,
                    'flatness':    audio_data.flatness,
                    'trend_slope': audio_data.trend_slope,
                    'rms_mean':    audio_data.rms_mean,
                    'is_noisy':    audio_data.is_noisy,
                    'noisy':       audio_data.is_noisy,
                }

            # Build segment BEFORE frame selection so cut decision uses it
            _stats = RTAudioStats(
                rms=self.audio_stats.get('rms', 0.0),
                beat=self.audio_stats.get('beat', False),
                bass=self.audio_stats.get('bass', 0.0),
                mid=self.audio_stats.get('mid', 0.0),
                treble=self.audio_stats.get('treble', 0.0),
                flatness=self.audio_stats.get('flatness', 0.3),
                trend_slope=self.audio_stats.get('trend_slope', 0.0),
                rms_mean=self.audio_stats.get('rms_mean', 0.001),
                is_noisy=self.audio_stats.get('is_noisy', False),
            )
            seg = make_segment_from_stats(_stats, self._prev_rms, self._frame_t,
                                          gate=self.audio.effective_gate)
            self._prev_rms = _stats.rms
            self._frame_t += 0.033

            _calibrated = self.audio._calibration_done
            _active = self.audio_stats['rms'] > self.audio.effective_gate

            # Frame selection:
            #   sequential_mode=True  → always sequential (ambient / no-cut mode)
            #   default               → sequential playback; cut ONLY on audio events
            #                          (IMPACT, BUILD, DROP, or SUSTAIN+beat)
            if self.settings.get('sequential_mode'):
                frame = self.video_pool.get_sequential_frame()
            else:
                should_cut = (
                    _calibrated and _active
                    and (
                        seg.type in _CUT_TYPES
                        or (seg.type == SegmentType.SUSTAIN
                            and self.audio_stats.get('beat', False))
                    )
                )
                if should_cut:
                    frame = self.video_pool.get_random_frame()
                else:
                    frame = self.video_pool.get_sequential_frame()

            self.frame_buffer.append(frame.copy())

            if not _calibrated:
                return frame

            if _active:
                _FX_KEYS = {
                    'FlashEffect':     'fx_flash',
                    'RGBShiftEffect':  'fx_rgb',
                    'PixelSortEffect': 'fx_pixel_sort',
                    'DatamoshEffect':  'fx_datamosh',
                }
                for fx in self._effects:
                    key = _FX_KEYS.get(type(fx).__name__, 'fx_unknown')
                    if self.settings.get(key, True):
                        frame = fx.apply(frame, seg, draft=True)

            if self.audio_stats['beat']:
                if self.settings['fx_stutter'] and random.random() < 0.4 + self.settings['chaos'] * 0.3:
                    if self.frame_buffer:
                        prev_frame = random.choice(list(self.frame_buffer))
                        frame = cv2.addWeighted(frame, 0.3, prev_frame, 0.7, 0)
                if self.settings['fx_flash'] and random.random() < 0.3:
                    flash = np.full_like(frame, 255)
                    intensity = min(self.audio_stats['rms'] * 5, 0.8)
                    frame = cv2.addWeighted(frame, 1 - intensity, flash, intensity, 0)

            if _active and self.settings['fx_rgb'] and random.random() < 0.1:
                b, g, r = cv2.split(frame)
                shift = random.randint(1, 3)
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                r = cv2.warpAffine(r, M, (self.width, self.height),
                                   borderMode=cv2.BORDER_WRAP)
                frame = cv2.merge([b, g, r])

            if self.audio_stats['noisy'] and self.settings['fx_pixel_sort'] and random.random() < 0.3:
                h, w = frame.shape[:2]
                result = frame.copy()
                for _ in range(int(h * 0.1)):
                    y = random.randint(0, h - 1)
                    line = result[y, :, :].copy()
                    brightness = line[:, 0] * 0.114 + line[:, 1] * 0.587 + line[:, 2] * 0.299
                    result[y, :, :] = line[np.argsort(brightness)]
                frame = result

            if _active and self.settings['fx_overlays'] and random.random() < self.settings['overlay_intensity']:
                overlay, ov_w, ov_h = self.overlay_mgr.get_random_overlay(
                    self.width // 2, self.height // 2)
                if overlay is not None:
                    x = random.randint(0, max(0, self.width  - ov_w))
                    y = random.randint(0, max(0, self.height - ov_h))
                    if overlay.shape[2] == 4:
                        alpha = overlay[:, :, 3] / 255.0
                        roi = frame[y:y+ov_h, x:x+ov_w]
                        if roi.shape[:2] == overlay[:, :, :3].shape[:2]:
                            for c in range(3):
                                roi[:, :, c] = (roi[:, :, c] * (1 - alpha)
                                                + overlay[:, :, c] * alpha)

            return frame

        except Exception as e:
            print(f"Frame processing error: {e}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
