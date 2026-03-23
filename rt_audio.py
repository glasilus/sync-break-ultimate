"""Real-time audio capture, analysis, and segment generation."""
import queue
import time
from collections import deque
from dataclasses import dataclass as _dc

import numpy as np
import sounddevice as _sd

from analyzer import Segment, SegmentType

# ── UI colour constants (shared across rt_* modules) ─────────────────────────
C_SILVER     = '#C0C0C0'
C_DARK_GRAY  = '#808080'
C_BLACK      = '#000000'
C_WHITE      = '#FFFFFF'
C_TITLE_BLUE = '#000080'
C_TEXT_BLACK = '#000000'


@_dc
class RTAudioStats:
    rms: float
    beat: bool
    bass: float
    mid: float
    treble: float
    flatness: float
    trend_slope: float
    rms_mean: float
    is_noisy: bool


class AudioAnalyzer:
    """Low-latency audio capture and spectral analysis via sounddevice."""
    CHUNK = 1024
    RATE = 44100
    BEAT_COOLDOWN = 0.08
    BEAT_RATIO = 1.3
    # Absolute RMS floor — below this level nothing is considered "audio".
    # Prevents AGC-style adaptation to background noise / silent desktop audio.
    # Typical mic noise floor: ~0.001–0.003. Music/speech starts ~0.01+.
    NOISE_FLOOR = 0.005

    # How many chunks to collect before declaring calibration done (~2 s).
    CALIBRATION_CHUNKS = 86

    def __init__(self):
        self.rate = self.RATE
        self.chunk = self.CHUNK
        self.stream = None
        self.running = False
        self._queue = queue.Queue(maxsize=10)

        self._rms_history = deque(maxlen=30)   # for rms_mean and beat
        self._flat_history = deque(maxlen=50)  # rolling flatness baseline
        self._trend_history = deque(maxlen=10) # for BUILD/DROP slope
        self._chunk_count = 0

        # Auto-calibrated gate: starts at the static NOISE_FLOOR, gets raised
        # to 4× the measured device noise after CALIBRATION_CHUNKS chunks.
        self._gate = self.NOISE_FLOOR
        self._calibration_buf = deque(maxlen=self.CALIBRATION_CHUNKS)
        self._calibration_done = False
        self._last_beat_time = 0.0
        self._last_rms = 0.0
        # Multiplier applied on top of auto-calibrated gate (user-adjustable)
        self.gate_multiplier = 1.0

    @property
    def effective_gate(self):
        return self._gate * self.gate_multiplier

    def _callback(self, indata, frames, time_info, status):
        if self.running:
            try:
                self._queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

    def start(self, device_index=None):
        """Start audio capture. Detects WASAPI loopback devices automatically."""
        is_loopback = False
        if device_index is not None:
            try:
                dev_name = _sd.query_devices(device_index)['name'].lower()
                is_loopback = any(x in dev_name for x in
                                  ['loopback', 'cable', 'virtual', 'voicemeeter'])
            except Exception:
                pass

        extra = None
        if is_loopback:
            try:
                extra = _sd.WasapiSettings(exclusive=False)
            except Exception:
                pass

        try:
            kwargs = dict(
                samplerate=self.rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk,
                device=device_index,
                callback=self._callback,
            )
            if extra is not None:
                kwargs['extra_settings'] = extra
            self.stream = _sd.InputStream(**kwargs)
            self.running = True
            self.stream.start()
            return True
        except Exception as e:
            print(f"Audio init error: {e}")
            return False

    def analyze_chunk(self):
        """Pull one chunk from queue, run spectral analysis. Returns RTAudioStats or None."""
        try:
            audio = self._queue.get_nowait()
        except queue.Empty:
            return None

        self._chunk_count += 1

        # RMS with exponential smoothing
        rms = float(np.sqrt(np.mean(audio ** 2)))
        smoothed = self._last_rms * 0.7 + rms * 0.3
        self._last_rms = smoothed
        self._rms_history.append(smoothed)
        self._trend_history.append(smoothed)
        rms_mean = float(np.mean(self._rms_history)) if self._rms_history else 1e-6

        # Auto-calibrate noise floor during the first CALIBRATION_CHUNKS chunks.
        if self._chunk_count <= self.CALIBRATION_CHUNKS:
            self._calibration_buf.append(rms)
            if self._chunk_count == self.CALIBRATION_CHUNKS:
                measured = float(np.percentile(list(self._calibration_buf), 95))
                self._gate = max(measured * 4.0, self.NOISE_FLOOR)
                self._calibration_done = True

        # FFT spectral analysis
        window = np.hanning(len(audio))
        fft = np.abs(np.fft.rfft(audio * window))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.rate)

        def _band(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.sqrt(np.mean(fft[mask] ** 2))) if mask.any() else 0.0

        bass   = _band(20,   300)
        mid    = _band(300,  3000)
        treble = _band(3000, 16000)

        # Spectral flatness
        eps = 1e-10
        flat = float(np.exp(np.mean(np.log(fft + eps))) / (np.mean(fft) + eps))
        self._flat_history.append(flat)
        flat_mean = float(np.mean(self._flat_history))
        eg = self.effective_gate
        is_noisy = self._calibration_done and (smoothed > eg) and (flat > flat_mean * 1.5)

        # Beat detection
        current_time = time.time()
        beat = False
        if (self._calibration_done and smoothed > eg * 2
                and rms_mean > eg and (smoothed / rms_mean) > self.BEAT_RATIO):
            if current_time - self._last_beat_time > self.BEAT_COOLDOWN:
                beat = True
                self._last_beat_time = current_time

        # Trend slope (BUILD / DROP detection)
        trend_slope = 0.0
        if len(self._trend_history) >= 5:
            xs = np.arange(len(self._trend_history), dtype=float)
            trend_slope = float(np.polyfit(xs, list(self._trend_history), 1)[0])

        return RTAudioStats(
            rms=smoothed, beat=beat,
            bass=bass, mid=mid, treble=treble,
            flatness=flat, trend_slope=trend_slope,
            rms_mean=rms_mean, is_noisy=is_noisy,
        )

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

    def get_audio_devices(self):
        """Return list of (index, name) for all input-capable devices."""
        devices = []
        try:
            for i, dev in enumerate(_sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    devices.append((i, dev['name']))
        except Exception:
            pass
        return devices


def make_segment_from_stats(stats: RTAudioStats, prev_rms: float, t: float,
                             gate: float = AudioAnalyzer.NOISE_FLOOR) -> Segment:
    """Convert RTAudioStats to a Segment compatible with effects.py."""
    if stats.rms < gate:
        return Segment(
            t_start=t, t_end=t + 0.023, duration=0.023,
            type=SegmentType.SILENCE, intensity=0.0,
            rms=stats.rms, flatness=stats.flatness,
            rms_change=stats.rms - prev_rms,
        )

    rms_mean = max(stats.rms_mean, 1e-6)
    trend_thresh = rms_mean * 0.07

    if stats.trend_slope > trend_thresh:
        seg_type = SegmentType.BUILD
    elif stats.trend_slope < -trend_thresh and stats.rms > rms_mean:
        seg_type = SegmentType.DROP
    elif stats.is_noisy:
        seg_type = SegmentType.NOISE
    elif stats.beat and stats.bass > stats.mid and stats.bass > stats.treble:
        seg_type = SegmentType.IMPACT
    elif stats.rms > rms_mean * 1.2:
        seg_type = SegmentType.SUSTAIN
    else:
        seg_type = SegmentType.SILENCE

    intensity = float(np.clip(stats.rms / (rms_mean * 2.0), 0.0, 1.0))

    return Segment(
        t_start=t, t_end=t + 0.023, duration=0.023,
        type=seg_type, intensity=intensity,
        rms=stats.rms, flatness=stats.flatness,
        rms_change=stats.rms - prev_rms,
    )
