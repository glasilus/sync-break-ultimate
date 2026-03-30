"""Complete audio-reactive effects library."""
from abc import ABC, abstractmethod
import random
import numpy as np
import cv2
from analyzer import Segment, SegmentType
from typing import List
from opensimplex import noise2


# ── BaseEffect (Task 5) ──────────────────────────────────────────────

class BaseEffect(ABC):
    trigger_types: List[SegmentType] = list(SegmentType)

    def __init__(self, enabled=True, chance=1.0, intensity_min=0.0, intensity_max=1.0):
        self.enabled = enabled
        self.chance = chance
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def apply(self, frame: np.ndarray, seg: Segment, draft: bool) -> np.ndarray:
        if not self.enabled:
            return frame
        if seg.type not in self.trigger_types:
            return frame
        if random.random() > self.chance:
            return frame
        return self._apply(frame, seg, draft)

    def scaled_intensity(self, seg: Segment) -> float:
        return self.intensity_min + seg.intensity * (self.intensity_max - self.intensity_min)

    @abstractmethod
    def _apply(self, frame: np.ndarray, seg: Segment, draft: bool) -> np.ndarray: ...


def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame, 0, 255).astype(np.uint8)


def _reseg(seg: Segment, intensity: float) -> Segment:
    return Segment(seg.t_start, seg.t_end, seg.duration, seg.type, intensity,
                   seg.rms, seg.flatness, seg.rms_change)


# ── Task 6 ────────────────────────────────────────────────────────────

class FlashEffect(BaseEffect):
    trigger_types = [SegmentType.DROP, SegmentType.IMPACT]

    def _apply(self, frame, seg, draft):
        alpha = 0.6 + self.scaled_intensity(seg) * 0.4
        flash = np.full_like(frame, 255 if random.random() > 0.5 else 0)
        result = cv2.addWeighted(frame, 1.0 - alpha, flash, alpha, 0)
        return _ensure_uint8(result)


class GhostTrailsEffect(BaseEffect):
    trigger_types = [SegmentType.SUSTAIN, SegmentType.BUILD]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.last_frame = None

    def _apply(self, frame, seg, draft):
        alpha = self.scaled_intensity(seg)
        if self.last_frame is not None and self.last_frame.shape == frame.shape:
            result = cv2.addWeighted(frame, 1.0 - alpha, self.last_frame, alpha, 0)
        else:
            result = frame.copy()
        self.last_frame = frame.copy()
        return _ensure_uint8(result)


# ── Task 7: PixelSortEffect ──────────────────────────────────────────

class PixelSortEffect(BaseEffect):
    trigger_types = [SegmentType.NOISE, SegmentType.IMPACT, SegmentType.DROP]

    def __init__(self, sort_axis='luminance', **kw):
        super().__init__(**kw)
        self.sort_axis = sort_axis

    def _apply(self, frame, seg, draft):
        result = frame.copy()
        h, w = result.shape[:2]
        intensity = self.scaled_intensity(seg)
        strip_h = max(1, int(h * (0.05 + intensity * 0.4)))
        n_strips = 1 if draft else max(1, int(intensity * 8))

        if self.sort_axis == 'hue':
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            key_idx = 0
        elif self.sort_axis == 'saturation':
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            key_idx = 1
        else:
            hsv = None
            key_idx = None  # luminance: use true grayscale

        for _ in range(n_strips):
            y = random.randint(0, max(0, h - strip_h))
            strip = result[y:y + strip_h]
            if key_idx is not None:
                key_strip = hsv[y:y + strip_h, :, key_idx]
                col_means = key_strip.mean(axis=0)
            else:
                gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
                col_means = gray.mean(axis=0)
            order = np.argsort(col_means)
            result[y:y + strip_h] = strip[:, order]

        return _ensure_uint8(result)


# ── Task 8: DatamoshEffect ───────────────────────────────────────────

class DatamoshEffect(BaseEffect):
    trigger_types = [SegmentType.NOISE, SegmentType.SUSTAIN, SegmentType.IMPACT, SegmentType.DROP]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.prev_frame = None

    def apply(self, frame: np.ndarray, seg: 'Segment', draft: bool) -> np.ndarray:
        """Override to always update prev_frame, even when effect doesn't fire."""
        should_fire = (
            self.enabled and
            seg.type in self.trigger_types and
            random.random() <= self.chance
        )
        if self.prev_frame is None or self.prev_frame.shape != frame.shape:
            self.prev_frame = frame.copy()
            return frame
        if not should_fire:
            self.prev_frame = frame.copy()
            return frame
        result = self._apply(frame, seg, draft)
        return result

    def _apply(self, frame, seg, draft):
        gray_cur = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)
        intensity = self.scaled_intensity(seg)
        # Amplify so the effect is clearly visible (1.0 = 5x flow, minimum 2x)
        flow_mul = 2.0 + intensity * 5.0

        try:
            preset = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST if draft else cv2.DISOPTICAL_FLOW_PRESET_FAST
            dis = cv2.DISOpticalFlow_create(preset)
            flow = dis.calc(gray_prev, gray_cur, None)
        except AttributeError:
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_cur, None, 0.5, 2 if draft else 3, 15, 3, 5, 1.2, 0)

        h, w = frame.shape[:2]
        flow_scaled = flow * flow_mul
        map_x = np.float32(np.tile(np.arange(w), (h, 1)) + flow_scaled[..., 0])
        map_y = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w)) + flow_scaled[..., 1])
        result = cv2.remap(self.prev_frame, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        self.prev_frame = frame.copy()
        return _ensure_uint8(result)


# ── Task 9: ASCIIEffect ──────────────────────────────────────────────

class ASCIIEffect(BaseEffect):
    """
    Full-frame ASCII art conversion via PIL.

    color_mode:
      'fixed'    — all chars drawn in fg_color on bg_color background
      'original' — each char colored from the average RGB of its source block
      'inverted' — same as original but colors inverted
    blend:
      0.0 = pure ASCII output
      1.0 = original frame only (ASCII invisible)
      values in between composite ASCII over original
    """
    trigger_types = [SegmentType.SUSTAIN, SegmentType.SILENCE, SegmentType.BUILD]

    # dark → light (dense characters = bright regions)
    DEFAULT_CHARSET = '@#%S?*+;:,. '

    def __init__(self, char_size=10, charset=None,
                 fg_color=(0, 255, 0), bg_color=(0, 0, 0),
                 blend=0.0, color_mode='fixed', **kw):
        super().__init__(**kw)
        self.char_size  = char_size
        self.charset    = charset or self.DEFAULT_CHARSET
        self.fg_color   = tuple(fg_color)
        self.bg_color   = tuple(bg_color)
        self.blend      = blend
        self.color_mode = color_mode   # 'fixed' | 'original' | 'inverted'
        self._pil_font  = None
        self._font_size = None

    def _get_pil_font(self, size):
        """Return a PIL ImageFont. Tries Courier/monospace, falls back to default."""
        if self._pil_font is not None and self._font_size == size:
            return self._pil_font
        from PIL import ImageFont
        candidates = [
            'cour.ttf', 'courbd.ttf',         # Windows Courier New
            'DejaVuSansMono.ttf',              # Linux
            'Menlo.ttc', 'Monaco.ttf',         # macOS
            'LiberationMono-Regular.ttf',
        ]
        font = None
        for name in candidates:
            try:
                font = ImageFont.truetype(name, size)
                break
            except (OSError, IOError):
                continue
        if font is None:
            font = ImageFont.load_default()
        self._pil_font  = font
        self._font_size = size
        return font

    def _apply(self, frame, seg, draft):
        from PIL import Image as PILImage, ImageDraw

        h, w = frame.shape[:2]
        # In draft mode, use larger blocks (faster + stylistically distinct)
        cell_h = (self.char_size * 2) if draft else self.char_size
        cell_h = max(4, cell_h)
        # Monospace chars are roughly half as wide as tall
        cell_w = max(2, cell_h // 2)

        cols = max(1, w // cell_w)
        rows = max(1, h // cell_h)

        charset = self.charset
        n = len(charset)

        # Compute grayscale for brightness mapping
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # PIL canvas filled with background
        canvas = PILImage.new('RGB', (w, h), self.bg_color)
        draw   = ImageDraw.Draw(canvas)
        font   = self._get_pil_font(cell_h)

        for r in range(rows):
            for c in range(cols):
                y0 = r * cell_h
                x0 = c * cell_w
                y1 = min(y0 + cell_h, h)
                x1 = min(x0 + cell_w, w)

                brightness = int(gray[y0:y1, x0:x1].mean())
                char_idx   = min(int(brightness / 256 * n), n - 1)
                ch         = charset[char_idx]

                if ch == ' ':
                    continue  # background already filled

                if self.color_mode == 'original':
                    cell_rgb = frame[y0:y1, x0:x1]
                    color = (
                        int(cell_rgb[:, :, 0].mean()),
                        int(cell_rgb[:, :, 1].mean()),
                        int(cell_rgb[:, :, 2].mean()),
                    )
                elif self.color_mode == 'inverted':
                    cell_rgb = frame[y0:y1, x0:x1]
                    color = (
                        255 - int(cell_rgb[:, :, 0].mean()),
                        255 - int(cell_rgb[:, :, 1].mean()),
                        255 - int(cell_rgb[:, :, 2].mean()),
                    )
                else:
                    color = self.fg_color

                draw.text((x0, y0), ch, fill=color, font=font)

        out = np.array(canvas)

        if self.blend > 0:
            out = cv2.addWeighted(out, 1.0 - self.blend,
                                  frame, self.blend, 0)

        return _ensure_uint8(out)


# ── Task 10: Codec Breakers ──────────────────────────────────────────

class RGBShiftEffect(BaseEffect):
    trigger_types = [SegmentType.IMPACT, SegmentType.BUILD,
                     SegmentType.NOISE, SegmentType.DROP]

    def _apply(self, frame, seg, draft):
        shift = int(self.scaled_intensity(seg) * 20)
        result = frame.copy()
        result[:, :, 0] = np.roll(frame[:, :, 0], shift, axis=1)   # R right
        result[:, :, 2] = np.roll(frame[:, :, 2], -shift, axis=1)  # B left
        return result


class BlockGlitchEffect(BaseEffect):
    trigger_types = [SegmentType.IMPACT, SegmentType.DROP, SegmentType.NOISE]

    def __init__(self, block_size=16, **kw):
        super().__init__(**kw)
        self.block_size = block_size

    def _apply(self, frame, seg, draft):
        result = frame.copy()
        h, w = result.shape[:2]
        n_blocks = int(self.scaled_intensity(seg) * 20)
        bs = self.block_size
        for _ in range(n_blocks):
            y = random.randint(0, max(0, h - bs))
            x = random.randint(0, max(0, w - bs))
            if random.random() < 0.5:
                sy = random.randint(0, max(0, h - bs))
                sx = random.randint(0, max(0, w - bs))
                result[y:y + bs, x:x + bs] = frame[sy:sy + bs, sx:sx + bs]
            else:
                result[y:y + bs, x:x + bs] = random.randint(0, 255)
        return result


class PixelDriftEffect(BaseEffect):
    trigger_types = [SegmentType.NOISE, SegmentType.IMPACT]

    def _apply(self, frame, seg, draft):
        result = frame.copy()
        h, w = result.shape[:2]
        max_shift = int(self.scaled_intensity(seg) * 30)
        step = 4 if draft else 1
        for row in range(0, h, step):
            n = noise2(float(row) * 0.1, seg.intensity * 100.0)
            shift = int(n * max_shift)
            result[row] = np.roll(frame[row], shift, axis=0)
        return result


class ScanLinesEffect(BaseEffect):
    trigger_types = [SegmentType.SUSTAIN, SegmentType.NOISE]

    def _apply(self, frame, seg, draft):
        result = frame.astype(np.float32)
        intensity = self.scaled_intensity(seg)
        n = max(2, int(8 - intensity * 6))
        darkness = 0.3 + intensity * 0.5
        result[::n] = result[::n] * (1.0 - darkness)
        return _ensure_uint8(result)


class BitcrushEffect(BaseEffect):
    trigger_types = list(SegmentType)

    def _apply(self, frame, seg, draft):
        intensity = self.scaled_intensity(seg)
        bits = max(1, int(7 - intensity * 5))
        shift = 8 - bits
        return ((frame >> shift) << shift).astype(np.uint8)


class ColorBleedEffect(BaseEffect):
    trigger_types = [SegmentType.NOISE, SegmentType.SUSTAIN]

    def _apply(self, frame, seg, draft):
        result = frame.copy()
        intensity = self.scaled_intensity(seg)
        channel = random.randint(0, 2)
        kernel_w = max(3, int(intensity * 40))
        if kernel_w % 2 == 0:
            kernel_w += 1
        result[:, :, channel] = cv2.blur(result[:, :, channel], (kernel_w, 1))
        return result


class FreezeCorruptEffect(BaseEffect):
    trigger_types = [SegmentType.DROP]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._held = None
        self._hold_count = 0
        self._glitch = BlockGlitchEffect(enabled=True, chance=1.0)

    def _apply(self, frame, seg, draft):
        intensity = self.scaled_intensity(seg)
        hold_frames = max(1, int(intensity * 6))
        if self._held is None or self._hold_count >= hold_frames:
            self._held = frame.copy()
            self._hold_count = 0
        self._hold_count += 1
        return self._glitch._apply(self._held, seg, draft)


class NegativeEffect(BaseEffect):
    trigger_types = [SegmentType.IMPACT, SegmentType.DROP, SegmentType.NOISE]

    def _apply(self, frame, seg, draft):
        return (255 - frame).astype(np.uint8)


# ── Task 11: Degradation Effects ─────────────────────────────────────

class JPEGCrushEffect(BaseEffect):
    trigger_types = [SegmentType.IMPACT, SegmentType.NOISE]

    def _apply(self, frame, seg, draft):
        intensity = self.scaled_intensity(seg)
        quality = max(1, int(40 - intensity * 38))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        result = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return _ensure_uint8(result)


class FisheyeEffect(BaseEffect):
    trigger_types = [SegmentType.BUILD, SegmentType.SUSTAIN]

    def _apply(self, frame, seg, draft):
        h, w = frame.shape[:2]
        intensity = self.scaled_intensity(seg)
        strength = intensity * 0.8
        K = np.array([[w, 0, w / 2.0],
                       [0, w, h / 2.0],
                       [0, 0, 1.0]], dtype=np.float64)
        D = np.array([[strength], [strength * 0.3], [0.0], [0.0]], dtype=np.float64)
        try:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), K, (w, h), cv2.CV_32FC1)
            result = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        except cv2.error:
            # Fallback: simple barrel distortion via remap
            cx, cy = w / 2.0, h / 2.0
            xs = np.linspace(-1, 1, w)
            ys = np.linspace(-1, 1, h)
            xg, yg = np.meshgrid(xs, ys)
            r2 = xg ** 2 + yg ** 2
            factor = 1.0 + strength * r2
            map_x = ((xg * factor + 1) / 2 * w).astype(np.float32)
            map_y = ((yg * factor + 1) / 2 * h).astype(np.float32)
            result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        return _ensure_uint8(result)


class VHSTrackingEffect(BaseEffect):
    trigger_types = [SegmentType.NOISE, SegmentType.DROP]

    def _apply(self, frame, seg, draft):
        result = frame.copy()
        h, w = result.shape[:2]
        intensity = self.scaled_intensity(seg)
        n_strips = max(1, int(intensity * 8))
        noise_amp = int(intensity * 20)
        strip_h = h // max(1, n_strips)
        for i in range(n_strips):
            y = random.randint(0, max(0, h - strip_h))
            shift = int(noise2(float(i) * 0.5, intensity * 50.0) * noise_amp)
            result[y:y + strip_h] = np.roll(result[y:y + strip_h], shift, axis=1)
            # luminance noise
            noise_val = np.clip(
                np.array([noise2(float(x) * 0.1, float(y) * 0.1) * noise_amp
                          for x in range(w)]),
                -30, 30).astype(np.int16)
            for row in range(y, min(y + strip_h, h)):
                result[row] = np.clip(
                    result[row].astype(np.int16) + noise_val.reshape(-1, 1), 0, 255
                ).astype(np.uint8)
        return result


class InterlaceEffect(BaseEffect):
    trigger_types = [SegmentType.SUSTAIN]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.prev_frame = None

    def _apply(self, frame, seg, draft):
        result = frame.copy()
        if self.prev_frame is not None and self.prev_frame.shape == frame.shape:
            result[1::2] = self.prev_frame[1::2]
        self.prev_frame = frame.copy()
        return result


class BadSignalEffect(BaseEffect):
    trigger_types = [SegmentType.DROP, SegmentType.NOISE]

    def _apply(self, frame, seg, draft):
        result = frame.copy()
        h, w = result.shape[:2]
        intensity = self.scaled_intensity(seg)
        # Vertical noise bars
        n_bars = int(intensity * 5)
        for _ in range(n_bars):
            x = random.randint(0, w - 1)
            bw = random.randint(1, 4)
            val = random.randint(0, 255)
            result[:, x:min(x + bw, w)] = val
        # Row shifts
        n_shift = int(intensity * h * 0.1)
        for _ in range(n_shift):
            row = random.randint(0, h - 1)
            shift = random.randint(-20, 20)
            result[row] = np.roll(result[row], shift, axis=0)
        return result


class DitheringEffect(BaseEffect):
    trigger_types = [SegmentType.SILENCE, SegmentType.SUSTAIN]

    BAYER_4X4 = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.float32) / 16.0

    def _apply(self, frame, seg, draft):
        intensity = self.scaled_intensity(seg)
        levels = max(2, int(16 - intensity * 12))
        h, w = frame.shape[:2]
        # Tile bayer matrix
        tile_r = (h + 3) // 4
        tile_c = (w + 3) // 4
        bayer = np.tile(self.BAYER_4X4, (tile_r, tile_c))[:h, :w]
        bayer3 = np.stack([bayer] * 3, axis=-1)
        # Quantize
        normalized = frame.astype(np.float32) / 255.0
        step = 1.0 / levels
        dithered = normalized + (bayer3 - 0.5) * step
        quantized = np.floor(dithered * levels) / levels
        return _ensure_uint8(quantized * 255.0)


class ZoomGlitchEffect(BaseEffect):
    trigger_types = [SegmentType.IMPACT, SegmentType.DROP]

    def _apply(self, frame, seg, draft):
        h, w = frame.shape[:2]
        intensity = self.scaled_intensity(seg)
        zoom = 1.0 + intensity * 0.4
        cw, ch = int(w / zoom), int(h / zoom)
        x1 = (w - cw) // 2
        y1 = (h - ch) // 2
        cropped = frame[y1:y1 + ch, x1:x1 + cw]
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)
        return _ensure_uint8(result)


# ── Task 12: Complex Effects ─────────────────────────────────────────

class FeedbackLoopEffect(BaseEffect):
    trigger_types = [SegmentType.SUSTAIN, SegmentType.BUILD]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.accumulated = None

    def _apply(self, frame, seg, draft):
        intensity = self.scaled_intensity(seg)
        weight = intensity * 0.7
        if seg.type == SegmentType.IMPACT:
            self.accumulated = None
        if self.accumulated is None or self.accumulated.shape != frame.shape:
            self.accumulated = frame.astype(np.float32)
            return frame.copy()
        self.accumulated = frame.astype(np.float32) * (1 - weight) + self.accumulated * weight
        return _ensure_uint8(self.accumulated)


class PhaseShiftEffect(BaseEffect):
    trigger_types = [SegmentType.NOISE, SegmentType.DROP]

    def _apply(self, frame, seg, draft):
        result = frame.copy()
        h, w = result.shape[:2]
        intensity = self.scaled_intensity(seg)
        band_h = max(4, int(h * 0.05))
        shift = int(intensity * w * 0.2)
        for y in range(0, h, band_h):
            band_idx = y // band_h
            s = shift if band_idx % 2 == 0 else -shift
            end = min(y + band_h, h)
            result[y:end] = np.roll(result[y:end], s, axis=1)
        return result


class MosaicPulseEffect(BaseEffect):
    trigger_types = [SegmentType.IMPACT, SegmentType.BUILD]

    def _apply(self, frame, seg, draft):
        h, w = frame.shape[:2]
        intensity = self.scaled_intensity(seg)
        block = max(2, int(4 + intensity * 40))
        small = cv2.resize(frame, (max(1, w // block), max(1, h // block)),
                           interpolation=cv2.INTER_NEAREST)
        result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return _ensure_uint8(result)


class EchoCompoundEffect(BaseEffect):
    trigger_types = [SegmentType.SUSTAIN, SegmentType.BUILD]

    def __init__(self, echo_n=8, **kw):
        super().__init__(**kw)
        self.echo_n = echo_n
        self.history = []

    def _apply(self, frame, seg, draft):
        self.history.append(frame.copy())
        max_len = self.echo_n * 2 + 1
        if len(self.history) > max_len:
            self.history = self.history[-max_len:]

        result = frame.astype(np.float32) * 0.5
        n = self.echo_n
        # blend frame N-ago
        if len(self.history) > n:
            past1 = self.history[-(n + 1)]
            result += past1.astype(np.float32) * 0.3
        else:
            result += frame.astype(np.float32) * 0.3
        # blend frame 2N-ago with hue shift
        if len(self.history) > 2 * n:
            past2 = self.history[-(2 * n + 1)]
            # hue shift by 60
            hsv = cv2.cvtColor(past2, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + 30) % 180  # 60° = 30 in OpenCV
            past2_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            result += past2_shifted.astype(np.float32) * 0.2
        else:
            result += frame.astype(np.float32) * 0.2
        return _ensure_uint8(result)


class KaliMirrorEffect(BaseEffect):
    trigger_types = [SegmentType.BUILD, SegmentType.SUSTAIN]

    def _apply(self, frame, seg, draft):
        h, w = frame.shape[:2]
        intensity = self.scaled_intensity(seg)
        # Mirror horizontally
        mirrored = np.hstack([frame, frame[:, ::-1]])
        # Vstack with inverted
        full = np.vstack([mirrored, 255 - mirrored])
        # Rotate
        angle = intensity * 180.0
        fh, fw = full.shape[:2]
        M = cv2.getRotationMatrix2D((fw / 2, fh / 2), angle, 1.0)
        rotated = cv2.warpAffine(full, M, (fw, fh), borderMode=cv2.BORDER_REFLECT)
        # Crop center back to original size
        cy, cx = fh // 2, fw // 2
        result = rotated[cy - h // 2:cy - h // 2 + h, cx - w // 2:cx - w // 2 + w]
        return _ensure_uint8(result)


class GlitchCascadeEffect(BaseEffect):
    trigger_types = [SegmentType.IMPACT, SegmentType.DROP, SegmentType.NOISE]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.pool = [
            RGBShiftEffect(enabled=True, chance=1.0),
            BlockGlitchEffect(enabled=True, chance=1.0),
            PixelDriftEffect(enabled=True, chance=1.0),
            BitcrushEffect(enabled=True, chance=1.0),
        ]

    def _apply(self, frame, seg, draft):
        intensity = self.scaled_intensity(seg)
        n = max(1, int(intensity * len(self.pool)))
        chosen = random.sample(self.pool, min(n, len(self.pool)))
        result = frame.copy()
        for fx in chosen:
            fx.trigger_types = list(SegmentType)
            result = fx._apply(result, seg, draft)
        return _ensure_uint8(result)


# ── Task 13: MysterySection ──────────────────────────────────────────

class MysterySection:
    def __init__(self):
        self.VESSEL = 0.0
        self.ENTROPY_7 = 0.0
        self.DELTA_OMEGA = 0.0
        self.STATIC_MIND = 0.0
        self.RESONANCE = 0.0
        self.COLLAPSE = 0.0
        self.ZERO = 0.0
        self.FLESH_K = 0.0
        self.DOT = 0.0
        self._feedback = FeedbackLoopEffect(enabled=True, chance=1.0)
        self._ghost = GhostTrailsEffect(enabled=True, chance=1.0)
        self._scanlines = ScanLinesEffect(enabled=True, chance=1.0)
        self._colorbleed = ColorBleedEffect(enabled=True, chance=1.0)
        self._frame_buffer = []

    def apply(self, frame, seg, draft):
        result = frame.copy()
        if self.ZERO > 0:
            random.seed(int(self.ZERO * 9999))
        if self.FLESH_K > 0.33:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2YCrCb)
            if self.FLESH_K > 0.66:
                result = cv2.cvtColor(result, cv2.COLOR_YCrCb2RGB)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
                result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
            else:
                result = cv2.cvtColor(result, cv2.COLOR_YCrCb2RGB)
        if self.VESSEL > 0:
            self._feedback.intensity_max = self.VESSEL
            self._ghost.intensity_max = 1.0 - self.VESSEL
            result = self._feedback._apply(result, seg, draft)
            result = self._ghost._apply(result, seg, draft)
        if self.STATIC_MIND > 0:
            scan_seg = _reseg(seg, self.STATIC_MIND)
            bleed_seg = _reseg(seg, 1.0 - self.STATIC_MIND)
            result = self._scanlines._apply(result, scan_seg, draft)
            result = self._colorbleed._apply(result, bleed_seg, draft)
        if self.COLLAPSE > 0:
            h, w = result.shape[:2]
            factor = max(2, int(2 + self.COLLAPSE * 8))
            small = cv2.resize(result, (max(1, w // factor), max(1, h // factor)),
                               interpolation=cv2.INTER_NEAREST)
            result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.ENTROPY_7 > 0:
            drift = PixelDriftEffect(enabled=True, chance=1.0)
            result = drift._apply(result, _reseg(seg, self.ENTROPY_7), draft)
        if self.DOT > 0:
            delay = max(1, int(self.DOT * 3))
            self._frame_buffer.append(result.copy())
            if len(self._frame_buffer) > delay + 1:
                self._frame_buffer.pop(0)
            if len(self._frame_buffer) > delay:
                delayed = self._frame_buffer[0]
                if delayed.shape == result.shape:
                    result = cv2.addWeighted(result, 0.7, delayed, 0.3, 0)
        random.seed()
        return _ensure_uint8(result)

    def get_threshold_shift(self):
        return self.DELTA_OMEGA

    def get_remap_curve(self):
        return self.RESONANCE


# ── Task 14: ChromaKeyEffect + OverlayEffect ─────────────────────────

def _dominant_hue(img_rgb: np.ndarray, rank: int = 0) -> int:
    """Return OpenCV hue (0-179) of the rank-th most frequent colour cluster."""
    rgb3 = img_rgb[:, :, :3]
    hsv = cv2.cvtColor(rgb3, cv2.COLOR_RGB2HSV)
    h_ch = hsv[:, :, 0].flatten()
    s_ch = hsv[:, :, 1].flatten()
    # prefer saturated pixels to avoid greys/blacks skewing the result
    saturated = h_ch[s_ch > 40]
    source = saturated if len(saturated) > 200 else h_ch
    bins, _ = np.histogram(source, bins=18, range=(0, 180))
    order = np.argsort(bins)[::-1]
    idx = int(order[rank]) if rank < len(order) else int(order[0])
    return idx * 10 + 5  # centre of the 10-degree bin


class ChromaKeyEffect:
    def __init__(self, key_color=(0, 255, 0), tolerance=30, edge_softness=5):
        self.key_color = key_color
        self.tolerance = tolerance
        self.edge_softness = edge_softness

    @classmethod
    def from_frame(cls, img_rgb: np.ndarray, rank: int = 0,
                   tolerance: int = 30, edge_softness: int = 5) -> 'ChromaKeyEffect':
        """Build a ChromaKeyEffect targeting the rank-th most frequent hue."""
        hue = _dominant_hue(img_rgb, rank)
        key_hsv = np.uint8([[[hue, 200, 200]]])
        key_rgb = cv2.cvtColor(key_hsv, cv2.COLOR_HSV2RGB)[0, 0]
        return cls(key_color=tuple(int(x) for x in key_rgb),
                   tolerance=tolerance, edge_softness=edge_softness)

    def get_mask(self, frame):
        """Return uint8 mask: 0 = keyed out, 255 = keep."""
        hsv = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2HSV)
        key_hsv = cv2.cvtColor(
            np.uint8([[list(self.key_color)]]), cv2.COLOR_RGB2HSV)[0, 0]
        h_center = int(key_hsv[0])
        lower = np.array([max(0, h_center - self.tolerance), 40, 40], dtype=np.uint8)
        upper = np.array([min(179, h_center + self.tolerance), 255, 255], dtype=np.uint8)
        keyed = cv2.inRange(hsv, lower, upper)
        mask = 255 - keyed  # 0 = keyed out, 255 = keep
        if self.edge_softness > 1:
            ks = self.edge_softness | 1
            mask = cv2.GaussianBlur(mask, (ks, ks), 0)
        return mask

    def apply_to_frame(self, frame, replacement=None):
        mask = self.get_mask(frame)
        mask3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        if replacement is None:
            replacement = np.zeros_like(frame)
        result = (frame.astype(np.float32) * mask3 +
                  replacement.astype(np.float32) * (1.0 - mask3))
        return _ensure_uint8(result)


class OverlayEffect(BaseEffect):
    """
    Composite image overlays at a scaled size and random/fixed position.

    position:  'random'        — position chosen once per segment, held for all frames
               'center'        — centred on frame
               'random_corner' — one of four corners, chosen once per segment
    scale:     fraction of frame short-side used as overlay height (0.05–1.0).
               Intensity modulates scale between scale_min and scale.
    blend_mode: 'normal' | 'screen' | 'multiply'

    The overlay decision (show/hide, frame choice, position, scale) is made
    ONCE per segment — matching the original moviepy behaviour where an overlay
    was applied to the whole clip for its full duration.  This prevents the
    per-frame flicker that occurred when the dice was re-rolled every frame.
    """
    trigger_types = list(SegmentType)

    def __init__(self, overlay_frames=None, chroma_key=None,
                 chroma_mode='none', chroma_tolerance=30, chroma_softness=5,
                 opacity=0.85, blend_mode='screen',
                 scale=0.4, scale_min=0.15,
                 position='random', **kw):
        super().__init__(**kw)
        self.overlay_frames    = overlay_frames or []
        self.chroma_key        = chroma_key       # ChromaKeyEffect for 'manual' mode
        self.chroma_mode       = chroma_mode      # 'none'|'dominant'|'secondary'|'manual'
        self.chroma_tolerance  = chroma_tolerance
        self.chroma_softness   = chroma_softness
        self.opacity           = opacity
        self.blend_mode        = blend_mode
        self.scale             = scale
        self.scale_min         = scale_min
        self.position          = position
        self._idx              = 0
        self._corner           = 0
        self._ck_cache: dict   = {}               # overlay idx → ChromaKeyEffect

        # Per-segment state — decided once when t_start changes
        self._seg_t_start: float = -1.0
        self._seg_active:  bool  = False
        self._seg_ov_idx:  int   = 0
        self._seg_x0:      int   = 0
        self._seg_y0:      int   = 0
        self._seg_tw:      int   = 0
        self._seg_th:      int   = 0

    # ---- blend helper -------------------------------------------------------
    def _blend(self, base_f, ov_f, alpha):
        """Blend ov_f over base_f with given per-pixel alpha (0–1 float)."""
        if self.blend_mode == 'screen':
            blended = 255.0 - (255.0 - base_f) * (255.0 - ov_f) / 255.0
        elif self.blend_mode == 'multiply':
            blended = base_f * ov_f / 255.0
        else:                          # normal
            blended = ov_f
        return base_f * (1.0 - alpha) + blended * alpha

    # Override apply() so the per-frame chance roll in BaseEffect is skipped —
    # the chance is consumed once per segment inside _apply() instead.
    def apply(self, frame, seg, draft):
        if not self.enabled:
            return frame
        if seg.type not in self.trigger_types:
            return frame
        return self._apply(frame, seg, draft)

    # ---- public composite (kept for backward compat) ------------------------
    def composite(self, base, overlay, opacity):
        bf = base.astype(np.float32)
        of = overlay.astype(np.float32)
        return _ensure_uint8(self._blend(bf, of, opacity))

    # ---- main apply ---------------------------------------------------------
    def _apply(self, frame, seg, draft):
        if not self.overlay_frames:
            return frame

        h, w = frame.shape[:2]

        # --- per-segment decision (original behaviour: decide once per clip) -
        if seg.t_start != self._seg_t_start:
            self._seg_t_start = seg.t_start
            # chance roll happens here, not in BaseEffect.apply, so we honour
            # the per-segment contract even when called from apply().
            self._seg_active = random.random() <= self.chance

            if self._seg_active:
                # Pick overlay frame and compute geometry once
                self._seg_ov_idx = self._idx % len(self.overlay_frames)
                self._idx += 1

                intensity = self.scaled_intensity(seg)
                cur_scale = self.scale_min + (self.scale - self.scale_min) * intensity
                cur_scale = max(0.05, min(1.0, cur_scale))

                ov_src = self.overlay_frames[self._seg_ov_idx]
                ov_h_src, ov_w_src = ov_src.shape[:2]
                th = max(4, int(h * cur_scale))
                tw = max(4, int(th * ov_w_src / max(ov_h_src, 1)))
                tw = min(tw, w)
                th = min(th, h)

                if self.position == 'center':
                    x0 = (w - tw) // 2
                    y0 = (h - th) // 2
                elif self.position == 'random_corner':
                    corners = [(0, 0), (w - tw, 0),
                               (0, h - th), (w - tw, h - th)]
                    x0, y0 = corners[self._corner % 4]
                    self._corner += 1
                else:  # 'random'
                    x0 = random.randint(0, max(0, w - tw))
                    y0 = random.randint(0, max(0, h - th))

                self._seg_x0 = max(0, min(x0, w - tw))
                self._seg_y0 = max(0, min(y0, h - th))
                self._seg_tw = tw
                self._seg_th = th

        if not self._seg_active:
            return frame

        # --- composite using the geometry decided at segment start -----------
        tw, th = self._seg_tw, self._seg_th
        x0, y0 = self._seg_x0, self._seg_y0

        ov_src = self.overlay_frames[self._seg_ov_idx]
        interp = cv2.INTER_AREA if draft else cv2.INTER_LINEAR
        ov = cv2.resize(ov_src, (tw, th), interpolation=interp)

        intensity = self.scaled_intensity(seg)
        alpha = min(1.0, self.opacity * (0.4 + intensity * 0.6))

        result = frame.copy()
        roi    = result[y0:y0 + th, x0:x0 + tw].astype(np.float32)
        ov_f   = ov[:, :, :3].astype(np.float32)

        # --- chroma key: build or retrieve cached ChromaKeyEffect ----------
        ck = None
        if self.chroma_mode == 'dominant':
            ck = self._ck_cache.get(self._seg_ov_idx)
            if ck is None:
                ck = ChromaKeyEffect.from_frame(ov_src, rank=0,
                                                tolerance=self.chroma_tolerance,
                                                edge_softness=self.chroma_softness)
                self._ck_cache[self._seg_ov_idx] = ck
        elif self.chroma_mode == 'secondary':
            ck = self._ck_cache.get(self._seg_ov_idx)
            if ck is None:
                ck = ChromaKeyEffect.from_frame(ov_src, rank=1,
                                                tolerance=self.chroma_tolerance,
                                                edge_softness=self.chroma_softness)
                self._ck_cache[self._seg_ov_idx] = ck
        elif self.chroma_mode == 'manual' and self.chroma_key is not None:
            ck = self.chroma_key

        if ck is not None:
            # Per-pixel alpha: mask (0=keyed, 255=keep) × flat opacity
            mask_src = ck.get_mask(ov_src)
            mask = cv2.resize(mask_src, (tw, th), interpolation=interp)
            per_pixel_alpha = (mask.astype(np.float32) / 255.0) * alpha
            ppa = per_pixel_alpha[:, :, np.newaxis]
            blended = self._blend(roi, ov_f, 1.0)
            blended_roi = _ensure_uint8(roi * (1.0 - ppa) + blended * ppa)
        else:
            blended_roi = _ensure_uint8(self._blend(roi, ov_f, alpha))

        result[y0:y0 + th, x0:x0 + tw] = blended_roi
        return result
