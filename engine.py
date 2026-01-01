import os
import random
import time
import numpy as np
import librosa
import cv2
from typing import List, Dict, Any, Optional, Tuple

from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    ImageClip, CompositeVideoClip, vfx, ColorClip
)
from moviepy.video.fx.colorx import colorx

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

from PIL import Image

# --- КОНСТАНТЫ ---
MIN_SEGMENT_DURATION = 0.04  # ~1 кадр при 24fps.
DEFAULT_FPS = 24

class GhostEffect:
    """
    Класс для реализации эффекта Ghost Trails (Шлейф).
    Оптимизирован для использования float32.
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.last_frame = None

    def apply(self, get_frame, t):
        current_frame = get_frame(t)
        
        # Если это первый кадр
        if self.last_frame is None:
            # Используем float32 для экономии памяти (в 2 раза меньше чем стандартный float64)
            self.last_frame = current_frame.astype(np.float32)
            return current_frame

        # Конвертируем текущий кадр
        curr_float = current_frame.astype(np.float32)
        
        # Смешивание: New = Current * (1-alpha) + Old * alpha
        # Используем cv2.addWeighted для скорости
        blended = cv2.addWeighted(curr_float, 1.0, self.last_frame, self.alpha, 0)
        
        self.last_frame = blended # Обновляем "память"
        
        # Безопасная конвертация обратно в uint8
        return cv2.convertScaleAbs(blended)

class BreakcoreEngine:
    def __init__(self, config: Dict[str, Any], progress_callback=None):
        self.cfg = config
        self.progress_callback = progress_callback
        self.abort = False
        self.scene_cuts: List[float] = []      
        self.scene_buffer: List[float] = []    
        self.target_resolution = (1280, 720) # Значение по умолчанию, обновится при старте
        
        if not os.path.exists(self.cfg['video_path']):
            raise FileNotFoundError(f"Video file not found: {self.cfg['video_path']}")
        if not os.path.exists(self.cfg['audio_path']):
            raise FileNotFoundError(f"Audio file not found: {self.cfg['audio_path']}")

    def log(self, message: str, value: Optional[int] = None):
        print(f"[ENGINE] {message}")
        if self.progress_callback:
            self.progress_callback(message, value)

    # --- 1. АНАЛИЗ ВИДЕО ---
    def detect_scenes(self, video_duration: float):
        if not self.cfg.get('use_scene_detect', True):
            self.log("Scene detection disabled by user.")
            return

        self.log("Starting Smart Scene Detection...")
        video_manager = VideoManager([self.cfg['video_path']])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
        
        try:
            video_manager.set_downscale_factor()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            
            scene_list = scene_manager.get_scene_list()
            self.scene_cuts = [x[0].get_seconds() for x in scene_list]
            self.scene_cuts = [t for t in self.scene_cuts if t < video_duration - 1.0]
            
            self.log(f"Detected {len(self.scene_cuts)} scenes.")
            
            buf_size = int(self.cfg.get('scene_buffer_size', 10))
            self.scene_buffer = self.scene_cuts[:buf_size]
            
        except Exception as e:
            self.log(f"Scene detection warning: {e}. Fallback to random seeking.")
            self.scene_cuts = []
        finally:
            video_manager.release()

    # --- 2. АНАЛИЗ АУДИО ---
    def analyze_audio(self):
        self.log("Computing audio features...")
        y, sr = librosa.load(self.cfg['audio_path'])
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time', backtrack=True)
        rms = librosa.feature.rms(y=y)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        
        return onsets, rms, flatness, sr, y.shape[0] / sr

    # --- 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
    def get_time_index(self, t: float, sr: int, array_len: int, hop_length: int = 512) -> int:
        frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
        return min(frame, array_len - 1)

    def get_source_clip(self, main_video: VideoFileClip, duration: float) -> VideoFileClip:
        chaos = self.cfg.get('chaos_level', 0.5)
        use_scenes = len(self.scene_buffer) > 0
        should_use_scene = use_scenes and (random.random() > chaos * 0.8)
        
        start_t = 0.0
        if should_use_scene:
            start_t = random.choice(self.scene_buffer)
            start_t += random.uniform(0, 1.0)
        else:
            start_t = random.uniform(0, max(0, main_video.duration - duration))

        if start_t + duration > main_video.duration:
            start_t = max(0, main_video.duration - duration - 0.1)
            
        return main_video.subclip(start_t, start_t + duration)

    # --- 4. ЭФФЕКТЫ ---
    def fx_pixel_sort(self, clip: VideoFileClip) -> VideoFileClip:
        intensity = self.cfg.get('fx_psort_int', 0.5)

        def fl(gf, t):
            frame = gf(t)
            img_arr = frame.copy()
            h, w, _ = img_arr.shape
            
            sort_h = int(h * (0.1 + intensity * 0.6))
            y_start = random.randint(0, h - sort_h)
            
            crop = img_arr[y_start:y_start+sort_h, :, :]
            pixels = crop.reshape(-1, 3)
            lum = pixels.sum(axis=1)
            sorted_indices = np.argsort(lum)
            
            sorted_pixels = pixels[sorted_indices]
            sorted_crop = sorted_pixels.reshape(sort_h, w, 3)
            img_arr[y_start:y_start+sort_h, :, :] = sorted_crop
            return img_arr

        return clip.fl(fl)

    def fx_stutter(self, clip: VideoFileClip, count: int) -> VideoFileClip:
        if clip.duration < MIN_SEGMENT_DURATION:
            return clip
            
        segment_len = clip.duration / count
        if segment_len < MIN_SEGMENT_DURATION:
            segment_len = MIN_SEGMENT_DURATION
            count = int(clip.duration / segment_len)
            if count < 1: count = 1

        sub = clip.subclip(0, segment_len)
        # ВАЖНО: При stutter также используем chain или просто склейку
        stuttered = concatenate_videoclips([sub] * count)
        
        if stuttered.duration > clip.duration:
            stuttered = stuttered.subclip(0, clip.duration)
        return stuttered

    def fx_ghost_trails(self, clip: VideoFileClip) -> VideoFileClip:
        opacity = self.cfg.get('fx_ghost_opacity', 0.5)
        effect = GhostEffect(alpha=opacity)
        return clip.fl(effect.apply)

    def fx_flash(self, duration: float) -> VideoFileClip:
        col = 'white' if random.random() > 0.5 else 'black'
        # Создаем клип сразу нужного размера, чтобы не ресайзить лишний раз
        return ColorClip(size=self.target_resolution, color=col, duration=duration).set_opacity(0.8)

    # --- 5. ГЛАВНЫЙ ЦИКЛ (RUN) ---
    def run(self, max_output_duration: Optional[float] = None):
        try:
            self.log("Initializing media...")
            main_audio = AudioFileClip(self.cfg['audio_path'])
            main_video = VideoFileClip(self.cfg['video_path'])
            
            # --- Четкая фиксация разрешения ---
            w, h = main_video.size
            if w > 1920: 
                main_video = main_video.resize(width=1920)
            
            # Сохраняем целевое разрешение в атрибут класса
            self.target_resolution = main_video.size
            self.log(f"Target resolution locked at: {self.target_resolution}")

            overlay_files = []
            if self.cfg.get('overlay_dir') and os.path.exists(self.cfg['overlay_dir']):
                valid_ext = ('.png', '.jpg', '.jpeg')
                overlay_files = [
                    os.path.join(self.cfg['overlay_dir'], f) 
                    for f in os.listdir(self.cfg['overlay_dir']) 
                    if f.lower().endswith(valid_ext)
                ]

            self.detect_scenes(main_video.duration)
            onsets, rms, flat, sr, audio_dur = self.analyze_audio()

            target_duration = audio_dur
            if max_output_duration:
                target_duration = min(audio_dur, max_output_duration)
                onsets = [t for t in onsets if t < target_duration]
                if onsets[-1] < target_duration:
                    onsets.append(target_duration)

            rms_mean = np.mean(rms)
            flat_mean = np.mean(flat)
            hop_len = 512

            final_clips = []
            self.log(f"Generating montage from {len(onsets)-1} segments...")

            # --- ЦИКЛ ПО СЕГМЕНТАМ ---
            for i in range(len(onsets) - 1):
                if self.abort:
                    self.log("Render aborted.")
                    break

                t_start = onsets[i]
                t_end = onsets[i+1]
                dur = t_end - t_start

                if dur < self.cfg.get('min_cut_duration', 0.05):
                    continue

                idx = self.get_time_index(t_start, sr, len(rms), hop_len)
                curr_rms = rms[idx]
                curr_flat = flat[idx]
                
                rms_change = 0
                if idx > 5:
                    prev_rms = rms[idx - 5]
                    rms_change = curr_rms - prev_rms

                beat_thresh = self.cfg.get('threshold', 1.2)
                is_loud = curr_rms > (rms_mean * beat_thresh)
                is_noisy = curr_flat > (flat_mean * 1.5)
                is_transient = rms_change > (rms_mean * 0.5)
                chaos = self.cfg.get('chaos_level', 0.5)
                
                segment_parts = []
                
                # 1. FLASH
                flash_active = False
                if self.cfg.get('fx_flash') and is_transient and random.random() < self.cfg.get('fx_flash_chance', 0.5):
                    flash_dur = MIN_SEGMENT_DURATION * 2
                    if dur > flash_dur * 2:
                        # fx_flash теперь использует self.target_resolution внутри
                        f_clip = self.fx_flash(flash_dur)
                        segment_parts.append(f_clip)
                        dur -= flash_dur
                        flash_active = True

                # 2. SOURCE CLIP
                # Сразу ресайзим к целевому разрешению
                clip = self.get_source_clip(main_video, dur).resize(newsize=self.target_resolution)

                # 3. EFFECTS
                if self.cfg.get('fx_stutter') and is_loud and dur < 0.3:
                    if random.random() < (0.3 + chaos * 0.5):
                        repeats = random.choice([2, 4, 8])
                        clip = self.fx_stutter(clip, repeats)

                elif self.cfg.get('fx_psort') and is_noisy:
                    if random.random() < self.cfg.get('fx_psort_chance', 0.5):
                        clip = self.fx_pixel_sort(clip)

                if self.cfg.get('fx_ghost') and is_loud:
                      clip = self.fx_ghost_trails(clip)

                if not is_loud and not is_noisy and dur > 1.0:
                    clip = clip.fx(colorx, 0.6)

                segment_parts.append(clip)
                
                # Сборка частей сегмента (Flash + Clip)
                # Поскольку мы гарантировали размер, здесь все ок
                full_segment = concatenate_videoclips(segment_parts)

                # 4. OVERLAYS
                if overlay_files and self.cfg.get('fx_overlay'):
                    if random.random() < (0.15 + chaos * 0.2):
                        ov_path = random.choice(overlay_files)
                        try:
                            img = Image.open(ov_path)
                            # Логика ресайза оверлея
                            max_h_target = self.target_resolution[1] // 2
                            if img.height > max_h_target * 1.5:
                                ratio = max_h_target * 1.5 / img.height
                                new_size = (int(img.width * ratio), int(img.height * ratio))
                                img = img.resize(new_size, Image.Resampling.LANCZOS)
                            
                            ov_clip = ImageClip(np.array(img)).set_duration(full_segment.duration)
                            
                            max_h = self.target_resolution[1] // 2
                            # Рандомный ресайз
                            ov_clip = ov_clip.resize(height=random.randint(max_h // 2, max_h))
                            
                            pos_x = random.randint(0, self.target_resolution[0] - ov_clip.w)
                            pos_y = random.randint(0, self.target_resolution[1] - ov_clip.h)
                            
                            # ВАЖНО: Явно указываем size при композитинге
                            full_segment = CompositeVideoClip(
                                [full_segment, ov_clip.set_position((pos_x, pos_y))],
                                size=self.target_resolution
                            )
                        except Exception as e:
                            self.log(f"Overlay warning: {e}") 
                            pass

                # Финальная гарантия размера перед добавлением в общий список.
                # Если какой-то эффект сдвинул размер на 1 пиксель, это исправит ситуацию.
                if full_segment.size != self.target_resolution:
                     full_segment = full_segment.resize(newsize=self.target_resolution)

                final_clips.append(full_segment)

                if self.progress_callback:
                    pct = int((i / len(onsets)) * 100)
                    self.progress_callback(f"Processing... {pct}%", pct)

            # --- ФИНАЛЬНЫЙ РЕНДЕР ---
            self.log("Concatenating all segments...")
            if not final_clips:
                self.log("Error: No clips generated.")
                return

            final_video = concatenate_videoclips(final_clips, method="chain")
            
            if final_video.duration > target_duration:
                final_video = final_video.subclip(0, target_duration)
            
            audio_sub = main_audio.subclip(0, target_duration)
            final_video = final_video.set_audio(audio_sub)

            self.log(f"Writing to disk: {self.cfg['output_path']}")
            
            final_video.write_videofile(
                self.cfg['output_path'],
                fps=DEFAULT_FPS,
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                threads=4,
                logger=None
            )

            self.log("DONE! Video saved.")
            
            main_audio.close()
            main_video.close()
            final_video.close()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log(f"CRITICAL ERROR: {str(e)}")
            raise e

if __name__ == "__main__":
    print("This module is part of the Breakcore Engine and should be run via GUI.")