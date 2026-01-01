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
MIN_SEGMENT_DURATION = 0.04  # ~1 кадр при 24fps. Меньше этого MoviePy может крашиться.
DEFAULT_FPS = 24

class GhostEffect:
    """
    Класс для реализации эффекта Ghost Trails (Шлейф) с сохранением состояния.
    Использует замыкание для хранения предыдущего кадра.
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.last_frame = None

    def apply(self, get_frame, t):
        current_frame = get_frame(t)
        
        # Если это первый кадр, просто сохраняем его
        if self.last_frame is None:
            self.last_frame = current_frame.astype('float')
            return current_frame

        # Смешивание: New = Current * alpha + Old * (1 - alpha)
        # Работаем во float для точности, потом конвертируем обратно в uint8
        curr_float = current_frame.astype('float')
        
        blended = cv2.addWeighted(curr_float, 1.0, self.last_frame, self.alpha, 0)
        
        self.last_frame = blended # Обновляем "память"
        return blended.astype('uint8')

class BreakcoreEngine:
    def __init__(self, config: Dict[str, Any], progress_callback=None):
        """
        Инициализация движка.
        :param config: Словарь с настройками из GUI.
        :param progress_callback: Функция для обновления прогресс-бара и логов в GUI.
        """
        self.cfg = config
        self.progress_callback = progress_callback
        self.abort = False
        self.scene_cuts: List[float] = []      # Список таймкодов смены сцен
        self.scene_buffer: List[float] = []    # Буфер для рандомизации
        
        # Валидация путей
        if not os.path.exists(self.cfg['video_path']):
            raise FileNotFoundError(f"Video file not found: {self.cfg['video_path']}")
        if not os.path.exists(self.cfg['audio_path']):
            raise FileNotFoundError(f"Audio file not found: {self.cfg['audio_path']}")

    def log(self, message: str):
        """Вывод логов в консоль GUI и stdout."""
        print(f"[ENGINE] {message}")
        if self.progress_callback:
            self.progress_callback(message)

    # --- 1. АНАЛИЗ ВИДЕО (Scene Detection) ---
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
            # Сохраняем начало каждой сцены (в секундах)
            self.scene_cuts = [x[0].get_seconds() for x in scene_list]
            
            # Фильтруем сцены, которые выходят за пределы видео (на всякий случай)
            self.scene_cuts = [t for t in self.scene_cuts if t < video_duration - 1.0]
            
            self.log(f"Detected {len(self.scene_cuts)} scenes.")
            
            # Заполняем буфер (берем N первых уникальных сцен)
            buf_size = int(self.cfg.get('scene_buffer_size', 10))
            self.scene_buffer = self.scene_cuts[:buf_size]
            
        except Exception as e:
            self.log(f"Scene detection warning: {e}. Fallback to random seeking.")
            self.scene_cuts = []
        finally:
            video_manager.release()

    # --- 2. АНАЛИЗ АУДИО (Librosa) ---
    def analyze_audio(self):
        self.log("Computing audio features (RMS, Onsets, Spectral Flatness)...")
        y, sr = librosa.load(self.cfg['audio_path'])
        
        # 1. Onsets (Ритмическая сетка)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time', backtrack=True)
        
        # 2. RMS (Громкость)
        rms = librosa.feature.rms(y=y)[0]
        
        # 3. Spectral Flatness (Шумность/Тон)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        
        return onsets, rms, flatness, sr, y.shape[0] / sr

    # --- 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
    def get_time_index(self, t: float, sr: int, array_len: int, hop_length: int = 512) -> int:
        """Переводит время в индекс массива features."""
        frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
        return min(frame, array_len - 1)

    def get_source_clip(self, main_video: VideoFileClip, duration: float) -> VideoFileClip:
        """
        Выбирает исходный кусок видео.
        Логика:
        - Если есть Scene Buffer и Chaos < 1.0 -> берем начало сцены.
        - Иначе -> берем случайное место.
        """
        chaos = self.cfg.get('chaos_level', 0.5)
        use_scenes = len(self.scene_buffer) > 0
        
        # Шанс взять начало сцены (уменьшается с ростом хаоса)
        should_use_scene = use_scenes and (random.random() > chaos * 0.8)
        
        start_t = 0.0
        if should_use_scene:
            start_t = random.choice(self.scene_buffer)
            # Добавляем небольшой случайный сдвиг (jitter) внутри сцены
            start_t += random.uniform(0, 1.0)
        else:
            start_t = random.uniform(0, max(0, main_video.duration - duration))

        # Защита границ
        if start_t + duration > main_video.duration:
            start_t = max(0, main_video.duration - duration - 0.1)
            
        return main_video.subclip(start_t, start_t + duration)

    # --- 4. ЭФФЕКТЫ ---

    def fx_pixel_sort(self, clip: VideoFileClip) -> VideoFileClip:
        """
        Реализация Pixel Sort (Datamosh simulation).
        Сортирует пиксели по яркости в случайной полосе.
        """
        intensity = self.cfg.get('fx_psort_int', 0.5)

        def fl(gf, t):
            frame = gf(t) # Получаем кадр (numpy array read-only)
            img_arr = frame.copy() # Копия для записи
            h, w, _ = img_arr.shape
            
            # Определяем область сортировки (высота зависит от интенсивности)
            sort_h = int(h * (0.1 + intensity * 0.6))
            y_start = random.randint(0, h - sort_h)
            
            # Вырезаем область
            crop = img_arr[y_start:y_start+sort_h, :, :]
            
            # Преобразуем в список пикселей
            # shape: (rows, cols, 3) -> (rows * cols, 3)
            pixels = crop.reshape(-1, 3)
            
            # Вычисляем яркость: (R+G+B)
            lum = pixels.sum(axis=1)
            
            # Сортируем индексы
            sorted_indices = np.argsort(lum)
            
            # Применяем сортировку и возвращаем форму
            sorted_pixels = pixels[sorted_indices]
            sorted_crop = sorted_pixels.reshape(sort_h, w, 3)
            
            # Вставляем обратно
            img_arr[y_start:y_start+sort_h, :, :] = sorted_crop
            return img_arr

        return clip.fl(fl)

    def fx_stutter(self, clip: VideoFileClip, count: int) -> VideoFileClip:
        """Эффект 'заедания' (Drill/Stutter)."""
        if clip.duration < MIN_SEGMENT_DURATION:
            return clip
            
        # Длина одного повторения
        segment_len = clip.duration / count
        
        # Если сегмент слишком мал, ограничиваем минималкой
        if segment_len < MIN_SEGMENT_DURATION:
            segment_len = MIN_SEGMENT_DURATION
            # Пересчитываем кол-во, чтобы влезть в длительность (примерно)
            count = int(clip.duration / segment_len)
            if count < 1: count = 1

        sub = clip.subclip(0, segment_len)
        # Склеиваем повторения
        stuttered = concatenate_videoclips([sub] * count)
        
        # Подрезаем под исходную длительность ровно
        if stuttered.duration > clip.duration:
            stuttered = stuttered.subclip(0, clip.duration)
            
        return stuttered

    def fx_ghost_trails(self, clip: VideoFileClip) -> VideoFileClip:
        """
        Наложение шлейфа (Ghost Trails).
        Использует stateful класс GhostEffect.
        """
        opacity = self.cfg.get('fx_ghost_opacity', 0.5)
        effect = GhostEffect(alpha=opacity)
        # clip.fl требует функцию (get_frame, t)
        return clip.fl(effect.apply)

    def fx_flash(self, duration: float) -> VideoFileClip:
        """Генерация белой или черной вспышки."""
        col = 'white' if random.random() > 0.5 else 'black'
        return ColorClip(size=(1280, 720), color=col, duration=duration).set_opacity(0.8)

    # --- 5. ГЛАВНЫЙ ЦИКЛ (RUN) ---
    def run(self, max_output_duration: Optional[float] = None):
        try:
            self.log("Initializing media...")
            main_audio = AudioFileClip(self.cfg['audio_path'])
            main_video = VideoFileClip(self.cfg['video_path'])
            
            # Принудительный ресайз видео к HD или исходному, чтобы эффекты работали предсказуемо
            # (Опционально, но рекомендуется для PixelSort)
            w, h = main_video.size
            if w > 1920: main_video = main_video.resize(width=1920)

            # Подготовка оверлеев
            overlay_files = []
            if self.cfg.get('overlay_dir') and os.path.exists(self.cfg['overlay_dir']):
                valid_ext = ('.png', '.jpg', '.jpeg')
                overlay_files = [
                    os.path.join(self.cfg['overlay_dir'], f) 
                    for f in os.listdir(self.cfg['overlay_dir']) 
                    if f.lower().endswith(valid_ext)
                ]

            # Запуск анализа
            self.detect_scenes(main_video.duration)
            onsets, rms, flat, sr, audio_dur = self.analyze_audio()

            # Ограничение длительности рендера (для Preview)
            target_duration = audio_dur
            if max_output_duration:
                target_duration = min(audio_dur, max_output_duration)
                # Обрезаем onsets, выходящие за пределы превью
                onsets = [t for t in onsets if t < target_duration]
                # Добавляем точку конца превью, если её нет
                if onsets[-1] < target_duration:
                    onsets.append(target_duration)

            # Статистика для порогов
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

                # Пропуск микро-сегментов
                if dur < self.cfg.get('min_cut_duration', 0.05):
                    continue

                # Анализ текущего сегмента
                idx = self.get_time_index(t_start, sr, len(rms), hop_len)
                curr_rms = rms[idx]
                curr_flat = flat[idx]
                
                # RMS Change (ищем резкий скачок вверх по сравнению с прошлым)
                rms_change = 0
                if idx > 5:
                    prev_rms = rms[idx - 5] # смотрим немного назад
                    rms_change = curr_rms - prev_rms

                # Флаги состояний
                beat_thresh = self.cfg.get('threshold', 1.2)
                is_loud = curr_rms > (rms_mean * beat_thresh)
                is_noisy = curr_flat > (flat_mean * 1.5)
                is_transient = rms_change > (rms_mean * 0.5)

                # Выбор вероятностей
                chaos = self.cfg.get('chaos_level', 0.5)
                
                # КОНТЕЙНЕР ДЛЯ ТЕКУЩЕГО КУСКА
                # Может состоять из [Flash, MainClip] или просто [MainClip]
                segment_parts = []
                
                # 1. FLASH FRAME (Приоритет: Transient/Удар)
                flash_active = False
                if self.cfg.get('fx_flash') and is_transient and random.random() < self.cfg.get('fx_flash_chance', 0.5):
                    flash_dur = MIN_SEGMENT_DURATION * 2 # 2 кадра
                    if dur > flash_dur * 2: # Вставляем только если есть место
                        f_clip = self.fx_flash(flash_dur)
                        # Ресайзим цветной клип под видео
                        f_clip = f_clip.resize(newsize=main_video.size) 
                        segment_parts.append(f_clip)
                        dur -= flash_dur # Укорачиваем основное видео
                        flash_active = True

                # 2. ГЕНЕРАЦИЯ ОСНОВНОГО КЛИПА
                clip = self.get_source_clip(main_video, dur)

                # 3. ПРИМЕНЕНИЕ ЭФФЕКТОВ К ОСНОВНОМУ КЛИПУ

                # A. Stutter (Если громко и коротко)
                if self.cfg.get('fx_stutter') and is_loud and dur < 0.3:
                    # Чем больше хаос, тем выше шанс
                    if random.random() < (0.3 + chaos * 0.5):
                        repeats = random.choice([2, 4, 8])
                        clip = self.fx_stutter(clip, repeats)

                # B. Pixel Sort (Если шумно)
                elif self.cfg.get('fx_psort') and is_noisy:
                    if random.random() < self.cfg.get('fx_psort_chance', 0.5):
                        clip = self.fx_pixel_sort(clip)

                # C. Ghost Trails (Если громко) - Теперь работает корректно!
                if self.cfg.get('fx_ghost') and is_loud:
                     # Применяем ко всему сегменту
                     clip = self.fx_ghost_trails(clip)

                # D. Ambient/Slowmo (Если тихо и длинно)
                if not is_loud and not is_noisy and dur > 1.0:
                    # Делаем чуть темнее и медленнее
                    clip = clip.fx(colorx, 0.6)
                    # speedx может менять длительность, поэтому аккуратно
                    # Оставим просто затемнение для атмосферы, чтобы не ломать синхру

                segment_parts.append(clip)
                
                # Сборка сегмента
                full_segment = concatenate_videoclips(segment_parts)

                # 4. OVERLAYS (Поверх всего сегмента)
                if overlay_files and self.cfg.get('fx_overlay'):
                    # Шанс оверлея 15% + хаос
                    if random.random() < (0.15 + chaos * 0.2):
                        ov_path = random.choice(overlay_files)
                        ov_clip = ImageClip(ov_path).set_duration(full_segment.duration)
                        
                        # Рандомный размер и позиция
                        max_h = main_video.h // 2
                        ov_clip = ov_clip.resize(height=random.randint(max_h // 2, max_h))
                        
                        # Позиция
                        pos_x = random.randint(0, main_video.w - ov_clip.w)
                        pos_y = random.randint(0, main_video.h - ov_clip.h)
                        
                        # Композитинг
                        full_segment = CompositeVideoClip([full_segment, ov_clip.set_position((pos_x, pos_y))])

                final_clips.append(full_segment)

                # Обновление прогресса
                if self.progress_callback:
                    pct = int((i / len(onsets)) * 100)
                    self.progress_callback(f"Processing... {pct}%", pct)

            # --- ФИНАЛЬНЫЙ РЕНДЕР ---
            self.log("Concatenating all segments...")
            if not final_clips:
                self.log("Error: No clips generated.")
                return

            final_video = concatenate_videoclips(final_clips, method="compose")
            
            # Подрезаем или зацикливаем аудио под длину видео
            # Но лучше: подрезать видео под target_duration
            if final_video.duration > target_duration:
                final_video = final_video.subclip(0, target_duration)
            
            # Накладываем оригинальное аудио
            audio_sub = main_audio.subclip(0, target_duration)
            final_video = final_video.set_audio(audio_sub)

            self.log(f"Writing to disk: {self.cfg['output_path']}")
            
            # Preset 'medium' - баланс скорости и сжатия
            # threads=4 - использование многопоточности
            final_video.write_videofile(
                self.cfg['output_path'],
                fps=DEFAULT_FPS,
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                threads=4,
                logger=None # Отключаем шумный логгер moviepy, у нас свой прогресс
            )

            self.log("DONE! Video saved.")
            
            # Очистка ресурсов
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