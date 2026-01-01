import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os
import cv2 
from PIL import Image, ImageTk, ImageFont 
import time 

from engine import BreakcoreEngine 

# --- ЦВЕТА WIN95 ---
C_SILVER = '#C0C0C0'     # Главный фон
C_DARK_GRAY = '#808080'  # Темный цвет тени
C_BLACK = '#000000'      # Черный
C_WHITE = '#FFFFFF'      # Белый
C_TITLE_BAR = '#000080'  # Темно-синий заголовок
C_TEXT_BLACK = '#000000' # Текст
C_TEXT_ACTIVE = '#FFFFFF' # Белый текст на синем фоне

# --- ПРЕСЕТЫ ---
PRESETS = {
    "Drillcore (Fast Cut/Stutter)": {
        'chaos_level': 0.8, 'threshold': 1.0, 'min_cut_duration': 0.05, 'scene_buffer_size': 5, 
        'fx_psort': True, 'fx_psort_chance': 0.2, 'fx_psort_int': 0.3,
        'fx_stutter': True, 'fx_flash': True, 'fx_flash_chance': 0.5,
        'fx_rgb': True, 'fx_ghost': False, 'fx_ghost_int': 0.2,
        'fx_overlay': True, 'fx_color': False, 'fx_color_int': 1.0,
        'use_scene_detect': False 
    },
    "Datamix (Glitch/Slowmo)": {
        'chaos_level': 0.4, 'threshold': 1.4, 'min_cut_duration': 0.1, 'scene_buffer_size': 20,
        'fx_psort': True, 'fx_psort_chance': 0.8, 'fx_psort_int': 0.7,
        'fx_stutter': False, 'fx_flash': False, 'fx_flash_chance': 0.0,
        'fx_rgb': True, 'fx_ghost': True, 'fx_ghost_int': 0.7,
        'fx_overlay': True, 'fx_color': True, 'fx_color_int': 1.8,
        'use_scene_detect': True
    },
    "Rhythm Flash (Scene Mix)": {
        'chaos_level': 0.6, 'threshold': 1.1, 'min_cut_duration': 0.08, 'scene_buffer_size': 15,
        'fx_psort': False, 'fx_psort_chance': 0.0, 'fx_psort_int': 0.0,
        'fx_stutter': True, 'fx_flash': True, 'fx_flash_chance': 1.0,
        'fx_rgb': False, 'fx_ghost': False, 'fx_ghost_int': 0.0,
        'fx_overlay': False, 'fx_color': True, 'fx_color_int': 1.3,
        'use_scene_detect': True
    },
}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Breaker.exe - Algorithmic Cut Editor")
        self.geometry("1400x800")
        self.configure(bg=C_SILVER) # Основной фон
        self.resizable(True, True)

        self.audio_path = ""; self.video_path = ""; self.overlay_dir = ""
        self.temp_preview_path = "temp_preview.mp4"

        # Плеер и его состояние
        self.progress_var = tk.DoubleVar(value=0)
        self.video_cap = None
        self.playback_thread = None
        self.stop_playback = threading.Event()

        self.style = ttk.Style(self)
        self.setup_styles()
        self.setup_vars()
        self.setup_ui()
        self.apply_preset("Drillcore (Fast Cut/Stutter)") 

    def setup_styles(self):
        # --- Глобальные стили (Цвета, Шрифт) ---
        self.style.theme_use('clam') # Используем 'clam' как базу, но переопределяем
        self.option_add('*Font', 'MS_Sans_Serif 10')
        self.style.configure('.', background=C_SILVER, foreground=C_TEXT_BLACK, font='MS_Sans_Serif 10')

        # --- Кнопки ---
        self.style.configure('W95.TButton', 
                            background=C_SILVER, 
                            foreground=C_TEXT_BLACK,
                            relief='raised',
                            borderwidth=2,
                            highlightthickness=0,
                            focusthickness=0)
        self.style.map('W95.TButton', 
                       background=[('active', '#D6D6D6'), ('disabled', C_SILVER)],
                       relief=[('pressed', 'sunken'), ('active', 'raised')])

        # --- Фреймы/Панели ---
        self.style.configure('W95.TFrame', 
                            background=C_SILVER, 
                            relief='sunken', 
                            borderwidth=2) # Фреймы настроек
        
        # --- Tab View (Вкладки) ---
        self.style.configure('W95.TNotebook', 
                            background=C_SILVER, 
                            borderwidth=2,
                            tabmargins=[2, 2, 2, 0])
        self.style.configure('W95.TNotebook.Tab', 
                             background=C_SILVER, 
                             foreground=C_TEXT_BLACK, 
                             relief='raised', 
                             borderwidth=2)
        self.style.map('W95.TNotebook.Tab', 
                       background=[('selected', C_SILVER)], 
                       expand=[('selected', [1, 1, 0, 0])])
        self.style.configure('W95.TNotebook.Tab.Container', 
                             bordercolor=C_SILVER)

        # --- Checkbutton/Radiobutton (Классический вид) ---
        self.style.configure('W95.TCheckbutton', 
                             background=C_SILVER, 
                             foreground=C_TEXT_BLACK)
        self.style.configure('W95.TRadiobutton', 
                             background=C_SILVER, 
                             foreground=C_TEXT_BLACK)

        # --- Ползунок (Slider) ---
        self.style.configure('W95.Horizontal.TScale', 
                             background=C_SILVER, 
                             troughcolor=C_SILVER, 
                             relief='sunken', 
                             borderwidth=2)
        self.style.configure('W95.TCombobox', 
                             fieldbackground=C_WHITE,
                             background=C_SILVER,
                             foreground=C_TEXT_BLACK)
        
        # --- Progress Bar (Черепаший) ---
        self.style.configure('W95.Horizontal.TProgressbar',
                            background=C_SILVER,
                            troughcolor=C_SILVER,
                            relief='sunken',
                            bordercolor=C_DARK_GRAY,
                            borderwidth=2,
                            thickness=18)
        self.style.layout('W95.Horizontal.TProgressbar',
                         [('W95.Horizontal.TProgressbar.trough', 
                           {'children': [('W95.Horizontal.TProgressbar.pbar',
                                         {'side': 'left', 'sticky': 'ns'})],
                            'sticky': 'nswe'})])
        self.style.configure("green.W95.Horizontal.TProgressbar", foreground=C_TITLE_BAR, background=C_TITLE_BAR)

    def setup_vars(self):
        # Инициализация переменных для ползунков и свитчей
        self.vars = {}
        vars_map = {
            'chaos_level': 0.6, 'threshold': 1.2, 'min_cut_duration': 0.05, 'scene_buffer_size': 10,
            'fx_psort': True, 'fx_psort_chance': 0.5, 'fx_psort_int': 0.5,
            'fx_stutter': True, 'fx_flash': True, 'fx_flash_chance': 0.8,
            'fx_rgb': True, 'fx_ghost': False, 'fx_ghost_int': 0.5,
            'fx_overlay': True, 'fx_color': False, 'fx_color_int': 1.5,
            'use_scene_detect': False
        }
        for name, default in vars_map.items():
            if isinstance(default, bool):
                self.vars[name] = tk.BooleanVar(value=default)
            elif isinstance(default, (int, float)):
                self.vars[name] = tk.DoubleVar(value=default)

    def setup_ui(self):
        # --- ОСНОВНОЙ КОНТЕЙНЕР ---
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- ЛЕВАЯ КОЛОНКА (Настройки) ---
        self.left_frame = ttk.Frame(self, style='W95.TFrame')
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        # Заголовок (Имитация Title Bar)
        title_frame = tk.Frame(self.left_frame, bg=C_TITLE_BAR, height=30)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="[S Y N C__B R E A K] ULTIMATE; - Configuration", fg=C_WHITE, bg=C_TITLE_BAR, font=("MS Sans Serif", 11, "bold")).pack(side="left", padx=5)

        content_frame = tk.Frame(self.left_frame, bg=C_SILVER)
        content_frame.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Files Frame (Styled as a sunken panel)
        f_frame = self.mk_panel(content_frame, text="Source Files")
        self.btn_audio = self.mk_btn(f_frame, "Load Audio (WAV/MP3)", self.sel_audio)
        self.btn_video = self.mk_btn(f_frame, "Load Source Video", self.sel_video)
        self.btn_overlay = self.mk_btn(f_frame, "Load Overlays Folder", self.sel_ov)

        self.btn_audio.pack(fill="x", padx=5, pady=2)
        self.btn_video.pack(fill="x", padx=5, pady=2)
        self.btn_overlay.pack(fill="x", padx=5, pady=2)

        # Пресеты (В отдельной sunken панели)
        preset_panel = self.mk_panel(content_frame, text="Presets")
        tk.Label(preset_panel, text="Load Configuration:").pack(side="left", padx=5)
        self.preset_option = ttk.Combobox(preset_panel, values=list(PRESETS.keys()), style='W95.TCombobox')
        self.preset_option.bind("<<ComboboxSelected>>", self.apply_preset_from_gui)
        self.preset_option.set(list(PRESETS.keys())[0])
        self.preset_option.pack(side="left", expand=True, fill="x", padx=5)
        
        # Tabs
        self.tab_view = ttk.Notebook(content_frame, style='W95.TNotebook')
        self.tab_view.pack(pady=10, padx=5, fill="both", expand=True)
        self.tab_cut_frame = self.mk_tab("I. Cut Logic")
        self.tab_datamix_frame = self.mk_tab("II. Datamix Core")
        self.tab_rhythm_frame = self.mk_tab("III. Rhythm FX")
        self.tab_finals_frame = self.mk_tab("IV. Finals")
        self.tab_view.add(self.tab_cut_frame, text="I. Cut Logic")
        self.tab_view.add(self.tab_datamix_frame, text="II. Datamix Core")
        self.tab_view.add(self.tab_rhythm_frame, text="III. Rhythm FX")
        self.tab_view.add(self.tab_finals_frame, text="IV. Finals")

        self._setup_cut_logic(self.tab_cut_frame)
        self._setup_datamix(self.tab_datamix_frame)
        self._setup_rhythm_fx(self.tab_rhythm_frame)
        self._setup_finals(self.tab_finals_frame)
        
        # --- ПРАВАЯ КОЛОНКА (Плеер и Консоль) ---
        self.right_frame = ttk.Frame(self, style='W95.TFrame')
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        # Header (Имитация Title Bar)
        title_frame_r = tk.Frame(self.right_frame, bg=C_TITLE_BAR, height=30)
        title_frame_r.pack(fill="x")
        tk.Label(title_frame_r, text="Live Preview and Console", fg=C_WHITE, bg=C_TITLE_BAR, font=("MS Sans Serif", 11, "bold")).pack(side="left", padx=5)

        content_frame_r = tk.Frame(self.right_frame, bg=C_SILVER)
        content_frame_r.pack(fill="both", expand=True, padx=2, pady=2)

        # Плеер (Имитация sunken поля)
        player_panel = self.mk_panel(content_frame_r, text="Preview Monitor (640x360)", relief_style='sunken')
        player_panel.pack(pady=10, padx=10, fill="x", expand=False)
        self.player_label = tk.Label(player_panel, text="Load Video to Start Preview Render", 
                                     bg=C_BLACK, fg=C_WHITE, width=80, height=20, bd=2, relief='sunken')
        self.player_label.pack(fill="x", expand=True, padx=5, pady=5)
        
        # Кнопки управления
        btn_frame = tk.Frame(content_frame_r, bg=C_SILVER)
        btn_frame.pack(pady=5, fill="x", padx=10)
        self.btn_preview = self.mk_btn(btn_frame, "RENDER PREVIEW (5 sec)", lambda: self.run(preview_mode=True), C_TITLE_BAR)
        self.btn_preview.pack(side="left", padx=5, expand=True, fill="x")
        self.btn_stop = self.mk_btn(btn_frame, "STOP PLAYBACK", self.stop_and_clear_playback, C_DARK_GRAY, state='disabled')
        self.btn_stop.pack(side="left", padx=5, expand=True, fill="x")

        # Progress Bar
        self.progress = ttk.Progressbar(content_frame_r, style='green.W95.Horizontal.TProgressbar', mode='determinate', maximum=100, variable=self.progress_var)
        self.progress.pack(fill="x", padx=10, pady=5)

        # Консоль (Имитация sunken text field)
        console_panel = self.mk_panel(content_frame_r, text="Status Log", relief_style='sunken')
        self.console = tk.Text(console_panel, height=10, font=("Courier New", 10), bg=C_WHITE, fg=C_BLACK, bd=2, relief='sunken')
        self.console.pack(pady=5, fill="both", expand=True, padx=5)
        
        self.btn_run_full = self.mk_btn(content_frame_r, "RENDER FULL VIDEO", lambda: self.run(preview_mode=False), C_TITLE_BAR)
        self.btn_run_full.pack(pady=(10, 5), fill="x", padx=10)

    # --- УТИЛИТЫ И КЛАССИЧЕСКИЕ ВИДЖЕТЫ ---
    def mk_panel(self, master, text="", relief_style='groove'):
        panel = tk.LabelFrame(master, text=text, bg=C_SILVER, fg=C_TEXT_BLACK, bd=2, relief=relief_style, font=("MS Sans Serif", 10, "bold"))
        panel.pack(pady=5, padx=10, fill="x")
        return panel

    def mk_tab(self, name):
        tab_frame = ttk.Frame(self, style='W95.TFrame')
        tab_frame.grid_columnconfigure((0, 1), weight=1)
        return tab_frame

    def mk_btn(self, master, txt, cmd, bg_color=C_SILVER, state='normal'):
        return ttk.Button(master, text=txt, command=cmd, style='W95.TButton', state=state)

    def mk_sw(self, master, txt, default_val, row, col, var_name):
        var = self.vars[var_name]
        sw = ttk.Checkbutton(master, text=txt, variable=var, style='W95.TCheckbutton')
        sw.grid(row=row, column=col, pady=5, padx=10, sticky="w")
        return sw

    def mk_slider(self, master, label_text, var_name, from_, to, row, col, steps=None):
        tk.Label(master, text=label_text, bg=C_SILVER, fg=C_TEXT_BLACK).grid(row=row, column=col, pady=2, sticky="w", padx=10)
        
        # Tk Scale (Slider)
        if steps:
            s = ttk.Scale(master, from_=from_, to=to, variable=self.vars[var_name], orient=tk.HORIZONTAL, style='W95.Horizontal.TScale', value=self.vars[var_name].get(), length=250)
        else:
            s = ttk.Scale(master, from_=from_, to=to, variable=self.vars[var_name], orient=tk.HORIZONTAL, style='W95.Horizontal.TScale', value=self.vars[var_name].get(), length=250)
            
        s.grid(row=row+1, column=col, padx=10, pady=2, sticky="ew")
        
        # Добавляем индикатор значения рядом
        val_label = tk.Label(master, textvariable=self.vars[var_name], bg=C_SILVER, fg=C_TEXT_BLACK)
        val_label.grid(row=row, column=col, sticky="e", padx=10)
        
        return s
    
    # --- НАСТРОЙКА ТАБОВ ---
    def _setup_cut_logic(self, master):
        self.mk_slider(master, "Global Chaos (0.0-1.0)", 'chaos_level', 0.0, 1.0, 0, 0)
        self.mk_slider(master, "Beat Threshold (0.5-2.0)", 'threshold', 0.5, 2.0, 0, 1)
        
        self.mk_slider(master, "Min Cut Duration (sec)", 'min_cut_duration', 0.0, 0.2, 2, 0, steps=20)
        
        self.mk_sw(master, "Smart Scene Detection", False, 4, 0, 'use_scene_detect')
        self.mk_slider(master, "Scene Buffer Size (2-30)", 'scene_buffer_size', 2, 30, 4, 1, steps=28)

    def _setup_datamix(self, master):
        self.mk_sw(master, "Pixel Sort Simulation (Datamosh Look)", True, 0, 0, 'fx_psort')
        self.mk_slider(master, "Pixel Sort Chance (0.0-1.0)", 'fx_psort_chance', 0.0, 1.0, 1, 0)
        self.mk_slider(master, "Pixel Sort Intensity (0.0-1.0)", 'fx_psort_int', 0.0, 1.0, 1, 1)

    def _setup_rhythm_fx(self, master):
        self.mk_sw(master, "Stutter / Drill", True, 0, 0, 'fx_stutter')
        self.mk_sw(master, "Flash Frame (on RMS Drop)", True, 0, 1, 'fx_flash')
        self.mk_slider(master, "Flash Chance (0.0-1.0)", 'fx_flash_chance', 0.0, 1.0, 1, 1)

        self.mk_sw(master, "RGB Shift", True, 3, 0, 'fx_rgb')
        self.mk_sw(master, "Ghost Trails", False, 3, 1, 'fx_ghost')
        self.mk_slider(master, "Ghost Trail Opacity (0.0-1.0)", 'fx_ghost_int', 0.0, 1.0, 4, 1)

    def _setup_finals(self, master):
        self.mk_sw(master, "Use Overlays", True, 0, 0, 'fx_overlay')
        self.mk_sw(master, "Color/Saturation Boost", False, 1, 0, 'fx_color')
        self.mk_slider(master, "Color Boost Intensity (1.0-3.0)", 'fx_color_int', 1.0, 3.0, 2, 0)
        
    # --- ЛОГИКА ФАЙЛОВ И КОНФИГУРАЦИИ ---
    def sel_audio(self):
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav")])
        if p: 
            self.audio_path = p
            self.btn_audio.configure(text=f"AUDIO: {os.path.basename(p)}")

    def sel_video(self):
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov *.mkv")])
        if p: 
            self.video_path = p
            self.btn_video.configure(text=f"VIDEO: {os.path.basename(p)}")

    def sel_ov(self):
        p = filedialog.askdirectory()
        if p: 
            self.overlay_dir = p
            self.btn_overlay.configure(text=f"OVERLAYS: {os.path.basename(p)}")
    
    def log(self, msg):
        # Добавляем [TIME] и скроллим
        self.console.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] > {msg}\n")
        self.console.see(tk.END)

    def get_current_config(self):
        # Собираем текущий конфиг из всех переменных
        cfg = {name: var.get() for name, var in self.vars.items()}
        # Конвертируем float-слайдеры, которые могут быть случайно int, обратно в float
        for key in ['chaos_level', 'threshold', 'min_cut_duration', 'fx_psort_chance', 'fx_psort_int', 'fx_flash_chance', 'fx_ghost_int', 'fx_color_int']:
            cfg[key] = float(cfg[key])
        cfg['scene_buffer_size'] = int(cfg['scene_buffer_size'])
        return cfg

    def apply_preset(self, preset_name):
        cfg = PRESETS[preset_name]
        for key, value in cfg.items():
            if key in self.vars:
                self.vars[key].set(value)
        self.log(f"Preset '{preset_name}' loaded.")

    def apply_preset_from_gui(self, event):
        self.apply_preset(self.preset_option.get())

    # --- ГЛАВНАЯ ЛОГИКА RUN И ПОТОКИ ---
    def run(self, preview_mode=False):
        if not self.audio_path or not self.video_path:
            self.log("ERROR: Select Audio and Video source!")
            return

        cfg = self.get_current_config()
        cfg['audio_path'] = self.audio_path
        cfg['video_path'] = self.video_path
        cfg['overlay_dir'] = self.overlay_dir
        
        if preview_mode:
            cfg['output_path'] = self.temp_preview_path
            cfg['max_duration'] = 5.0
            self.log("Starting QUICK PREVIEW RENDER (5 sec)...")
            self.btn_preview.configure(state="disabled")
            self.btn_run_full.configure(state="disabled")
            self.progress.configure(mode='indeterminate')
            self.progress.start(10)
        else:
            cfg['output_path'] = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")], initialfile="render_breakcore.mp4")
            if not cfg['output_path']:
                self.btn_run_full.configure(state="normal"); return
            cfg['max_duration'] = None
            self.log("Starting FULL VIDEO RENDER...")
            self.btn_preview.configure(state="disabled")
            self.btn_run_full.configure(state="disabled")
            self.progress.configure(mode='determinate', value=0)


        t = threading.Thread(target=self.thread_task, args=(cfg, preview_mode))
        t.start()

    def thread_task(self, cfg, preview_mode):
        # Callback для обновления прогресса
        def progress_update(message=None, value=None):
            self.after(0, self.log, message)
            if not preview_mode and value is not None:
                 self.after(0, self.progress_var.set, value)
        
        engine = BreakcoreEngine(cfg, progress_callback=progress_update)
        try:
            engine.run(max_output_duration=cfg.get('max_duration'))
            self.after(0, self.log, f"--- {'PREVIEW' if preview_mode else 'FULL RENDER'} COMPLETE! Output: {cfg['output_path']} ---")
            
            if preview_mode:
                self.after(0, self.start_playback, self.temp_preview_path)
            
        except Exception as e:
            self.after(0, self.log, f"ERROR: {e}")
        finally:
            self.after(0, lambda: self.btn_preview.configure(state="normal"))
            self.after(0, lambda: self.btn_run_full.configure(state="normal"))
            self.after(0, lambda: self.progress.stop())
            self.after(0, lambda: self.progress_var.set(0))
            if preview_mode:
                self.after(0, lambda: self.progress.configure(mode='determinate', value=0))

    # --- ЛОГИКА ПЛЕЕРА (CV2) ---
    def start_playback(self, video_path):
        """Начинает проигрывание видео в отдельном потоке."""
        self.stop_and_clear_playback()
        
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            self.log("ERROR: Could not open preview video for playback.")
            return

        self.log("Starting playback (loop)...")
        self.btn_stop.configure(state="normal")
        self.stop_playback.clear()
        
        self.playback_thread = threading.Thread(target=self._play_video_thread, daemon=True)
        self.playback_thread.start()

    def _play_video_thread(self):
        fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 24
        delay_ms = int(1000 / fps)
        
        # Получаем размер, чтобы ресайзить кадры
        w_widget = self.player_label.winfo_width()
        h_widget = self.player_label.winfo_height()
        
        while not self.stop_playback.is_set():
            ret, frame = self.video_cap.read()
            
            if ret:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_image = Image.fromarray(cv2image)
                
                # Масштабирование
                if w_widget > 100 and h_widget > 100:
                    current_image = current_image.resize((w_widget, h_widget), Image.Resampling.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=current_image)
                
                self.after(0, self._update_player_label, imgtk)
                
                # Ждем
                self.stop_playback.wait(delay_ms / 1000.0)
            else:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
    def _update_player_label(self, imgtk):
        self.player_label.imgtk = imgtk 
        self.player_label.configure(image=imgtk, text="")
        
    def stop_and_clear_playback(self):
        if self.playback_thread and self.playback_thread.is_alive():
            self.stop_playback.set()
            self.playback_thread.join(timeout=0.1)
            
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
            
        self.btn_stop.configure(state="disabled")
        self.player_label.configure(image=None, text="Preview Stopped/Ready", bg=C_BLACK)

    def on_closing(self):
        self.stop_and_clear_playback()
        if os.path.exists(self.temp_preview_path):
            os.remove(self.temp_preview_path)
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()