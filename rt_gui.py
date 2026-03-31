"""Real-time GUI for the audio-reactive video engine."""
import json
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from rt_audio import (
    C_SILVER, C_DARK_GRAY, C_BLACK, C_WHITE, C_TITLE_BLUE, C_TEXT_BLACK,
)
from rt_engine import RealtimeEngine


def _make_button(parent, text, command, **kw):
    """Standard styled tk.Button."""
    return tk.Button(
        parent, text=text, command=command,
        bg=C_SILVER, activebackground='#D6D6D6',
        fg=C_TEXT_BLACK, activeforeground=C_TEXT_BLACK,
        relief='raised', bd=2, font='MS_Sans_Serif 9',
        **kw,
    )


def _make_scale(parent, label_text, variable, from_, to, resolution, **kw):
    """Labelled horizontal Scale on a shared row frame. Returns frame."""
    f = tk.Frame(parent, bg=C_SILVER)
    tk.Label(f, text=label_text, bg=C_SILVER, fg=C_TEXT_BLACK,
             font='MS_Sans_Serif 8', anchor='w', width=11).pack(side='left')
    tk.Scale(
        f, variable=variable, from_=from_, to=to, resolution=resolution,
        orient='horizontal', showvalue=True,
        bg=C_SILVER, troughcolor='#808080', activebackground=C_SILVER,
        highlightthickness=0, bd=1, sliderrelief='raised',
        font='MS_Sans_Serif 7',
        **kw,
    ).pack(side='left', fill='x', expand=True)
    return f


class RealtimeGUI(tk.Tk):
    """Single-window real-time controller."""

    # Effect keys and display labels — grouped by type
    _FX_DEFS = [
        # ── core ───────────────────────
        ('fx_rgb',        'RGB Shift'),
        ('fx_flash',      'Flash'),
        ('fx_stutter',    'Stutter'),
        ('fx_pixel_sort', 'Pxl Sort'),
        ('fx_datamosh',   'Datamosh'),
        ('fx_overlays',   'Overlays'),
        # ── glitch ─────────────────────
        ('fx_ghost',      'Ghost'),
        ('fx_scanlines',  'Scanlines'),
        ('fx_bitcrush',   'Bitcrush'),
        ('fx_blockglitch','Blk Glitch'),
        ('fx_negative',   'Negative'),
        ('fx_colorbleed', 'Clr Bleed'),
        ('fx_interlace',  'Interlace'),
        ('fx_badsignal',  'Bad Signal'),
        # ── spatial / color ────────────
        ('fx_zoomglitch', 'Zoom Glitch'),
        ('fx_mosaic',     'Mosaic'),
        ('fx_phaseshift', 'PhaseShift'),
        ('fx_dither',     'Dither'),
        ('fx_feedback',   'Feedback'),
        ('fx_temporalrgb','Temp RGB'),
        ('fx_waveshaper', 'Waveshaper'),
    ]

    # Separator positions (inserted before index i)
    _FX_SEPARATORS = {6: 'Glitch', 14: 'Spatial / Color'}

    # Available output resolutions
    _RESOLUTIONS = ['640×360', '800×450', '1280×720', '1920×1080']
    _RES_MAP = {
        '640×360':   (640,  360),
        '800×450':   (800,  450),
        '1280×720':  (1280, 720),
        '1920×1080': (1920, 1080),
    }

    def __init__(self):
        super().__init__()
        self.title("Disc VPC 01 R-T")
        self.geometry("1000x640")
        self.minsize(900, 580)
        self.configure(bg=C_SILVER)
        self.resizable(True, True)

        self.engine = RealtimeEngine(width=640, height=360)
        self._band_max = {'bass': 0.15, 'mid': 0.08, 'treble': 0.05}

        self.is_running = False
        self.update_id = None
        self.fullscreen_window = None
        self.second_monitor_window = None
        self.is_fullscreen = False

        # Toggle state for each effect (all ON by default)
        self._fx_state: dict[str, bool] = {k: True for k, _ in self._FX_DEFS}
        self._fx_btns:  dict[str, tk.Button] = {}

        # Stage control states
        self._sequential_on = False
        self._blackout_on   = False
        self._freeze_on     = False

        self._build_ui()
        self.load_audio_devices()

        # Keyboard shortcuts
        self.bind('<F11>',  self.toggle_fullscreen)
        self.bind('<Escape>', self.exit_fullscreen)
        self.bind('<F12>',  self.toggle_second_monitor)
        self.bind('<space>', self._kb_blackout)
        self.bind('<f>',    self._kb_freeze)
        self.bind('<F>',    self._kb_freeze)
        self.bind('<p>',    self._kb_panic)
        self.bind('<P>',    self._kb_panic)
        self.bind('<r>',    lambda _: self._recalibrate())
        self.bind('<R>',    lambda _: self._recalibrate())
        # 1-9 toggle first 9 effects; 0 toggles 10th
        for _i in range(9):
            self.bind(str(_i + 1), lambda e, i=_i: self._kb_toggle_fx(i))
        self.bind('0', lambda e: self._kb_toggle_fx(9))

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        tb = tk.Frame(self, bg=C_TITLE_BLUE, height=28)
        tb.pack(fill='x')
        tk.Label(tb, text="Disc VPC 01 R-T",
                 fg=C_WHITE, bg=C_TITLE_BLUE,
                 font=("MS Sans Serif", 10, "bold")).pack(side='left', padx=6)

        # Main area
        main = tk.Frame(self, bg=C_SILVER)
        main.pack(fill='both', expand=True, padx=6, pady=4)

        left = tk.Frame(main, bg=C_SILVER, width=258)
        left.pack(side='left', fill='y', padx=(0, 5))
        left.pack_propagate(False)

        right = tk.Frame(main, bg=C_SILVER)
        right.pack(side='left', fill='both', expand=True)

        self._build_left(left)
        self._build_right(right)

        # Status bar
        self.status_bar = tk.Label(self, text="Ready", bg=C_SILVER,
                                   bd=1, relief='sunken', anchor='w',
                                   font='MS_Sans_Serif 9')
        self.status_bar.pack(side='bottom', fill='x', padx=6, pady=(0, 3))

    def _build_left(self, parent):
        # Scrollable wrapper so Effects block is never clipped
        canvas = tk.Canvas(parent, bg=C_SILVER, highlightthickness=0)
        sb = tk.Scrollbar(
            parent, orient='vertical', command=canvas.yview,
            bg=C_SILVER, troughcolor='#808080',
            activebackground='#D6D6D6',
            relief='raised', bd=1, width=14,
        )
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg=C_SILVER)
        win_id = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(
            scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(
            win_id, width=e.width))
        canvas.bind('<MouseWheel>',
                    lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units'))
        parent = inner   # all subsequent widgets go into the scrollable frame

        # ── Setup ────────────────────────────────────────────────────────────
        sf = tk.LabelFrame(parent, text="Setup", bg=C_SILVER, fg=C_TEXT_BLACK,
                           font=("MS Sans Serif", 9, "bold"), bd=2, relief='groove')
        sf.pack(fill='x', pady=(0, 3))

        # Audio device row
        ar = tk.Frame(sf, bg=C_SILVER)
        ar.pack(fill='x', padx=4, pady=(4, 2))
        self.audio_var = tk.StringVar()
        self.audio_combo = ttk.Combobox(ar, textvariable=self.audio_var,
                                        state='readonly', width=22,
                                        font='MS_Sans_Serif 9')
        self.audio_combo.pack(side='left', fill='x', expand=True)
        self.audio_combo.bind('<<ComboboxSelected>>', self._on_device_change)
        _make_button(ar, "R", self.load_audio_devices, width=2).pack(side='left', padx=(3, 0))

        # Resolution + file buttons row
        rr = tk.Frame(sf, bg=C_SILVER)
        rr.pack(fill='x', padx=4, pady=(0, 2))
        self.resolution_var = tk.StringVar(value='640×360')
        res_combo = ttk.Combobox(rr, textvariable=self.resolution_var,
                                 values=self._RESOLUTIONS, state='readonly',
                                 width=9, font='MS_Sans_Serif 9')
        res_combo.pack(side='left', padx=(0, 3))

        _make_button(rr, "Video",    self.load_video
                 ).pack(side='left', fill='x', expand=True)
        _make_button(rr, "Overlays", self.load_overlays
                 ).pack(side='left', fill='x', expand=True, padx=(3, 0))

        self.file_info_label = tk.Label(sf, text="No video loaded",
                                        bg=C_SILVER, fg=C_DARK_GRAY,
                                        font='MS_Sans_Serif 8', anchor='w')
        self.file_info_label.pack(fill='x', padx=4, pady=(0, 3))

        # ── Controls ─────────────────────────────────────────────────────────
        cf = tk.LabelFrame(parent, text="Controls", bg=C_SILVER, fg=C_TEXT_BLACK,
                           font=("MS Sans Serif", 9, "bold"), bd=2, relief='groove')
        cf.pack(fill='x', pady=(0, 3))

        br = tk.Frame(cf, bg=C_SILVER)
        br.pack(fill='x', padx=4, pady=(4, 3))
        self.start_btn = tk.Button(
            br, text="START", command=self.start_engine,
            bg=C_SILVER, activebackground='#D6D6D6',
            fg='#000080', font=("MS Sans Serif", 9, "bold"),
            relief='raised', bd=2,
        )
        self.start_btn.pack(side='left', fill='x', expand=True)
        self.stop_btn = tk.Button(
            br, text="STOP", command=self.stop_engine,
            bg=C_SILVER, activebackground='#D6D6D6',
            fg='#800000', font=("MS Sans Serif", 9, "bold"),
            relief='raised', bd=2, state='disabled',
        )
        self.stop_btn.pack(side='left', fill='x', expand=True, padx=3)
        _make_button(br, "FS", self.toggle_fullscreen, width=3).pack(side='left')

        self.chaos_var       = tk.DoubleVar(value=0.6)
        self.sensitivity_var = tk.DoubleVar(value=1.0)
        self.master_var      = tk.DoubleVar(value=1.0)
        self.cut_interval_var = tk.DoubleVar(value=0.3)
        _make_scale(cf, "Chaos:",       self.chaos_var,        0.0, 1.0, 0.05
                   ).pack(fill='x', padx=4, pady=1)
        _make_scale(cf, "Sensitivity:", self.sensitivity_var,  0.2, 5.0, 0.1
                   ).pack(fill='x', padx=4, pady=1)
        _make_scale(cf, "Master FX:",   self.master_var,       0.0, 1.0, 0.05
                   ).pack(fill='x', padx=4, pady=1)
        _make_scale(cf, "Cut Min s:",   self.cut_interval_var, 0.0, 2.0, 0.05
                   ).pack(fill='x', padx=4, pady=(1, 3))

        self.chaos_var.trace_add('write', self._sync_settings)
        self.sensitivity_var.trace_add('write', self._sync_settings)
        self.master_var.trace_add('write', self._sync_settings)
        self.cut_interval_var.trace_add('write', self._sync_settings)

        # ── Stage controls ────────────────────────────────────────────────────
        sc = tk.Frame(cf, bg=C_SILVER)
        sc.pack(fill='x', padx=4, pady=(0, 4))
        sc.columnconfigure(0, weight=1)
        sc.columnconfigure(1, weight=1)
        sc.columnconfigure(2, weight=1)

        self.blackout_btn = tk.Button(sc, text="BLACKOUT", font=("MS Sans Serif", 8, "bold"),
                                      bd=2, command=self._toggle_blackout)
        self.blackout_btn.grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        self._apply_fx_style(self.blackout_btn, False)

        self.freeze_btn = tk.Button(sc, text="FREEZE", font=("MS Sans Serif", 8, "bold"),
                                    bd=2, command=self._toggle_freeze)
        self.freeze_btn.grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        self._apply_fx_style(self.freeze_btn, False)

        _make_button(sc, "PANIC", self._panic).grid(
            row=0, column=2, sticky='ew', padx=2, pady=2)

        # Preset save / load
        pr = tk.Frame(cf, bg=C_SILVER)
        pr.pack(fill='x', padx=4, pady=(0, 4))
        _make_button(pr, "Save Preset", self.save_preset
                 ).pack(side='left', fill='x', expand=True)
        _make_button(pr, "Load Preset", self.load_preset
                 ).pack(side='left', fill='x', expand=True, padx=(3, 0))

        # ── Effects ───────────────────────────────────────────────────────────
        ef = tk.LabelFrame(parent, text="Effects", bg=C_SILVER, fg=C_TEXT_BLACK,
                           font=("MS Sans Serif", 9, "bold"), bd=2, relief='groove')
        ef.pack(fill='x', pady=(0, 3))

        grid = tk.Frame(ef, bg=C_SILVER)
        grid.pack(fill='x', padx=4, pady=(4, 2))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)

        _COLS = 3
        row = 0
        col = 0
        for i, (key, label) in enumerate(self._FX_DEFS):
            # Insert group separator label spanning all columns
            if i in self._FX_SEPARATORS:
                if col != 0:   # flush current row first
                    row += 1
                    col = 0
                tk.Label(grid, text=self._FX_SEPARATORS[i],
                         bg=C_SILVER, fg='#555555',
                         font='MS_Sans_Serif 7', anchor='w',
                         ).grid(row=row, column=0, columnspan=_COLS,
                                sticky='w', padx=4, pady=(3, 0))
                row += 1
            btn = tk.Button(grid, text=label, font='MS_Sans_Serif 8', bd=2,
                            command=lambda k=key: self._toggle_fx(k))
            btn.grid(row=row, column=col, sticky='ew', padx=2, pady=2)
            self._fx_btns[key] = btn
            self._apply_fx_style(btn, self._fx_state[key])
            col += 1
            if col >= _COLS:
                col = 0
                row += 1

        self.overlay_intensity_var = tk.DoubleVar(value=0.5)
        _make_scale(ef, "Ovl Freq:", self.overlay_intensity_var, 0.0, 1.0, 0.05
                   ).pack(fill='x', padx=4, pady=1)
        self.overlay_intensity_var.trace_add('write', self._sync_settings)

        # Chroma key section
        ck_lf = tk.LabelFrame(ef, text="Chroma Key", bg=C_SILVER, fg=C_TEXT_BLACK,
                               font='MS_Sans_Serif 8', bd=2, relief='groove')
        ck_lf.pack(fill='x', padx=4, pady=(2, 0))

        self.ck_mode_var = tk.StringVar(value='none')
        ttk.Combobox(ck_lf, values=['none', 'dominant', 'secondary', 'manual'],
                     textvariable=self.ck_mode_var, width=11,
                     state='readonly').pack(padx=4, pady=(3, 1))
        self.ck_mode_var.trace_add('write', self._sync_chroma)

        self.ck_tol_var = tk.DoubleVar(value=30.0)
        _make_scale(ck_lf, "Tolerance:", self.ck_tol_var, 5, 60, 1
                   ).pack(fill='x', padx=4, pady=1)
        self.ck_tol_var.trace_add('write', self._sync_chroma)

        self.ck_soft_var = tk.DoubleVar(value=5.0)
        _make_scale(ck_lf, "Softness:", self.ck_soft_var, 1, 21, 1
                   ).pack(fill='x', padx=4, pady=1)
        self.ck_soft_var.trace_add('write', self._sync_chroma)

        # Manual color RGB
        self.ck_r_var = tk.DoubleVar(value=0.0)
        self.ck_g_var = tk.DoubleVar(value=255.0)
        self.ck_b_var = tk.DoubleVar(value=0.0)
        _make_scale(ck_lf, "Key R:", self.ck_r_var, 0, 255, 1).pack(fill='x', padx=4, pady=1)
        _make_scale(ck_lf, "Key G:", self.ck_g_var, 0, 255, 1).pack(fill='x', padx=4, pady=1)
        _make_scale(ck_lf, "Key B:", self.ck_b_var, 0, 255, 1).pack(fill='x', padx=4, pady=(1, 3))
        self.ck_r_var.trace_add('write', self._sync_chroma)
        self.ck_g_var.trace_add('write', self._sync_chroma)
        self.ck_b_var.trace_add('write', self._sync_chroma)

        bot = tk.Frame(ef, bg=C_SILVER)
        bot.pack(fill='x', padx=4, pady=(1, 4))
        self.seq_btn = tk.Button(bot, text="Sequential", font='MS_Sans_Serif 9',
                                 bd=2, command=self._toggle_sequential)
        self.seq_btn.pack(side='left', fill='x', expand=True)
        self._apply_fx_style(self.seq_btn, False)  # starts OFF
        _make_button(bot, "Recalibrate", self._recalibrate
                 ).pack(side='left', fill='x', expand=True, padx=(3, 0))

    def _build_right(self, parent):
        # Video output — label updated on resolution change
        res_label = f"Live Output ({self.resolution_var.get()})"
        vf = tk.LabelFrame(parent, text=res_label, bg=C_SILVER,
                           fg=C_TEXT_BLACK, font=("MS Sans Serif", 9, "bold"),
                           bd=2, relief='groove')
        vf.pack(fill='both', expand=True)
        vf.pack_propagate(False)
        self.video_label = tk.Label(vf, text="Load video and press START",
                                    bg=C_BLACK, fg=C_WHITE,
                                    font=("MS Sans Serif", 10))
        self.video_label.pack(fill='both', expand=True, padx=3, pady=3)

        # Indicator strip (two rows)
        ind = tk.Frame(parent, bg=C_SILVER, bd=1, relief='sunken')
        ind.pack(fill='x', pady=(3, 0))

        r0 = tk.Frame(ind, bg=C_SILVER)
        r0.pack(fill='x', padx=4, pady=(2, 1))
        # Use grid so labels share width proportionally and never overflow
        r0.columnconfigure(2, weight=1)  # GATE label expands, others fixed

        self.beat_indicator = tk.Label(r0, text="BEAT", bg=C_SILVER,
                                       fg=C_DARK_GRAY,
                                       font=("MS Sans Serif", 9, "bold"))
        self.beat_indicator.grid(row=0, column=0, sticky='w', padx=(0, 6))

        self.rms_indicator = tk.Label(r0, text="RMS: 0.0000", bg=C_SILVER,
                                      font='MS_Sans_Serif 9')
        self.rms_indicator.grid(row=0, column=1, sticky='w', padx=(0, 6))

        self.gate_indicator = tk.Label(r0, text="CALIBRATING...",
                                       bg=C_SILVER, fg='#AA6600',
                                       font='MS_Sans_Serif 9', anchor='w')
        self.gate_indicator.grid(row=0, column=2, sticky='ew')

        self.fps_indicator = tk.Label(r0, text="FPS: --", bg=C_SILVER,
                                      font='MS_Sans_Serif 9', anchor='e')
        self.fps_indicator.grid(row=0, column=3, sticky='e', padx=(6, 0))

        # Spectral bars — label+canvas pairs in a single grid row.
        # Even columns = labels (fixed), odd columns = canvases (expand).
        r1 = tk.Frame(ind, bg=C_SILVER)
        r1.pack(fill='x', padx=4, pady=(0, 3))
        for i in range(4):
            r1.columnconfigure(i * 2 + 1, weight=1)  # canvas cols expand

        for i, (label, color, attr) in enumerate([
            ("B", '#0000CC', 'bar_bass'),
            ("M", '#008800', 'bar_mid'),
            ("T", '#AA0000', 'bar_treble'),
            ("F", '#AA8800', 'bar_flat'),
        ]):
            lc = i * 2      # label column
            cc = i * 2 + 1  # canvas column
            tk.Label(r1, text=label, bg=C_SILVER, fg=color,
                     font='MS_Sans_Serif 8').grid(
                         row=0, column=lc, sticky='w', padx=(4 if i else 0, 2))
            c = tk.Canvas(r1, height=8, bg='#808080',
                          highlightthickness=0, bd=1, relief='sunken')
            c.grid(row=0, column=cc, sticky='ew',
                   padx=(0, 6 if i < 3 else 0))
            setattr(self, attr, c)

        # Console
        con = tk.LabelFrame(parent, text="Console", bg=C_SILVER, fg=C_TEXT_BLACK,
                            font=("MS Sans Serif", 9, "bold"), bd=2, relief='groove')
        con.pack(fill='x', pady=(3, 0))
        self.console = tk.Text(con, height=5, font=("Courier New", 8),
                               bg=C_WHITE, fg=C_BLACK, bd=2, relief='sunken',
                               state='disabled')
        self.console.pack(fill='x', padx=4, pady=3)
        self._log("System ready.")

    # ── Toggle helpers ────────────────────────────────────────────────────────

    def _apply_fx_style(self, btn: tk.Button, active: bool):
        if active:
            btn.config(relief='sunken', bg='#000080', fg='#FFFFFF',
                       activebackground='#0000aa', activeforeground='#FFFFFF')
        else:
            btn.config(relief='raised', bg=C_SILVER, fg=C_TEXT_BLACK,
                       activebackground='#D6D6D6', activeforeground=C_TEXT_BLACK)

    def _toggle_fx(self, key: str):
        self._fx_state[key] = not self._fx_state[key]
        self._apply_fx_style(self._fx_btns[key], self._fx_state[key])
        self._sync_settings()

    def _toggle_sequential(self):
        self._sequential_on = not self._sequential_on
        self._apply_fx_style(self.seq_btn, self._sequential_on)
        self._sync_settings()

    def _toggle_blackout(self):
        self._blackout_on = not self._blackout_on
        self.engine.blackout = self._blackout_on
        self._apply_fx_style(self.blackout_btn, self._blackout_on)
        self._log("BLACKOUT ON" if self._blackout_on else "Blackout off")

    def _toggle_freeze(self):
        self._freeze_on = not self._freeze_on
        self.engine.freeze = self._freeze_on
        self._apply_fx_style(self.freeze_btn, self._freeze_on)
        self._log("FREEZE ON" if self._freeze_on else "Freeze off")

    def _panic(self):
        """Reset all effect buffers instantly — use when visuals get stuck/corrupt."""
        self.engine.reset_effects()
        # Turn off blackout/freeze if on
        if self._blackout_on:
            self._toggle_blackout()
        if self._freeze_on:
            self._toggle_freeze()
        self._log("PANIC — all effect buffers cleared")

    def _kb_blackout(self, event=None):
        self._toggle_blackout()

    def _kb_freeze(self, event=None):
        self._toggle_freeze()

    def _kb_panic(self, event=None):
        self._panic()

    def _kb_toggle_fx(self, index: int):
        """Toggle effect by index in _FX_DEFS (keyboard shortcut)."""
        if 0 <= index < len(self._FX_DEFS):
            key = self._FX_DEFS[index][0]
            self._toggle_fx(key)

    # ── Settings sync ─────────────────────────────────────────────────────────

    def _sync_chroma(self, *_):
        self.engine.overlay_mgr.set_chroma_key(
            mode=self.ck_mode_var.get(),
            tolerance=int(self.ck_tol_var.get()),
            softness=int(self.ck_soft_var.get()),
            color=(int(self.ck_r_var.get()),
                   int(self.ck_g_var.get()),
                   int(self.ck_b_var.get())),
        )

    def _sync_settings(self, *_):
        self.engine.settings.update({
            'chaos':             self.chaos_var.get(),
            'master_intensity':  self.master_var.get(),
            'min_cut_interval':  self.cut_interval_var.get(),
            'overlay_intensity': self.overlay_intensity_var.get(),
            'sequential_mode':   self._sequential_on,
            **self._fx_state,
        })
        self.engine.audio.gate_multiplier = self.sensitivity_var.get()

    def _recalibrate(self):
        a = self.engine.audio
        a._gate = a.NOISE_FLOOR
        a._calibration_buf.clear()
        a._chunk_count = 0
        a._calibration_done = False
        a._last_rms = 0.0
        a._rms_history.clear()
        self.gate_indicator.config(text="CALIBRATING...", fg='#AA6600')
        self._log("Recalibrating noise floor...")

    def _on_device_change(self, event=None):
        """Hot-swap audio device while running."""
        if not self.is_running:
            return
        device_str = self.audio_combo.get()
        try:
            device_idx = int(device_str.split(":")[0])
        except Exception:
            device_idx = None
        a = self.engine.audio
        a.stop()
        a._gate = a.NOISE_FLOOR
        a._calibration_buf.clear()
        a._chunk_count = 0
        a._calibration_done = False
        a._last_rms = 0.0
        a._rms_history.clear()
        a.start(device_idx)
        self.gate_indicator.config(text="CALIBRATING...", fg='#AA6600')
        self._log(f"Audio device changed — recalibrating...")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        self.console.config(state='normal')
        self.console.insert('end', f"[{ts}] {message}\n")
        self.console.see('end')
        self.console.config(state='disabled')

    def log(self, message: str):
        """Public alias for compatibility."""
        self._log(message)

    # ── Device / file loading ─────────────────────────────────────────────────

    def load_audio_devices(self):
        try:
            devices = self.engine.audio.get_audio_devices()
            if devices:
                device_list = [f"{i}: {name}" for i, name in devices]
                self.audio_combo['values'] = device_list
                self.audio_combo.current(0)
                self._log(f"Found {len(devices)} audio devices")
                self.status_bar.config(text=f"Found {len(devices)} audio devices")
            else:
                self.audio_combo['values'] = ["No audio devices found"]
                self._log("Warning: No audio input devices found")
                self.status_bar.config(text="Warning: No audio input devices found")
        except Exception as e:
            self._log(f"Audio device error: {e}")

    def load_video(self):
        paths = filedialog.askopenfilenames(
            title="Select Video Source(s)",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.mpg *.mpeg")])
        if not paths:
            return
        try:
            paths = list(paths)
            if self.engine.set_video_sources(paths):
                count = len(paths)
                if count == 1:
                    name = os.path.basename(paths[0])
                else:
                    name = f"{count} videos loaded"
                self.file_info_label.config(text=name, fg=C_TEXT_BLACK)
                self._log(f"Video loaded: {name}")
                self.status_bar.config(text=f"Video: {name}")
            else:
                messagebox.showerror("Error", "Failed to load video file(s)")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._log(f"Video error: {e}")

    def load_overlays(self):
        folder = filedialog.askdirectory(title="Select Overlays Folder")
        if not folder:
            return
        try:
            self.engine.load_overlays(folder)
            count = len(self.engine.overlay_mgr.overlays)
            self.file_info_label.config(
                text=f"video | {count} overlays", fg=C_TEXT_BLACK)
            self._log(f"Loaded {count} overlays from {os.path.basename(folder)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._log(f"Overlay error: {e}")

    # ── Engine start / stop ───────────────────────────────────────────────────

    def save_preset(self):
        path = filedialog.asksaveasfilename(
            title="Save Preset",
            defaultextension='.json',
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        data = {
            'fx_state':        self._fx_state,
            'chaos':           self.chaos_var.get(),
            'sensitivity':     self.sensitivity_var.get(),
            'master_intensity':self.master_var.get(),
            'cut_interval':    self.cut_interval_var.get(),
            'overlay_intensity':self.overlay_intensity_var.get(),
            'sequential':      self._sequential_on,
            'resolution':      self.resolution_var.get(),
            'ck_mode':         self.ck_mode_var.get(),
            'ck_tolerance':    self.ck_tol_var.get(),
            'ck_softness':     self.ck_soft_var.get(),
            'ck_r':            self.ck_r_var.get(),
            'ck_g':            self.ck_g_var.get(),
            'ck_b':            self.ck_b_var.get(),
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self._log(f"Preset saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save preset: {e}")

    def load_preset(self):
        path = filedialog.askopenfilename(
            title="Load Preset",
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load preset: {e}")
            return

        if 'fx_state' in data:
            for k, v in data['fx_state'].items():
                if k in self._fx_state:
                    self._fx_state[k] = bool(v)
                    if k in self._fx_btns:
                        self._apply_fx_style(self._fx_btns[k], self._fx_state[k])
        if 'chaos'            in data: self.chaos_var.set(data['chaos'])
        if 'sensitivity'      in data: self.sensitivity_var.set(data['sensitivity'])
        if 'master_intensity' in data: self.master_var.set(data['master_intensity'])
        if 'cut_interval'     in data: self.cut_interval_var.set(data['cut_interval'])
        if 'overlay_intensity'in data: self.overlay_intensity_var.set(data['overlay_intensity'])
        if 'sequential'       in data:
            if bool(data['sequential']) != self._sequential_on:
                self._toggle_sequential()
        if 'resolution'       in data: self.resolution_var.set(data['resolution'])
        if 'ck_mode'          in data: self.ck_mode_var.set(data['ck_mode'])
        if 'ck_tolerance'     in data: self.ck_tol_var.set(data['ck_tolerance'])
        if 'ck_softness'      in data: self.ck_soft_var.set(data['ck_softness'])
        if 'ck_r'             in data: self.ck_r_var.set(data['ck_r'])
        if 'ck_g'             in data: self.ck_g_var.set(data['ck_g'])
        if 'ck_b'             in data: self.ck_b_var.set(data['ck_b'])

        self._sync_settings()
        self._log(f"Preset loaded: {os.path.basename(path)}")

    def start_engine(self):
        if not self.engine.video_pool.loaded:
            messagebox.showwarning("Warning", "Please load a video file first")
            return

        device_str = self.audio_combo.get()
        if not device_str or "No audio" in device_str:
            messagebox.showwarning("Warning", "Please select an audio input device")
            return

        try:
            device_idx = int(device_str.split(":")[0])
        except Exception:
            device_idx = None

        # Apply selected resolution before starting
        res_str = self.resolution_var.get()
        w, h = self._RES_MAP.get(res_str, (640, 360))
        self.engine.set_resolution(w, h)

        self._sync_settings()
        success, message = self.engine.start(device_idx)
        if not success:
            messagebox.showerror("Error", message)
            self._log(f"Start failed: {message}")
            return

        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self._log("Engine started — Live audio reactive video")
        self.status_bar.config(text="LIVE — Audio reactive video running")

        self.last_update_time = time.time()
        self.frame_count = 0
        self.update_video()

    def stop_engine(self):
        self.is_running = False
        if self.update_id:
            self.after_cancel(self.update_id)
            self.update_id = None
        self.engine.stop()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.beat_indicator.config(fg=C_DARK_GRAY)
        # Reset stage controls
        if self._blackout_on:
            self._blackout_on = False
            self.engine.blackout = False
            self._apply_fx_style(self.blackout_btn, False)
        if self._freeze_on:
            self._freeze_on = False
            self.engine.freeze = False
            self._apply_fx_style(self.freeze_btn, False)
        self._log("Engine stopped")
        self.status_bar.config(text="Stopped")
        self.exit_fullscreen()
        self.close_second_monitor()

    # ── Frame update loop ─────────────────────────────────────────────────────

    def update_video(self):
        if not self.is_running:
            return
        try:
            t0 = time.time()
            frame = self.engine.process_frame()

            if frame is not None:
                # Read size from the parent LabelFrame to avoid feedback-loop growth
                parent = self.video_label.master
                lw = parent.winfo_width() - 6
                lh = parent.winfo_height() - 6
                if lw > 10 and lh > 10:
                    img = Image.fromarray(frame).resize((lw, lh), Image.LANCZOS)
                else:
                    img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

                if self.fullscreen_window:
                    try:
                        fw = self.fullscreen_window.winfo_width()
                        fh = self.fullscreen_window.winfo_height()
                        if fw < 10: fw = self.fullscreen_window.winfo_screenwidth()
                        if fh < 10: fh = self.fullscreen_window.winfo_screenheight()
                        fs_img = ImageTk.PhotoImage(
                            image=Image.fromarray(cv2.resize(frame, (fw, fh))))
                        self.fs_label.imgtk = fs_img
                        self.fs_label.config(image=fs_img)
                    except Exception as _e:
                        print(f"[fullscreen render] {_e}")

                if self.second_monitor_window:
                    try:
                        sw = self.second_monitor_window.winfo_width()
                        sh = self.second_monitor_window.winfo_height()
                        if sw < 10: sw = self.winfo_screenwidth()
                        if sh < 10: sh = self.winfo_screenheight()
                        sm_img = ImageTk.PhotoImage(
                            image=Image.fromarray(cv2.resize(frame, (sw, sh))))
                        self.sm_label.imgtk = sm_img
                        self.sm_label.config(image=sm_img)
                    except Exception as _e:
                        print(f"[second monitor render] {_e}")

                self._update_indicators()

            self.frame_count += 1
            now = time.time()
            if now - self.last_update_time >= 1.0:
                fps = self.frame_count / (now - self.last_update_time)
                self.fps_indicator.config(text=f"FPS: {fps:.1f}")
                self.last_update_time = now
                self.frame_count = 0

            elapsed_ms = (time.time() - t0) * 1000
            self.update_id = self.after(max(1, int(33 - elapsed_ms)), self.update_video)

        except Exception as e:
            self._log(f"Update error: {e}")
            self.stop_engine()

    # ── Indicators ────────────────────────────────────────────────────────────

    def _draw_bar(self, canvas, value, color):
        canvas.delete('all')
        w = canvas.winfo_width() or 160
        fill_w = int(min(max(value, 0.0), 1.0) * w)
        if fill_w > 0:
            canvas.create_rectangle(0, 0, fill_w, canvas.winfo_height() or 8,
                                    fill=color, outline='')

    def _update_indicators(self):
        try:
            if self.engine.audio_stats['beat']:
                self.beat_indicator.config(fg='red')
                self.after(100, lambda: self.beat_indicator.config(fg=C_DARK_GRAY))

            rms = self.engine.audio_stats['rms']
            self.rms_indicator.config(text=f"RMS: {rms:.4f}")

            a = self.engine.audio
            if not a._calibration_done:
                remaining = a.CALIBRATION_CHUNKS - a._chunk_count
                self.gate_indicator.config(
                    text=f"CALIBRATING ({remaining})...", fg='#AA6600')
            else:
                eg = a.effective_gate
                self.gate_indicator.config(
                    text=f"GATE: {eg:.4f}",
                    fg='#006600' if rms > eg else C_TEXT_BLACK)

            active = rms > self.engine.audio.effective_gate
            if active:
                for k in ('bass', 'mid', 'treble'):
                    v = self.engine.audio_stats.get(k, 0.0)
                    if v > self._band_max[k]:
                        self._band_max[k] = v

            if active:
                bass = self.engine.audio_stats.get('bass',   0.0) / self._band_max['bass']
                mid  = self.engine.audio_stats.get('mid',    0.0) / self._band_max['mid']
                trbl = self.engine.audio_stats.get('treble', 0.0) / self._band_max['treble']
                flat = min(self.engine.audio_stats.get('flatness', 0.0), 1.0)
            else:
                bass = mid = trbl = flat = 0.0

            self._draw_bar(self.bar_bass,   bass, '#0000CC')
            self._draw_bar(self.bar_mid,    mid,  '#008800')
            self._draw_bar(self.bar_treble, trbl, '#AA0000')
            self._draw_bar(self.bar_flat,   flat, '#AA8800')
        except Exception:
            pass

    # ── Fullscreen / second monitor ───────────────────────────────────────────

    def toggle_fullscreen(self, event=None):
        if not self.is_running:
            messagebox.showinfo("Info", "Start the engine first")
            return
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def enter_fullscreen(self):
        try:
            self.fullscreen_window = tk.Toplevel(self)
            self.fullscreen_window.title("Disc VPC 01 R-T")
            self.fullscreen_window.configure(bg=C_BLACK)
            self.fullscreen_window.attributes('-fullscreen', True)
            self.fullscreen_window.attributes('-topmost', True)
            self.fs_label = tk.Label(self.fullscreen_window, bg=C_BLACK)
            self.fs_label.pack(expand=True, fill='both')
            tk.Button(self.fullscreen_window, text="ESC to Exit",
                      command=self.exit_fullscreen,
                      bg=C_DARK_GRAY, fg=C_WHITE,
                      font=("MS Sans Serif", 8)).place(x=10, y=10)
            self.is_fullscreen = True
            self.status_bar.config(text="Fullscreen active (ESC to exit)")
        except Exception as e:
            self._log(f"Fullscreen error: {e}")

    def exit_fullscreen(self, event=None):
        if self.fullscreen_window:
            try:
                self.fullscreen_window.destroy()
            except Exception:
                pass
            self.fullscreen_window = None
        self.is_fullscreen = False

    def toggle_second_monitor(self, event=None):
        if not self.is_running:
            messagebox.showinfo("Info", "Start the engine first")
            return
        if self.second_monitor_window:
            self.close_second_monitor()
        else:
            self.open_second_monitor()

    def open_second_monitor(self):
        try:
            self.second_monitor_window = tk.Toplevel(self)
            self.second_monitor_window.title("Disc VPC 01 R-T")
            self.second_monitor_window.configure(bg=C_BLACK)
            sw = self.winfo_screenwidth()
            x = 1920 if sw > 1920 else sw - 800
            self.second_monitor_window.geometry(f"800x600+{x}+100")
            self.second_monitor_window.attributes('-topmost', True)
            self.sm_label = tk.Label(self.second_monitor_window, bg=C_BLACK)
            self.sm_label.pack(expand=True, fill='both')
            self._log("Second monitor output started")
            self.status_bar.config(text="Second monitor active")
        except Exception as e:
            self._log(f"Second monitor error: {e}")
            messagebox.showwarning("Warning", "Could not open second monitor window")

    def close_second_monitor(self):
        if self.second_monitor_window:
            try:
                self.second_monitor_window.destroy()
            except Exception:
                pass
            self.second_monitor_window = None

    def on_closing(self):
        self.stop_engine()
        self.destroy()
