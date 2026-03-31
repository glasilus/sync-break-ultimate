# Disc VPC 01 — User Manual

**Version:** 0.1.2
**Requires:** Python 3.10.x · Windows 10/11 (macOS, Ubuntu 22.* best-effort)

---

## Table of Contents

- [Disc VPC 01 — User Manual](#disc-vpc-01--user-manual)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. Installation](#2-installation)
    - [From source](#from-source)
    - [Launch](#launch)
    - [Pre-built executables](#pre-built-executables)
  - [3. Disc VPC 01 — Offline Editor](#3-disc-vpc-01--offline-editor)
    - [3.1 Interface Layout](#31-interface-layout)
    - [3.2 Loading Source Files](#32-loading-source-files)
    - [3.3 Audio Analysis Engine](#33-audio-analysis-engine)
    - [3.4 Cut Logic](#34-cut-logic)
    - [3.5 Effects — Core](#35-effects--core)
    - [3.6 Effects — Glitch](#36-effects--glitch)
    - [3.7 Effects — Complex](#37-effects--complex)
    - [3.8 Effects — Degradation](#38-effects--degradation)
    - [3.9 Effects — Overlays and Chroma Key](#39-effects--overlays-and-chroma-key)
    - [Chroma Key (Green Screen)](#chroma-key-green-screen)
    - [3.10 Effects — Signal Domain](#310-effects--signal-domain)
    - [3.11 Export Settings](#311-export-settings)
    - [3.12 Presets](#312-presets)
    - [3.13 Render Modes](#313-render-modes)
    - [3.14 Mystery Panel](#314-mystery-panel)
    - [3.15 Multiple Video Sources](#315-multiple-video-sources)
    - [3.16 Silence Treatment](#316-silence-treatment)
  - [4. Disc VPC 01 R-T — Real-Time](#4-disc-vpc-01-r-t--real-time)
    - [4.1 Interface Layout](#41-interface-layout)
    - [4.2 Setup](#42-setup)
    - [4.3 Audio Calibration](#43-audio-calibration)
    - [4.4 Controls and Parameters](#44-controls-and-parameters)
    - [4.5 Effects](#45-effects)
      - [Core](#core)
      - [Glitch](#glitch)
      - [Spatial / Color](#spatial--color)
    - [4.6 Indicators](#46-indicators)
    - [4.7 Fullscreen and Second Monitor](#47-fullscreen-and-second-monitor)
    - [4.8 Keyboard Shortcuts](#48-keyboard-shortcuts)
  - [5. Segment Types Reference](#5-segment-types-reference)
  - [6. Recommended Workflows](#6-recommended-workflows)
    - [Disc VPC 01 — Iterative editing](#disc-vpc-01--iterative-editing)
    - [Disc VPC 01 R-T — Live performance](#disc-vpc-01-r-t--live-performance)
  - [7. Troubleshooting](#7-troubleshooting)

---

## 1. Overview

**Disc VPC 01** is an offline algorithmic video montage tool. It analyzes an audio file, divides it into typed segments, and assembles a new video from a source clip by applying audio-reactive cuts and visual effects. The result is exported as a video file.

**Disc VPC 01 R-T** is the real-time counterpart. It listens to a live audio input (microphone, line-in, or system loopback) and drives video playback and effects in real time, frame by frame. Intended for live performance and VJ use.

Both tools share the same audio analysis core and the same effects library.

---

## 2. Installation

### From source

```
Python 3.10.x is required. Python 3.11+ may cause dependency errors.
```

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Purpose |
|---|---|
| `librosa` | Audio analysis, onset detection |
| `opencv-contrib-python` | Video I/O and image processing |
| `scipy` / `numpy` | Signal processing |
| `sounddevice` | Real-time audio capture (R-T only) |
| `scenedetect` | Scene boundary detection |
| `opensimplex` | Noise functions for effects |
| `Pillow` | Image manipulation |
| `imageio-ffmpeg` | FFmpeg backend for video export |

### Launch

```bash
# Offline editor
python gui.py

# Real-time
python rt.py
```

### Pre-built executables

Pre-built `.exe` bundles for Windows are attached to each release on GitHub.
The bundle contains:

```
Disc-VPC-01-windows/
├── DiscVPC01.exe            ← offline editor
├── DiscVPC01-RT.exe         ← real-time
├── presets/
│   └── rt_blank.json        ← blank preset for R-T (load via Load Preset)
├── README.MD                ← this document (English)
└── README_RU.MD             ← this document (Russian)
```

No installation required. Run directly.

---

## 3. Disc VPC 01 — Offline Editor

### 3.1 Interface Layout

The window is divided into three vertical panels:

| Panel | Content |
|---|---|
| **Left (Sidebar)** | Source file loading, presets, render buttons |
| **Center** | Effects tabs: EFFECTS / EXPORT / MYSTERY |
| **Right** | Live preview monitor, progress bar, status log |

### 3.2 Loading Source Files

Before rendering, two files must be loaded:

**Load Audio (WAV / MP3)** — the audio track that drives the entire analysis. The engine reads this file in full before any rendering begins. Supported formats: `.wav`, `.mp3`, and any format readable by `librosa`.

**Load Source Video** — the video clip that will be cut and processed. The engine reads random or scene-detected frames from this clip to assemble the output. Any resolution is accepted; the output resolution is determined by the export settings.

A green indicator dot (●) next to each button confirms the file has been loaded successfully.

Optionally, an **overlay folder** can be loaded via the Overlays tab (see §3.9).

### 3.3 Audio Analysis Engine

When a render is initiated, the engine performs a full analysis of the audio track using `librosa`:

1. **Onset detection** — identifies transient attacks in the waveform (drum hits, note onsets, etc.)
2. **Beat tracking** — extracts the beat grid for snap-to-beat alignment
3. **RMS computation** — calculates root-mean-square energy for each segment
4. **Spectral flatness** — measures how "noise-like" the spectrum is (white noise = 1.0, pure tone ≈ 0)
5. **Segment classification** — assigns each segment one of six types (see §5)

This analysis runs once per session per audio file. Results are cached in memory for repeated renders.

### 3.4 Cut Logic

The **Cut Logic** section (first accordion group in the EFFECTS tab) controls how the engine selects and sequences video clips.

---

**Smart Scene Detection** *(on/off)*

When enabled, the engine uses `PySceneDetect` to find scene boundaries in the source video before rendering. Clips are then selected starting from detected scene changes rather than random frames, producing more visually coherent transitions.

Recommended for source material with distinct visual scenes (films, footage reels). Disable for abstract or repetitive source video.

---

**Global Chaos Level** *(0.0 – 1.0)*

Controls the overall randomness of clip selection. At `0.0`, clips are chosen predictably based on scene detection or sequential order. At `1.0`, every cut selects a fully random frame regardless of scene boundaries or sequential position. Also scales the probability of most effects firing.

---

**Beat Threshold** *(0.5 – 2.0)*

The multiplier applied to the mean RMS to determine whether a segment is "loud." A segment is loud if:

```
segment_rms > rms_mean × threshold
```

Lower values → more segments classified as loud → more cuts and effects. Higher values → only the loudest peaks trigger reactions.

**Recommended starting values:**
- Quiet or dynamic material: `0.9 – 1.1`
- Dense/compressed material: `1.3 – 1.6`

---

**Transient Sensitivity** *(0.1 – 1.5)*

Controls the threshold for detecting a sharp attack (rapid RMS rise). Used to classify IMPACT segments. Lower values detect more transients; higher values only respond to very abrupt hits.

---

**Min Cut Duration (sec)** *(0.0 – 0.3)*

Segments shorter than this value are merged with the following segment and do not produce a cut. Prevents micro-cuts and flickering on dense audio. Typical useful range: `0.04 – 0.1`.

---

**Scene Buffer Size** *(2 – 30)*

The number of unique scene timestamps held in memory for random selection during high-chaos moments. A larger buffer allows more variety; a smaller buffer causes the edit to cycle through fewer scenes more rapidly.

---

**Snap Cuts to Beat Grid** *(on/off)*

When enabled, each detected onset is snapped to the nearest beat-grid position if it falls within the **Beat Snap Tolerance** window. This aligns cuts more tightly to the rhythmic pulse of the track.

**Beat Snap Tolerance** *(0.01 – 0.15 sec)* — the maximum distance an onset can be moved to snap to a beat. If the onset is farther than this from the nearest beat, it is left in place.

---

### 3.5 Effects — Core

Core effects are the primary visual transformations applied to each video frame. Each effect has an on/off toggle and, where applicable, a **chance** slider that controls the probability of the effect firing on any given eligible frame.

---

**Stutter / Drill**

*Trigger: IMPACT segments only*

Repeats the first micro-segment of an impact 2, 4, or 8 times in rapid succession, creating a drill or stutter effect. There is no chance slider; the effect fires on every IMPACT if enabled.

---

**Flash Frame**

*Trigger: DROP, IMPACT*

Inserts a brief white or black frame (or a blend toward white/black) at moments of sharp energy change. The intensity of the flash scales with the segment's intensity value.

**Chance** *(0.0 – 1.0)* — probability of firing on each eligible segment.

---

**Ghost Trails**

*Trigger: SUSTAIN, BUILD*

Blends the current frame with the previous frame, creating a motion trail or ghosting effect. Suitable for slow builds and sustained sections.

**Opacity** *(0.0 – 1.0)* — how strongly the previous frame bleeds through. At `0.0` the effect is invisible; at `1.0` the image is nearly frozen.

---

**Pixel Sort**

*Trigger: NOISE, IMPACT*

Sorts columns of pixels within horizontal strips by their luminance, hue, or saturation value. Produces a characteristic glitch aesthetic that resembles codec failure or data corruption.

**Chance** *(0.0 – 1.0)* — probability per eligible segment.

**Pixel Sort Intensity** *(0.0 – 1.0)* — controls the height of the strips being sorted. At low values only thin bands are affected; at high values nearly the entire frame is sorted.

**Sort Axis** — the pixel property used as the sort key:
- `luminance` — sorts by perceived brightness (default, most legible)
- `hue` — sorts by colour angle in HSV space (more colourful result)
- `saturation` — sorts by colour purity

---

**Datamosh**

*Trigger: NOISE*

In **Draft** and **Preview** modes: simulates the P-frame bleed effect using optical flow — the previous frame's motion vectors are applied to warp the image, approximating datamosh compression artefacts.

In **Final render** mode: actual I-frame dropping is applied at the video codec level, producing authentic datamosh output.

**Chance** *(0.0 – 1.0)* — probability per eligible segment.

---

**ASCII Filter**

*Trigger: SUSTAIN*

Converts the frame to ASCII art rendered back as an image. Each character represents a block of pixels, with the character density mapping to the luminance of that block.

**Char Size (px)** *(4 – 40)* — the size of each character block in pixels. Smaller values produce finer detail; larger values give a more blocky, abstract look.

**Blend (0 – 1)** — composites the ASCII output over the original frame. At `0.0` the output is pure ASCII. At `1.0` the original frame is shown unchanged (effect invisible). Intermediate values overlay ASCII onto the source image.

**ASCII Color Mode:**
- `fixed` — all characters drawn in the configured FG color on the configured BG color
- `original` — each character takes the average color of the source pixels it represents
- `inverted` — same as `original` but with colors inverted

**ASCII FG/BG Color** — RGB sliders (0–255) for foreground and background in `fixed` mode.

---

### 3.6 Effects — Glitch

These effects simulate signal degradation, codec artefacts, and transmission errors.

| Effect | Trigger | Description |
|---|---|---|
| **RGB Shift** | IMPACT, BUILD | Shifts the red channel right and blue channel left by a pixel offset proportional to segment intensity. Produces chromatic aberration. |
| **Block Glitch** | IMPACT, DROP | Selects random rectangular blocks and either replaces them with garbage pixels or copies them from a random position. |
| **Pixel Drift** | NOISE, IMPACT | Displaces entire rows of pixels sideways by random offsets, simulating horizontal sync loss. |
| **Color Bleed / VHS Smear** | NOISE | Bleeds colour channels horizontally past edges, mimicking analog chroma delay. |
| **Freeze + Corrupt** | DROP | Freezes the current frame and applies random pixel corruption, as though the video stream has stalled. |
| **Negative** | IMPACT | Inverts all colour values (`255 - value`) for one frame. |

Each effect has a **Chance** slider controlling the probability of firing on an eligible segment.

---

### 3.7 Effects — Complex

These effects involve multi-frame state or more computationally involved transforms.

---

**Feedback Loop**

*Trigger: SUSTAIN, BUILD — fires on every eligible frame when enabled, no chance*

Blends the current frame with a decayed copy of itself accumulated over previous frames, creating a video feedback tunnel effect. The accumulation reinforces strong signals and fades quiet ones.

---

**Phase Shift** *(Chance)*

Divides the frame into horizontal bands and shifts alternating bands left and right by increasing offsets, creating a vertically layered displacement.

---

**Mosaic Pulse** *(Chance)*

Pixelates the frame by downscaling and upscaling. The block size is driven by the bass RMS value, so the pixelation pulses with low-frequency energy.

---

**Echo Compound** *(Chance)*

Composites multiple copies of the frame onto itself with hue-shifted colour offsets and slight spatial displacement, creating a layered chromatic echo.

---

**Kali Mirror** *(Chance)*

Applies a kaleidoscopic mirror transform, reflecting and rotating quadrants of the frame to create symmetric geometric patterns.

---

**Glitch Cascade** *(Chance)*

Chains several glitch effects in sequence on a single frame, compounding their results. The exact chain is randomized per frame.

---

### 3.8 Effects — Degradation

Degradation effects simulate worn or low-quality recording/playback media.

| Effect | Trigger | Description |
|---|---|---|
| **Scan Lines** | SUSTAIN, NOISE | Draws horizontal dark lines at every other row, simulating a CRT monitor. |
| **Bitcrush / Posterize** | Any | Reduces colour depth by quantizing pixel values to a smaller number of levels. |
| **JPEG Crush** | IMPACT, NOISE | Re-encodes the frame at very low JPEG quality and decodes it back, producing compression block artefacts. |
| **Fisheye / Barrel** | BUILD | Applies barrel lens distortion, bowing the image outward from the center. |
| **VHS Tracking** | NOISE, DROP | Simulates VHS tape tracking errors: horizontal displacement at the top and bottom of the frame with luminance noise. |
| **Interlace** | SUSTAIN | Splits even and odd scan lines and offsets them by a few pixels, simulating interlaced video. |
| **Bad Signal** | DROP, NOISE | Overlays digital signal breakup artefacts — horizontal stripes, colour shifts, and noise bands. |
| **Dithering** | SILENCE, SUSTAIN | Applies an ordered dither pattern, reducing the effective colour palette while adding a structured noise texture. |
| **Zoom Glitch** | IMPACT | Applies a rapid random zoom and crop for a single frame, creating a sudden jump-cut-within-a-frame pulse. |

Each effect has a **Chance** slider.

---

### 3.9 Effects — Overlays and Chroma Key

Overlays composite external image files onto video frames. Load an overlay folder via **Load Source Video → Load Overlays folder** (sidebar, below the video button), or through the Overlays tab.

**Enable Overlays** *(on/off + Chance)*

The engine selects a random image from the folder on each eligible frame and composites it onto the video.

**Opacity** *(0.0 – 1.0)* — transparency of the overlay image.

**Scale Max** *(0.05 – 1.0)* — maximum size of the overlay as a fraction of the frame dimensions.

**Scale Min** *(0.05 – 1.0)* — minimum size. The engine selects a random scale between Min and Max for each occurrence.

**Blend Mode:**
- `screen` — lightens the image (good for glows, light leaks, textures)
- `normal` — standard alpha composite over the frame
- `multiply` — darkens the image (good for grunge, film grain)

**Position:**
- `random` — the overlay is placed at a random position within the frame bounds
- `center` — the overlay is centered on the frame
- `corners` — the overlay cycles through corner positions

Supported overlay formats: `.png`, `.jpg`, `.gif` (first frame only), `.bmp`, `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.mpg`, `.mpeg`

Video files are loaded frame-by-frame; each frame becomes an independent overlay image in the pool.

---

### Chroma Key (Green Screen)

Removes a background colour from each overlay image before compositing, making that colour transparent.

**Mode:**
- `none` — chroma key disabled (default)
- `dominant` — automatically detects and removes the most frequent colour in the overlay (e.g. green screen)
- `secondary` — removes the second most frequent colour
- `manual` — removes the colour specified by the R/G/B sliders below

**Tolerance** *(5 – 60)* — how wide a hue range around the key colour is keyed out. Increase if green screen edges are not fully removed; decrease to avoid removing too much of the subject.

**Edge Softness** *(1 – 21)* — applies a Gaussian blur to the key mask edge, feathering the transition between keyed and non-keyed areas.

**Manual Key Color (R / G / B)** *(0 – 255 each)* — the target colour for `manual` mode. Default is pure green (0, 255, 0).

---

### 3.10 Effects — Signal Domain

The **SIGNAL DOMAIN** accordion group applies physically unusual transformations to the frame — processes borrowed from audio engineering, information theory, and signal analysis, applied to pixels where they do not belong.

All effects in this group are triggered by audio segment type and intensity, same as all other effects. Each has an enable checkbox, a Chance slider, and one or two parameter sliders.

---

**Resonant Rows**

Applies an IIR bandpass filter along each pixel row, treating the row as a one-dimensional audio signal. The filter creates spatial ringing at vertical edges — not blur, not sharpening, but resonance: a narrow-frequency echo around every boundary. At high Q values, the ringing overtakes the original image.

- *Freq* — centre frequency of the resonance (cycles per pixel). Lower = wider horizontal features resonate. Higher = finer detail resonates.
- *Q factor* — sharpness of the resonance peak. Low Q = gentle halo. High Q = aggressive oscillating fringe.

---

**Temporal RGB Shift**

The three colour channels are read from different points in time. Red is always the current frame. Green is N/2 frames behind. Blue is N frames behind. Moving objects cast colour-separated shadows across their own history. Stationary images are unaffected.

This is not the same as RGB Shift (which is spatial). This is chromatic temporal aberration.

- *Lag (frames)* — maximum temporal offset for the Blue channel. Scales proportionally with audio intensity.

---

**FFT Phase Corrupt**

Computes the 2D discrete Fourier transform of the frame, adds random noise to the phase of each frequency component while leaving the magnitude untouched, then transforms back. The image loses spatial coherence while preserving its overall brightness distribution — edges dissolve into wave interference patterns.

- *Phase Noise Amount* — amount of phase randomisation (0 = no effect, 1 = full random phase = maximum scramble).

---

**Waveshaper / Tube Sat**

Applies a tanh soft-clipping transfer function to pixel values, identical to tube amplifier saturation in audio production. Mid-range values compress toward the centre; peaks saturate. The result is a characteristic analogue warmth/distortion depending on drive level.

- *Drive* — gain before the clipping stage. Low = gentle saturation. High = hard, coloured distortion.

---

**Histogram Lag**

Matches the colour histogram of the current frame to the histogram of a frame from N frames ago, using per-channel cumulative distribution function remapping. The image retains its structure but borrows its colour palette from the past. When scene changes occur, the current frame takes on the colours of a completely different moment.

- *Palette Lag (frames)* — how far back in time the reference palette is drawn from.

---

**Wrong Chroma Sub (4:1:N)**

Converts the frame to YCrCb colour space, downsamples the Cr and Cb chroma channels by a large factor (2× to 8×), then upsamples them back with nearest-neighbour interpolation. The result: large, blocky colour patches overlaid on sharp luminance detail. This is the visual signature of extreme chroma subsampling — what happens when a bad codec discards too much colour information.

- *Downsample Factor* — how aggressively chroma is reduced. At 2 the effect is mild; at 8 colours appear in large rectangular blocks.

---

**Game of Life Mask**

Binarises the frame at half resolution using a luminance threshold, runs Conway's Game of Life for N iterations using the image content as the initial cell state, then uses the resulting live cells as a corruption mask — XOR-ing noise into those regions.

Brighter image regions tend to produce active cell populations. After several iterations, the pattern develops into Conway gliders, still lifes, and oscillators, creating an organic and semi-predictable corruption structure tied to the image content.

- *Iterations* — how many Game of Life generations to run per frame. More iterations = more evolved, less image-derived pattern.

---

**ELA (Error Level Analysis)**

Re-encodes the current frame as a JPEG at fixed quality, then computes the absolute difference between the original and the re-encoded version, amplified. Regions that were already heavily JPEG-compressed appear dark. Regions with fine detail, sharp edges, or recently edited content appear bright. Applied as a blend layer, this creates a forensic edge map that reveals the spatial structure of image compression.

- *ELA Blend* — 0 = pure ELA map (original image replaced by the error map), 1 = original only. Values between blend the two.

---

**Dtype Reinterpret**

Takes the raw bytes of the frame buffer and reinterprets the first half as 16-bit floating-point values. Small noise is added in the float16 domain. The bytes are then reinterpreted back as unsigned 8-bit integers and reshaped into the frame.

The corruption is not random — it is determined by the specific numerical patterns in the pixel data. Certain pixel value combinations produce NaN or Inf when interpreted as float16, which resolve to specific byte patterns (0 or 255 in uint8), creating structured bright and dark corruptions that are entirely content-dependent.

- *Noise Amount* — standard deviation of the float16-domain noise.

---

**Spatial Reverb**

Constructs a decaying impulse response (a series of diminishing echoes at fixed spatial intervals) and convolves it with each pixel row. The result is a horizontal reverberation — bright pixels produce trailing copies of themselves to the right, decaying in amplitude. This is room acoustic reverb applied to spatial light.

- *Reverb Decay* — rate at which echoes diminish. High values produce longer, louder reverb tails.

---

### 3.11 Export Settings

Located in the **EXPORT** tab.

**FPS** *(1 – 60, default 24)* — output frame rate. Does not affect the source audio timing; only controls how many video frames are generated per second of output.

**CRF** *(1 – 51, default 18)* — Constant Rate Factor for H.264/H.265 encoding. Lower values = higher quality and larger file size. A value of `18` is visually near-lossless. `23` is the H.264 default. Values above `30` produce visible compression artefacts.

---

### 3.12 Presets

The **Presets** panel (sidebar) provides named parameter sets for common styles.

**Built-in presets:**

| Preset | Character |
|---|---|
| **Drillcore (Fast Cut/Stutter)** | High chaos, fast cuts, heavy stutter and block glitch, RGB shift |
| **Datamosh (P-frame Bleed)** | Scene-detected source, P-frame datamosh, pixel sort, ghost trails, color bleed |
| **ASCII Rave** | ASCII filter dominant, flash, scanlines, stutter |
| **Death Grips Mode** | Maximum chaos, minimum cut duration, JPEG crush, bad signal, VHS, negative |
| **Rhythm Flash (Scene Mix)** | Scene detection, flash on every beat, zoom glitch |

**Working with presets:**

- **Double-click** or select + **Load** — applies all stored parameter values
- **Save Current** — prompts for a name and saves all current slider/toggle values to `presets.json`
- **Delete** — removes the selected preset from `presets.json`
- The active preset name is shown below the list

Custom presets persist between sessions in `presets.json` in the program directory.

---

### 3.13 Render Modes

Three render buttons are pinned to the bottom of the sidebar:

---

**DRAFT (5 sec / 480p)**

Renders the first 5 seconds of the audio track at 480p resolution. Effects that are computationally expensive (datamosh optical flow, ASCII) use simplified fast paths. Use this to quickly check cut timing and basic effect behavior.

---

**PREVIEW (5 sec)**

Renders the first 5 seconds at full configured resolution and quality. All effects run at full fidelity. Use this to validate the visual result before committing to a full render.

---

**RENDER FULL VIDEO**

Renders the complete output video for the entire duration of the audio track. A progress bar and status log show real-time progress. The output file path is shown in the console on completion.

The **STOP** button (right panel) interrupts any active render or preview playback.

---

### 3.14 Mystery Panel

The **[ ? ]** tab holds nine sliders. They are named.

The names are not metaphors.

Each one addresses a quantity that exists in the pipeline — a real parameter that shapes the signal in a real way — but the relationship between the name and what it does is not described here, because the description would be inaccurate. Not because we are concealing it. Because the accurate description requires a frame of reference that this document was not written in.

The sliders are not independent. What you give to one changes what another is capable of. A position that does nothing alone may do a great deal in the presence of another. The system has memory of the other sliders. So do the effects.

**VESSEL** feeds. It is not empty. What you gave it earlier is still present — weighted, layered behind what you are giving it now. At high values, what was begins to outweigh what is. It speaks to DOT.

**ENTROPY\_7** erodes a boundary that was never supposed to be there. The erosion is not random — it is living. It reads the frame as a condition and continues from it. What VESSEL has accumulated, ENTROPY\_7 grows inside.

**DELTA\_OMEGA** governs the speed of a transformation that happens whether you turn it on or not. You are only adjusting how much of it the rest of the system is allowed to hear.

**STATIC\_MIND** is a gate, but it gates the wrong thing. What passes through at 0 is not what passes through at 1. The transition between them is not a blend. RESONANCE, if present, does something to the structure that STATIC\_MIND was trying to hold in place.

**RESONANCE** does not merely resonate. Rows ring. At sufficient values the image stops being an image and becomes the record of its own frequency. It speaks to COLLAPSE. Together they lower each other's thresholds.

**COLLAPSE** initiates. It works in two directions simultaneously, though the second is not visible until the first is already underway. ZERO changes what it arrives at.

**ZERO** is not zero. It is a seed. The same values, a different ZERO — a different outcome. It also holds a record the original does not know it carries. FLESH\_K amplifies what ZERO reveals.

**FLESH\_K** is what it says. It converts. The conversion is not clean. What exits is not in the same space as what entered. At low values nothing appears to happen. The threshold is not fixed.

**DOT** is small. It separates what arrived together. Three things depart from the same moment but do not arrive at the same time. It lengthens when VESSEL is present.

Move them. Render. Observe what changed. Then move them again.

No combination is wrong. Some combinations are irreversible within the session.

### 3.15 Multiple Video Sources

The **Load Source Video** button accepts multiple files simultaneously. Hold `Ctrl` or `Shift` to select several files in the dialog, or drag-select a range.

When more than one file is loaded, the engine maintains an independent video reader for each source. At the beginning of every audio segment, a source is selected at random from the pool. Within that segment, frames are read sequentially from the chosen source.

This means:
- Different segments can come from completely different video files with no preparation or manual splicing
- Scene detection (if enabled) runs on all loaded files and collects cut points from each
- Datamosh pre-processing uses the first file in the selection as the P-frame source
- Files of different resolutions and frame rates are all valid; each is read at its native rate and scaled to the output resolution

The sidebar label shows the filename when one file is loaded, and the count (`3 videos loaded`) when multiple are selected.

---

### 3.16 Silence Treatment

Controls how the engine processes frames during **SILENCE** segments longer than 1 second.

Located in the **Cut Logic** section, under *Silence Treatment*.

| Option | Behavior |
|---|---|
| **Dim (darken)** | Multiplies all pixel values by 0.6, reducing brightness. The original behavior. Good for maintaining visual contrast between active and quiet moments. |
| **Soft Blur** | Applies a Gaussian blur (15×15 kernel). The image stays full brightness but loses sharpness, creating a dreamlike soft-focus look during quiet passages. |
| **Blur + Dim** | Applies a lighter blur (11×11) combined with a 0.7 brightness multiplier. The strongest treatment — defocused and darkened simultaneously. |
| **None** | No treatment. Silent segments are rendered identically to active ones. Use when silence is compositionally intentional and should not be visually marked. |

---

## 4. Disc VPC 01 R-T — Real-Time

### 4.1 Interface Layout

The window is divided into two areas:

**Left panel (258 px fixed):**
- Setup section — audio device selection, video and overlay file loading
- Controls section — start/stop, chaos and sensitivity sliders
- Effects section — effect toggle buttons, overlay frequency, sequential mode

**Right panel (expandable):**
- Live video output — resizes with the window
- Indicator strip — beat, RMS, gate, FPS, spectral bars
- Console — timestamped log

### 4.2 Setup

**Audio device selector**

The dropdown lists all audio input devices detected by the system. This includes:
- Microphones
- Line inputs
- WASAPI loopback devices (e.g., "Stereo Mix", virtual cables, VoiceMeeter) for capturing system audio

Select the desired device before starting. The **↺** button refreshes the device list.

**Changing the device while running:** Select a new device from the dropdown while the engine is active. The audio stream switches immediately and recalibration begins automatically.

**📁 Video** — load one or more video files to be played and processed. Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.mpg`, `.mpeg`. Hold `Ctrl` or `Shift` to select multiple files at once.

When multiple files are loaded, the engine keeps a separate sequential reader for each. Random cuts (triggered by audio events) jump to a random frame from a randomly selected source. Sequential playback stays within the current source until its end, then continues from the next source on the following call. Pre-caching runs on all sources in parallel on startup. The label shows `N videos loaded` when more than one file is active.

**📁 Overlays** — load a folder of overlay images (same formats as §3.9).

---

### 4.3 Audio Calibration

When the engine starts, it enters a **calibration period** lasting approximately 2 seconds (86 audio chunks at 44100 Hz). During this time:

- The engine measures the ambient noise floor of the selected audio device
- All cuts and effects are suppressed
- The gate indicator displays `CALIBRATING (N)...` counting down

After calibration completes, the engine sets its gate threshold to `4 × the 95th-percentile RMS` measured during calibration. This ensures the engine ignores background noise and only reacts to actual audio signal.

The **Recalibrate** button resets the calibration state and begins a new measurement cycle. Use this if the noise environment changes or after switching audio devices manually.

---

### 4.4 Controls and Parameters

**START / STOP**

START begins audio capture and video processing. A video file must be loaded first. STOP halts everything. The video source remains loaded after stopping; press START again to resume without reloading files.

**Resolution** selector — choose output resolution before pressing START: `640×360`, `800×450`, `1280×720`, `1920×1080`. Cannot be changed while the engine is running.

---

**Chaos** *(0.0 – 1.0)*

Scales the probability of all effects firing and increases the frequency of random video cuts. At `0.0` the engine reacts minimally; at `1.0` it reacts maximally to every audio event.

---

**Sensitivity** *(0.2 – 5.0)*

A multiplier applied to the auto-calibrated noise gate:

```
effective_gate = calibrated_gate × sensitivity
```

At `1.0` the gate is at its calibrated value. Increase to require louder audio before the engine reacts. Decrease toward `0.2` to react to quieter signals.

---

**Master FX** *(0.0 – 1.0)*

Global wet/dry blend for all effects. At `1.0` effects are fully applied. At `0.0` the original unprocessed frame is shown. Useful for fading effects in or out during a performance without toggling individual buttons.

---

**Cut Min s** *(0.0 – 2.0)*

Minimum interval in seconds between video cuts. Prevents strobing caused by rapid consecutive cuts on dense audio. Default `0.3`. Set to `0.0` to disable limiting.

---

**Stage controls**

| Button | Keyboard | Behaviour |
|---|---|---|
| **BLACKOUT** | `Space` | Outputs a black frame. The engine keeps running and processing audio — remove blackout to resume instantly. |
| **FREEZE** | `F` | Holds the last processed frame. Audio analysis continues. |
| **PANIC** | `P` | Clears all internal effect buffers (datamosh history, ghost trail, feedback accumulation). Use when visuals get stuck. Also turns off Blackout and Freeze if active. |

---

**Presets**

| Button | Action |
|---|---|
| **Save Preset** | Saves all current settings (effects, sliders, chroma key, resolution) to a `.json` file. |
| **Load Preset** | Loads a previously saved `.json` preset. A blank preset with all effects off is included in the release bundle as `presets/rt_blank.json`. |

---

**Sequential** toggle

When **active** (button highlighted): the engine plays the video sequentially, frame by frame, with no random cuts. Effects still apply.

When **inactive** (default): the engine plays sequentially but cuts to a random frame on audio events (IMPACT, BUILD, DROP, or SUSTAIN with a beat).

---

### 4.5 Effects

Each effect is a toggle button in the Effects panel. Active effects appear highlighted (dark blue background, white text). Inactive effects appear in the default raised state. Effects are organized into three groups.

All effects run in **draft mode** — lightweight code paths that preserve responsiveness. Probability of firing is governed by each effect's internal `chance` value (tuned for realtime use) and the **Chaos** slider.

---

#### Core

| Effect | Trigger segments | Description |
|---|---|---|
| **RGB Shift** | IMPACT, BUILD, NOISE, DROP | Shifts red channel right, blue channel left by an offset scaled to intensity. Produces chromatic aberration. |
| **Flash** | DROP, IMPACT | Blends the frame toward white or black on sharp transients. |
| **Stutter** | any (beat events) | Blends the current frame with a randomly chosen recent buffered frame. Creates a momentary freeze/smear. |
| **Pxl Sort** | NOISE, IMPACT, DROP | Sorts pixel columns within horizontal strips by luminance. |
| **Datamosh** | NOISE, SUSTAIN, IMPACT, DROP | Computes optical flow between consecutive frames and warps the previous frame forward using that motion, simulating P-frame bleed. Uses `DISOpticalFlow` (ultrafast preset in draft mode). |
| **Overlays** | any (when active) | Composites a randomly selected image from the overlay folder onto the frame at a random position and scale. |

---

#### Glitch

| Effect | Trigger segments | Description |
|---|---|---|
| **Ghost** | SUSTAIN, BUILD | Blends the current frame with the previous frame using `addWeighted`. Leaves a motion trail. |
| **Scanlines** | SUSTAIN, NOISE | Darkens every N-th horizontal row, simulating a CRT screen. Spacing and darkness scale with intensity. |
| **Bitcrush** | all | Reduces colour bit-depth by right-shifting and restoring pixel values. Produces posterisation and banding. |
| **Blk Glitch** | IMPACT, DROP, NOISE | Randomly copies or fills rectangular blocks from other locations on the frame. |
| **Negative** | IMPACT, DROP, NOISE | Inverts all colour values (`255 − value`) for one frame. |
| **Clr Bleed** | NOISE, SUSTAIN | Applies a horizontal blur to a single randomly chosen colour channel, producing colour channel smear. |
| **Interlace** | SUSTAIN | Replaces every odd row with the corresponding row from the previous frame, simulating interlaced video. |
| **Bad Signal** | DROP, NOISE | Draws vertical noise bars and shifts random rows horizontally, emulating broadcast signal loss. |

---

#### Spatial / Color

| Effect | Trigger segments | Description |
|---|---|---|
| **Zoom Glitch** | IMPACT, DROP | Crops a zoomed-in center region and resizes it back to full frame using nearest-neighbour, creating a pixelated punch-in. |
| **Mosaic** | IMPACT, BUILD | Downscales the frame to a tiny grid and upscales back with nearest-neighbour, creating a blocky mosaic that pulses with intensity. |
| **PhaseShift** | NOISE, DROP | Divides the frame into horizontal bands and shifts alternating bands left and right. |
| **Dither** | SILENCE, SUSTAIN | Applies ordered Bayer dithering at reduced colour depth. Produces a retro halftone pattern. |
| **Feedback** | SUSTAIN, BUILD | Accumulates a weighted blend of current and previous frames, creating a visual feedback/burn-in trail. |
| **Temp RGB** | all | Assigns R, G, B channels from frames captured at different time offsets, producing chromatic temporal aberration. |
| **Waveshaper** | all | Applies a tanh soft-clipping waveshaper to pixel values, producing colour saturation and mild overexposure. |

---

**Ovl Freq** *(0.0 – 1.0)* — probability per frame that an overlay is placed when audio is active and Overlays is enabled.

**Chroma Key** — background removal for overlay images. Uses the same modes and parameters as §3.9: `none`, `dominant`, `secondary`, `manual`. Tolerance, Edge Softness, and Manual Key Color sliders are in the Chroma Key sub-section of the Effects panel.

---

### 4.6 Indicators

The indicator strip below the video output shows real-time audio analysis state.

**● BEAT** — flashes red briefly on each detected beat event.

**RMS: 0.0000** — the current smoothed RMS level of the audio signal (exponentially weighted).

**GATE / CALIBRATING** — after calibration, shows the effective gate threshold value. Turns green when the RMS exceeds the gate (audio is active). During calibration, shows a countdown.

**FPS: --** — frames per second of the video processing loop, updated every second.

**Spectral bars:**

| Bar | Color | Frequency Band |
|---|---|---|
| **B** (Bass) | Blue | 20 – 300 Hz |
| **M** (Mid) | Green | 300 – 3000 Hz |
| **T** (Treble) | Red | 3000 – 16000 Hz |
| **F** (Flatness) | Amber | Spectral flatness (0 = tonal, 1 = noise) |

All bars are normalized to the maximum value observed during the session. Bars are only drawn when the audio level is above the gate.

---

### 4.7 Fullscreen and Second Monitor

**Fullscreen (F11 or FS button)**

Opens a separate window covering the full display. The video is scaled to fill the screen. The engine must be running to open fullscreen. Press **ESC** or click the on-screen button to exit.

**Second Monitor (F12)**

Opens a second output window positioned to the right of the primary display (at x=1920 if screen width > 1920). The video is independently scaled to this window's dimensions. Resize or move the window manually to fit any monitor. Close it with F12 again or by stopping the engine.

Both windows are updated on every frame in sync with the main window.

---

### 4.8 Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Toggle Blackout |
| `F` | Toggle Freeze |
| `P` | Panic — clear all effect buffers |
| `R` | Recalibrate noise gate |
| `1` – `9` | Toggle effects by position (1 = first in list, 9 = ninth) |
| `0` | Toggle tenth effect |
| `F11` | Toggle fullscreen output |
| `ESC` | Exit fullscreen |
| `F12` | Toggle second monitor output |

---

## 5. Segment Types Reference

Both programs use the same segment classifier. Understanding segment types helps predict which effects will fire.

| Type | Condition | Typical audio moment |
|---|---|---|
| **IMPACT** | Short duration (< 0.3 s) + loud + sharp attack | Drum hit, snare, sharp transient |
| **BUILD** | Rising RMS trend over recent history | Pre-drop buildup, crescendo |
| **DROP** | Falling RMS after a loud period | Post-peak decay, breakdown |
| **SUSTAIN** | Loud (above threshold), no strong trend | Sustained note, chord, steady beat |
| **NOISE** | High spectral flatness (> 1.5 × mean) | Distortion, white noise, dense texture |
| **SILENCE** | RMS below threshold | Quiet passage, gap between phrases |

Effects declare which segment types they respond to. An effect will not fire on a segment type not in its trigger list, regardless of the chance setting.

---

## 6. Recommended Workflows

### Disc VPC 01 — Iterative editing

1. Load audio and video source files
2. Select a built-in preset matching your intended style
3. Click **DRAFT** to render a quick 5-second 480p preview
4. Review in the preview monitor. Adjust Beat Threshold if cuts feel too frequent or too sparse
5. Toggle specific effects on/off and adjust their chance sliders
6. Click **PREVIEW** to verify at full quality
7. Repeat steps 4–6 until satisfied
8. Click **RENDER FULL VIDEO**

### Disc VPC 01 R-T — Live performance

1. Load a video file (ideally abstract, loopable, or clip-rich material)
2. Optionally load an overlay folder
3. Select the appropriate audio input device (use system loopback to react to a DAW/player)
4. Click **▶ START** and wait for calibration to complete (~2 seconds)
5. Adjust **Sensitivity** if the engine reacts to silence or fails to react to audio
6. Toggle effects and adjust **Chaos** to taste
7. Use **F11** for fullscreen output or **F12** for a second monitor
8. Use **Recalibrate** after major environment changes or device switches

---

## 7. Troubleshooting

**Engine reacts to silence / fires constantly**

The noise gate has not calibrated properly. Click **Recalibrate** and wait 2 seconds. If the problem persists, increase **Sensitivity** (higher multiplier raises the gate).

**Black screen on video output after reloading a video**

Load the new video file using the **📁 Video** button. The previous source is closed automatically and replaced.

**Audio device not listed**

Click the **↺** refresh button. On Windows, ensure the device is enabled in Sound settings. For system loopback, enable "Stereo Mix" in the Windows Sound control panel or use a virtual cable.

**`Illegal Audio-MPEG-Header` in console**

Benign FFmpeg warning from malformed MP3 headers. The audio is still decoded correctly. This warning can be safely ignored.

**Slow render / low FPS in R-T**

Disable computationally expensive effects (Datamosh, Pixel Sort). Reduce the source video resolution. Close other GPU-intensive applications.

**`madmom` install fails on Python 3.11+**

Use Python 3.10.x exactly. `madmom` requires Cython extensions that do not build on newer Python versions.

**Effects never fire**

Verify that at least one effect is toggled on, the engine has finished calibrating, and the audio signal is above the gate (RMS indicator turns green). Increase Chaos or decrease Sensitivity if needed.

---

*Disc VPC 01 / Disc VPC 01 R-T*
