"""Render benchmark — measure encode speed and output size.

Usage (Windows):
    .venv\\Scripts\\python.exe tools\\bench_render.py [--codec LABEL] [--crf N]
        [--preset NAME] [--quality NAME] [--duration SEC] [--video PATH]
        [--audio PATH] [--keep]

Usage (Linux / macOS):
    .venv/bin/python tools/bench_render.py [...same flags...]

Or simply: `python -m tools.bench_render --help` if the venv is active.

Auto-generates a 10s testsrc + sine fixture under `tools/.bench_fixtures/`
if --video / --audio are not given (and TEST_VIDEO / TEST_AUDIO env-vars are
unset). Writes output under `tools/.bench_out/` and prints one tab-separated
line per run plus a final wall-time / file-size / ffprobe-duration / frame-
count summary. Designed to be safe to re-run; fixtures are cached.

ffmpeg is sourced from `imageio_ffmpeg.get_ffmpeg_exe()` (a wheel
dependency in requirements.txt), so the bench is self-contained on every
platform — no system ffmpeg required. ffprobe is looked up next to the
ffmpeg binary; if missing, falls back to the PATH lookup.

Use this BEFORE and AFTER each optimization step. Numbers go in the commit
message so we can see which step actually paid off.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Make the package importable when run from repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vpc.render import BreakcoreEngine, RENDER_FINAL  # noqa: E402
from vpc.render.sink import ffmpeg_bin  # noqa: E402
from vpc.render.quality import QUALITY_PRESETS  # noqa: E402


FIXTURE_DIR = ROOT / 'tools' / '.bench_fixtures'
OUT_DIR = ROOT / 'tools' / '.bench_out'


def ensure_fixtures(duration: float) -> tuple[Path, Path]:
    """Generate (or reuse) a synthetic test video + audio pair."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    video = FIXTURE_DIR / f'testsrc_{int(duration)}s.mp4'
    audio = FIXTURE_DIR / f'sine_{int(duration)}s.wav'
    ff = ffmpeg_bin()
    if not video.exists():
        subprocess.run([
            ff, '-y', '-f', 'lavfi',
            '-i', f'testsrc=duration={duration}:size=1280x720:rate=24',
            '-pix_fmt', 'yuv420p', '-c:v', 'libx264', '-preset', 'ultrafast',
            str(video),
        ], check=True, capture_output=True)
    if not audio.exists():
        subprocess.run([
            ff, '-y', '-f', 'lavfi',
            '-i', f'sine=frequency=220:duration={duration}',
            '-ar', '44100', str(audio),
        ], check=True, capture_output=True)
    return video, audio


def ffprobe_info(path: Path) -> dict:
    """Return {'duration': float, 'nb_frames': int} via ffprobe."""
    # imageio_ffmpeg ships ffmpeg only; ffprobe is usually next to it.
    ff = ffmpeg_bin()
    probe = Path(ff).with_name('ffprobe.exe' if os.name == 'nt' else 'ffprobe')
    if not probe.exists():
        probe = 'ffprobe'  # PATH fallback
    try:
        r = subprocess.run([
            str(probe), '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries',
            'stream=nb_read_frames,duration',
            '-of', 'json', str(path),
        ], capture_output=True, text=True, timeout=30)
        data = json.loads(r.stdout or '{}')
        s = (data.get('streams') or [{}])[0]
        return {
            'duration': float(s.get('duration') or 0.0),
            'nb_frames': int(s.get('nb_read_frames') or 0),
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return {'duration': -1.0, 'nb_frames': -1}


def build_cfg(video: Path, audio: Path, output: Path, *,
              codec: str, crf: int, preset: str,
              quality: str | None, tune: str | None) -> dict:
    # If a Quality preset is named, expand it into the manual fields the
    # engine actually reads (crf / export_preset / tune). This mirrors
    # what the GUI does when the dropdown changes — keeps bench numbers
    # honest about what the preset means.
    if quality and QUALITY_PRESETS.get(quality):
        spec = QUALITY_PRESETS[quality]
        crf = int(spec['crf'])
        preset = str(spec['export_preset'])
        tune = str(spec['tune'])
    cfg = {
        'video_paths': [str(video)],
        'audio_path': str(audio),
        'output_path': str(output),
        'fps': 24,
        'crf': crf,
        'export_preset': preset,
        'video_codec': codec,
        'resolution': '720p',
        'resolution_mode': 'preset',
        'silence_mode': 'none',
        'chaos_level': 0.5,
        'threshold': 1.2,
        'transient_thresh': 0.5,
        'min_cut_duration': 0.05,
        'snap_to_beat': False,
        'snap_tolerance': 0.05,
        'use_scene_detect': False,
    }
    if quality:
        cfg['quality_preset'] = quality
    if tune:
        cfg['tune'] = tune
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--codec', default='H.264 (MP4)')
    ap.add_argument('--crf', type=int, default=18)
    ap.add_argument('--preset', default='medium')
    ap.add_argument('--quality', default=None,
                    help='Quality preset (Archive/High/Web/Compact); '
                         'overrides --crf/--preset/--tune if set.')
    ap.add_argument('--tune', default=None,
                    help='x264/x265 tune (film/grain/animation/stillimage).')
    ap.add_argument('--duration', type=float, default=10.0)
    ap.add_argument('--video', default=os.environ.get('TEST_VIDEO'))
    ap.add_argument('--audio', default=os.environ.get('TEST_AUDIO'))
    ap.add_argument('--keep', action='store_true',
                    help='Keep output file after run.')
    args = ap.parse_args()

    if args.video and args.audio:
        video, audio = Path(args.video), Path(args.audio)
    else:
        print(f'[bench] generating {args.duration:.0f}s testsrc/sine fixture...')
        video, audio = ensure_fixtures(args.duration)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ext = {'H.264 (MP4)': 'mp4', 'H.265 (MP4)': 'mp4',
           'H.264 (MKV)': 'mkv', 'H.265 (MKV)': 'mkv',
           'H.264 (MOV)': 'mov', 'ProRes (MOV)': 'mov',
           'VP9 (WebM)': 'webm'}.get(args.codec, 'mp4')
    tag = (args.quality or f'crf{args.crf}-{args.preset}').replace(' ', '_')
    output = OUT_DIR / f'bench_{tag}.{ext}'
    if output.exists():
        try: output.unlink()
        except OSError: pass

    cfg = build_cfg(video, audio, output,
                    codec=args.codec, crf=args.crf, preset=args.preset,
                    quality=args.quality, tune=args.tune)
    # Read back the values that actually went into the encoder, so the
    # printed row reflects reality (Quality preset overrides --crf etc).
    eff_crf = cfg['crf']
    eff_preset = cfg['export_preset']
    eff_tune = cfg.get('tune', 'none')

    def _progress(msg: str, value=None):
        if value is not None and value % 25 == 0:
            print(f'  [{value:3d}%] {msg}')

    engine = BreakcoreEngine(cfg, progress_callback=_progress)
    t0 = time.perf_counter()
    ok = engine.run(render_mode=RENDER_FINAL,
                    max_output_duration=args.duration)
    elapsed = time.perf_counter() - t0

    if not ok or not output.exists():
        print(f'[bench] FAILED — no output produced')
        return 1

    info = ffprobe_info(output)
    size = output.stat().st_size
    rate = args.duration / elapsed if elapsed > 0 else 0.0
    print()
    print('codec\tcrf\tpreset\tquality\ttune\twall_s\trealtime_x\t'
          'size_mb\tdur_s\tframes')
    print(f'{args.codec}\t{eff_crf}\t{eff_preset}\t{args.quality or "-"}\t'
          f'{eff_tune or "-"}\t{elapsed:.2f}\t{rate:.2f}x\t'
          f'{size/1e6:.2f}\t{info["duration"]:.3f}\t{info["nb_frames"]}')

    if not args.keep:
        try: output.unlink()
        except OSError: pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
