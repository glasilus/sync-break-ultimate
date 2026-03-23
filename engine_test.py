"""
engine.py - Breakcore Engine
Pipeline: cv2.VideoCapture -> EffectChain -> ffmpeg subprocess pipe -> output file.
Replaces MoviePy. All frame processing is numpy/cv2.
"""

import os
import random
import subprocess
import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Callable

from analyzer import AudioAnalyzer, Segment, SegmentType
from effects import (
    PixelSortEffect, DatamoshEffect, ASCIIEffect, FlashEffect,
    GhostTrailsEffect, RGBShiftEffect, BlockGlitchEffect, PixelDriftEffect,
    ScanLinesEffect, BitcrushEffect, ColorBleedEffect, FreezeCorruptEffect,
    NegativeEffect, JPEGCrushEffect, FisheyeEffect, VHSTrackingEffect,
    InterlaceEffect, BadSignalEffect, DitheringEffect, ZoomGlitchEffect,
    FeedbackLoopEffect, PhaseShiftEffect, MosaicPulseEffect, EchoCompoundEffect,
    KaliMirrorEffect, GlitchCascadeEffect, OverlayEffect, ChromaKeyEffect,
    MysterySection,
)

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

RENDER_DRAFT = 'draft'
RENDER_PREVIEW = 'preview'
RENDER_FINAL = 'final'