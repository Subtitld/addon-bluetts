"""BlueV3 on ONNX Runtime, through upstream's own `blue_onnx` (vendored).

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The downloaded files keep the release bundle's layout under one revision
folder (onnx/, voices/, frontend/), because `blue_onnx` finds the front end
next to the graphs. What the bundle ships as code (the Yiddish front end) is
in the add-on instead; `blue_onnx` is pointed at it through BLUE_YI_FRONTEND.

Synthesis is upstream's: text normalization, G2P per language, chunking at
the model's 300-token limit, 8 flow steps at CFG 4 and the 0.1 s join, then
the 0.95 peak limit. The add-on only adds a seed, cancellation between flow
steps, and the choice of Hebrew G2P weights.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

import numpy as np

YI_FRONTEND = Path(__file__).resolve().parent / 'vendor' / 'yi'
PEAK_LIMIT = 0.95


class Cancelled(Exception):
    """The request was cancelled."""


def default_threads() -> int:
    """Upstream's own default: at most 8, more only oversubscribes."""
    return max(1, min(8, os.cpu_count() or 1))


def _blue_onnx():
    from bluetts_addon.vendor import blue_onnx
    return blue_onnx


class _CancellableSession:
    """An ONNX Runtime session that checks for a cancel before each run: the flow graph runs 8 times a chunk."""

    def __init__(self, session, cancelled: Callable[[], bool]):
        self._session = session
        self.cancelled = cancelled

    def run(self, *args, **kwargs):
        if self.cancelled():
            raise Cancelled()
        return self._session.run(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


class BlueTTS:
    """The loaded model: graphs, G2P front ends, and the clone's reference encoder on demand."""

    def __init__(self, root: Path, threads: int, renikud_path: Path | None = None,
                 yi_model_dir: Path | None = None):
        os.environ['ORT_NUM_THREADS'] = str(threads)
        os.environ['BLUE_YI_FRONTEND'] = str(YI_FRONTEND)
        if yi_model_dir is not None:
            os.environ['PHONIKUD_YI_RENIKUD_MODEL'] = str(yi_model_dir)
        os.environ.setdefault('POLYG2P_EN_MODEL', 'large')
        os.environ.pop('POLYG2P_DATA', None)  # the data folder next to the graphs, never another one
        os.environ.pop('BLUE_G2P_NOLEX', None)
        blue = _blue_onnx()
        self.root = Path(root)
        self.onnx_dir = self.root / 'onnx'
        self.threads = threads
        self.tts = blue.load_text_to_speech(str(self.onnx_dir), renikud_path=str(renikud_path) if renikud_path else None)
        self._cancel = [lambda: False]
        self.tts.vector_est_ort = _CancellableSession(self.tts.vector_est_ort, lambda: self._cancel[0]())
        self.release = self.tts.release or {}
        self.sample_rate = int(self.tts.sample_rate)
        self._reference_session = None
        self._lock = threading.Lock()

    def set_renikud(self, path: Path | None) -> None:
        """Switch the Hebrew G2P weights; the G2P is rebuilt on the next Hebrew line."""
        wanted = str(path) if path else None
        if self.tts.g2p.renikud_path != wanted:
            self.tts.g2p.renikud_path = wanted
            self.tts.g2p.renikud = None

    def set_yiddish(self, model_dir: Path) -> None:
        os.environ['PHONIKUD_YI_RENIKUD_MODEL'] = str(model_dir)

    def stock_style(self, file_name: str):
        blue = _blue_onnx()
        style = blue.load_voice_style([str(self.root / 'voices' / file_name)])
        self.tts.check_style(style)
        return style

    def clone_style(self, samples: np.ndarray, rate: int):
        """The style of a reference clip: mono float32 samples at any rate."""
        from bluetts_addon.vendor.blue_onnx import style as blue_style
        import onnxruntime as ort

        if self._reference_session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            self._reference_session = ort.InferenceSession(str(self.onnx_dir / 'reference_encoder.onnx'),
                                                           sess_options=options,
                                                           providers=['CPUExecutionProvider'])
        return blue_style.release_style_from_samples(samples, rate, self.release, str(self.onnx_dir),
                                                     session=self._reference_session)

    def synthesize(self, text: str, style, language: str, speed: float = 1.0, steps: int | None = None,
                   seed: int = 42, progress: Callable[[float], None] | None = None,
                   cancelled: Callable[[], bool] = lambda: False) -> np.ndarray:
        """Mono float32 samples at `sample_rate`, limited to a 0.95 peak."""
        inference = self.release.get('inference', {})
        steps = int(steps or inference.get('steps', 8))
        cfg_scale = float(inference.get('cfg_scale', 4.0))
        silence = float(inference.get('join_silence_s', 0.1))
        # Upstream draws its noise from numpy's global generator.
        np.random.seed(seed & 0xFFFFFFFF)
        # A rough length, for progress only: Hebrew and English read at about 14 characters a second.
        expected = max(1.0, len(text) / 14.0 / max(speed, 0.1))
        pieces = []
        produced = 0.0
        self._cancel[0] = cancelled
        try:
            chunks = self.tts(text, lang=language, style=style, total_step=steps, speed=speed, cfg_scale=cfg_scale,
                              silence_duration=silence, normalize_text=True, stream=True)
            for wav, _ in chunks:
                if cancelled():
                    raise Cancelled()
                chunk = np.asarray(wav, dtype=np.float32).reshape(-1)
                pieces.append(chunk)
                produced += len(chunk) / self.sample_rate
                if progress is not None:
                    progress(min(0.95, produced / expected))
        finally:
            self._cancel[0] = lambda: False
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return _blue_onnx().limit_peak(audio, PEAK_LIMIT)
