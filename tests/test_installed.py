"""The frozen binary, unpacked from its release archive the way Subtitld installs it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

$BLUETTS_TEST_BINARY names the binary; without it these tests skip (and the
release workflow refuses skipped tests). The binary downloads what each line
needs into $SUBTITLD_MODELS_DIR (the model, the Hebrew and Yiddish G2P and
the reference encoder: 2.1 GB), with nothing but what is in its own bundle,
then speaks every language, clones a clip and cancels a line.

    BLUETTS_TEST_BINARY=dist/bluetts-addon/bluetts-addon pytest tests/test_installed.py
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import wave
from pathlib import Path

import pytest

BINARY = os.environ.get('BLUETTS_TEST_BINARY', '')
pytestmark = pytest.mark.skipif(not BINARY, reason='BLUETTS_TEST_BINARY is not set')

LINES = [
    ('he', 'bluetts-noa', 'שלום, זהו מודל דיבור בעברית.'),
    ('en-US', 'bluetts-adam', 'Hello, this line was recorded on March 3rd.'),
    ('de', 'bluetts-maya', 'Guten Morgen, wie geht es Ihnen heute?'),
    ('es-MX', 'bluetts-sara', 'Buenos días, ¿cómo estás hoy?'),
    ('it', 'bluetts-daniel', 'Buongiorno, come stai oggi?'),
    ('ru', 'bluetts-lily', 'Привет, как у тебя дела сегодня?'),
    ('yi', 'bluetts-david', 'גוט מאָרגן, ווי גייט עס?'),
]


class Addon:
    """The binary on the protocol, with a watchdog so a hang fails instead of stalling CI."""

    def __init__(self, tmp_path: Path, timeout: float):
        env = dict(os.environ, SUBTITLD_TEMP_DIR=str(tmp_path))
        env.setdefault('SUBTITLD_MODELS_DIR', str(tmp_path / 'models'))
        self.proc = subprocess.Popen([BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env)
        self.timer = threading.Timer(timeout, self.proc.kill)
        self.timer.start()

    def send(self, frame: dict) -> None:
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, 'the add-on closed its output (crashed or timed out)'
        return json.loads(line)

    def until_terminal(self, req_id: str, on_progress=None) -> dict:
        while True:
            frame = self.read()
            if frame.get('id') != req_id:
                continue
            if frame['type'] in ('result', 'error'):
                return frame
            if on_progress is not None:
                on_progress(frame)

    def close(self) -> int:
        try:
            self.send({'type': 'shutdown'})
        except (BrokenPipeError, OSError):
            pass
        try:
            return self.proc.wait(timeout=15)
        finally:
            self.timer.cancel()
            self.proc.kill()


def test_version():
    out = subprocess.run([BINARY, '--version'], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.startswith('bluetts-addon ')


def test_self_test_loads_the_front_ends():
    out = subprocess.run([BINARY, '--self-test'], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report['ok'] is True and report['voices'] == 16 and report['yiddish']


def test_speaks_every_language_clones_and_cancels(tmp_path):
    addon = Addon(tmp_path, timeout=3600)
    try:
        hello = addon.read()
        assert hello['type'] == 'hello' and hello['addon'] == 'bluetts'
        addon.send({'type': 'ready', 'host': 'test'})

        for n, (language, voice, text) in enumerate(LINES):
            out = tmp_path / f'{n}.wav'
            addon.send({'id': str(n), 'type': 'tts.synthesize', 'params': {
                'text': text, 'voice': voice, 'language': language, 'seed': 5, 'output_path': str(out)}})
            result = addon.until_terminal(str(n))
            assert result['type'] == 'result', (language, result)
            assert result['data']['sample_rate'] == 44100 and result['data']['duration_sec'] > 1, language
            with wave.open(str(out)) as clip:
                assert clip.getframerate() == 44100 and clip.getnframes() > 44100, language

        clone = tmp_path / 'clone.wav'
        addon.send({'id': 'b', 'type': 'tts.synthesize', 'params': {
            'text': 'See you tomorrow.', 'voice': 'bluetts-clone', 'language': 'en', 'seed': 5,
            'output_path': str(clone), 'voice_ref_audio': str(tmp_path / '1.wav')}})
        assert addon.until_terminal('b')['type'] == 'result'
        assert clone.stat().st_size > 1000

        cancelled = []

        def on_progress(frame):
            if frame.get('message') == 'Speaking...' and not cancelled:
                cancelled.append(frame['value'])
                addon.send({'type': 'cancel', 'target': 'c'})

        addon.send({'id': 'c', 'type': 'tts.synthesize', 'params': {
            'text': 'This line is long enough to be interrupted before it ends. ' * 8,
            'voice': 'bluetts-noa', 'language': 'en', 'output_path': str(tmp_path / 'c.wav')}})
        error = addon.until_terminal('c', on_progress)
        assert cancelled and error['type'] == 'error' and error['code'] == 'cancelled'
        assert not (tmp_path / 'c.wav').exists()
    finally:
        assert addon.close() == 0
