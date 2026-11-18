"""Real synthesis with the pinned model, from source.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Every download group is fetched into $SUBTITLD_MODELS_DIR, or
~/.cache/subtitld/models, on the first run (2.4 GB with both Hebrew models),
checked against pins.json.
"""

from __future__ import annotations

import json
import platform
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bluetts_addon import __main__ as addon, download, engine, voices  # noqa: E402

pytestmark = pytest.mark.model
PARITY = json.loads((REPO_ROOT / 'tests' / 'fixtures' / 'upstream_parity.json').read_text(encoding='utf-8'))
# The fixture was made on x86-64 Linux; ONNX Runtime's kernels round differently elsewhere.
SAME_KERNELS = sys.platform.startswith('linux') and platform.machine().lower() in ('x86_64', 'amd64')


@pytest.fixture(scope='module')
def model():
    for group in ('core', 'clone', 'he', 'he-compact', 'yi'):
        download.ensure(addon.pins(group), lambda *_: None, lambda: False)
    return engine.BlueTTS(addon.model_root(), threads=engine.default_threads(),
                          renikud_path=addon.renikud_path('accurate'), yi_model_dir=addon.yi_model_dir())


def _noa(model):
    return model.stock_style('noa.json')


@pytest.mark.parametrize('language,text', [
    ('he', 'אנחנו צריכים לצאת לפני הזריחה.'),
    ('en', 'We have to leave before sunrise.'),
    ('de', 'Wir müssen vor Sonnenaufgang aufbrechen.'),
    ('es', 'Tenemos que salir antes del amanecer.'),
    ('it', "Dobbiamo partire prima dell'alba."),
    ('ru', 'Нам нужно уйти до рассвета.'),
    ('yi', 'מיר דאַרפֿן אַוועקגיין פֿאַר זונאויפֿגאַנג.'),
])
def test_speaks_each_language(model, language, text):
    wav = model.synthesize(text, _noa(model), language, seed=3)
    seconds = len(wav) / model.sample_rate
    assert model.sample_rate == voices.SAMPLE_RATE
    assert 0.8 < seconds < 8, seconds
    assert wav.dtype == np.float32 and np.isfinite(wav).all()
    assert 0.01 < float(np.sqrt(np.mean(wav ** 2))) < 0.5
    assert float(np.max(np.abs(wav))) <= engine.PEAK_LIMIT + 1e-6


@pytest.mark.parametrize('language', sorted(PARITY['lines']))
def test_matches_upstream(model, language):
    """The vendored blue_onnx, driven by the add-on, speaks what upstream's BlueTTS class speaks."""
    line = PARITY['lines'][language]
    wav = model.synthesize(line['text'], model.stock_style(line['voice']), language, seed=line['seed'])
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if SAME_KERNELS:
        assert len(wav) == line['samples']
        np.testing.assert_allclose(wav[::1000], line['every_1000th'], atol=2e-4)
        assert rms == pytest.approx(line['rms'], rel=1e-3)
    else:
        assert abs(len(wav) - line['samples']) <= 3072  # one latent frame
        assert rms == pytest.approx(line['rms'], rel=0.05)


def test_a_seed_gives_the_same_line(model):
    a = model.synthesize('Same seed, same line.', _noa(model), 'en', seed=11)
    b = model.synthesize('Same seed, same line.', _noa(model), 'en', seed=11)
    np.testing.assert_array_equal(a, b)


def test_speed_changes_the_length(model):
    normal = model.synthesize('This sentence is spoken at two speeds.', _noa(model), 'en', seed=4)
    fast = model.synthesize('This sentence is spoken at two speeds.', _noa(model), 'en', speed=1.5, seed=4)
    assert len(fast) < 0.8 * len(normal)


def test_long_text_is_split_and_joined(model):
    text = ('This is a long line, long enough that BlueTTS splits it into chunks before speaking it. ' * 6).strip()
    wav = model.synthesize(text, _noa(model), 'en', seed=2)
    assert len(wav) / model.sample_rate > 20


def test_every_stock_voice_loads(model):
    for voice in voices.stock():
        style = model.stock_style(voice.file)
        assert style.ttl.shape[0] == 1 and np.isfinite(style.ttl).all(), voice.id


def test_clones_a_clip_at_any_rate(model, tmp_path):
    speech = model.synthesize('A reference clip of one voice, long enough to clone it from.', _noa(model), 'en',
                              seed=5)
    from bluetts_addon.vendor.blue_onnx.style import _resample_training
    clip = _resample_training(speech, model.sample_rate, 16000)  # as Subtitld's own clips are
    style = model.clone_style(clip, 16000)
    assert style.ttl.shape == _noa(model).ttl.shape and np.isfinite(style.ttl).all()
    wav = model.synthesize('And now the clone speaks.', style, 'en', seed=5)
    assert len(wav) / model.sample_rate > 0.8


def test_a_clip_under_a_second_is_refused(model):
    with pytest.raises(ValueError, match='at least 1 s'):
        model.clone_style(np.zeros(8000, dtype=np.float32), 16000)


def test_the_compact_hebrew_model_reads_too(model):
    try:
        model.set_renikud(addon.renikud_path('compact'))
        wav = model.synthesize('שלום, מה שלומך היום?', _noa(model), 'he', seed=1)
        assert len(wav) / model.sample_rate > 0.8
        assert model.tts.g2p.renikud_path.endswith('model_int8.onnx')
    finally:
        model.set_renikud(addon.renikud_path('accurate'))


def test_yiddish_digits_are_refused(model):
    with pytest.raises(ValueError, match='spell the numbers out'):
        model.synthesize('ער האָט 3 קינדער.', _noa(model), 'yi', seed=1)


def test_cancel_stops_between_flow_steps(model):
    calls = []
    flag = threading.Event()
    original = model.tts.vector_est_ort._session.run

    def run(*args, **kwargs):
        calls.append(1)
        flag.set()  # cancel during the first flow step
        return original(*args, **kwargs)

    model.tts.vector_est_ort._session.run = run
    try:
        with pytest.raises(engine.Cancelled):
            model.synthesize('A line that is cancelled as soon as it starts to be spoken.', _noa(model), 'en',
                             cancelled=flag.is_set)
    finally:
        del model.tts.vector_est_ort._session.run
    assert len(calls) == 1


def test_writes_a_44k_wav(model, tmp_path):
    from bluetts_addon import audio
    wav = model.synthesize('Written to disk.', _noa(model), 'en', seed=1)
    out = tmp_path / 'line.wav'
    audio.write_wav(out, wav, model.sample_rate)
    with wave.open(str(out)) as clip:
        assert clip.getframerate() == 44100 and clip.getnchannels() == 1 and clip.getnframes() == len(wav)
