"""manifest.json, pins.json, the vendored code and the add-on agree with each other.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bluetts_addon import ADDON_ID, __version__, voices  # noqa: E402
from bluetts_addon import __main__ as addon  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
PINS = json.loads((REPO_ROOT / 'bluetts_addon' / 'pins.json').read_text(encoding='utf-8'))['groups']
VENDOR = REPO_ROOT / 'bluetts_addon' / 'vendor'
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def test_versions_agree():
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert MANIFEST['version'] == __version__ == pyproject['project']['version']
    assert re.fullmatch(r'\d\d\.\d\d\.\d+', __version__), 'versions are YY.MM.N'
    assert MANIFEST['id'] == ADDON_ID


def test_manifest_lists_what_the_code_offers():
    assert MANIFEST['languages'] == list(voices.LANGUAGES)
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['tasks'] == ['tts.synthesize']
    assert MANIFEST['voice_clone'] is True and MANIFEST['compute'] == 'cpu'
    keys = {f['key'] for f in MANIFEST['config_schema']['fields']}
    assert keys == {'threads', 'steps', 'hebrew_g2p', 'voice_ref_audio'}
    [hebrew] = [f for f in MANIFEST['config_schema']['fields'] if f['key'] == 'hebrew_g2p']
    assert [o['value'] for o in hebrew['options']] == ['accurate', 'compact'] and hebrew['default'] == 'accurate'


def test_manifest_models_are_the_pins():
    by_group = {'bluev3': 'core', 'bluev3-reference-encoder': 'clone', 'renikudplus': 'he',
                'renikudplus-int8': 'he-compact', 'renikud-yi': 'yi'}
    assert [m['id'] for m in MANIFEST['models']] == list(by_group)
    for model in MANIFEST['models']:
        group = PINS[by_group[model['id']]]
        assert (model['repo'], model['revision']) == (group['repo'], group['revision'])
        assert model['size_mb'] == round(sum(f['size'] for f in group['files']) / 1e6)
        assert model['bundled'] is False


def test_pins_are_complete():
    assert set(PINS) == {'core', 'clone', 'he', 'he-compact', 'yi'}
    for name, group in PINS.items():
        assert re.fullmatch(r'[0-9a-f]{40}', group['revision']), name
        for pin in group['files']:
            assert _SHA256.match(pin['sha256']) and pin['size'] > 0, pin
    core = {p['path'] for p in PINS['core']['files']}
    graphs = {'duration_predictor_style.onnx', 'text_encoder.onnx', 'vector_estimator.onnx', 'vocoder.onnx'}
    assert {f'onnx/{g}' for g in graphs} | {'onnx/tts.json', 'onnx/vocab.json', 'LICENSE.OpenRAIL-M'} <= core
    assert {f'voices/{v.file}' for v in voices.stock()} <= core
    assert {'frontend/polyg2p_data/en-large.g2p', 'frontend/polyg2p_data/ru_accents.lex'} <= core
    assert {p['path'] for p in PINS['clone']['files']} == {'onnx/reference_encoder.onnx'}
    assert 'frontend/renikud/model.onnx' in {p['path'] for p in PINS['he']['files']}
    assert {p['path'] for p in PINS['yi']['files']} == {'frontend/yi/onnx_renikud_yi/model.onnx',
                                                        'frontend/yi/onnx_renikud_yi/labels.json'}
    # Everything the bundle repo serves lands in one revision folder, laid out as the bundle is.
    assert addon.model_root().name == PINS['core']['revision']
    assert addon.yi_model_dir().parent.parent == addon.model_root() / 'frontend'


def test_vendored_code_is_there():
    record = json.loads((REPO_ROOT / 'tools' / 'vendored.json').read_text(encoding='utf-8'))
    assert record['bundle_revision'] == PINS['core']['revision']
    for dest in record['files']:
        assert (VENDOR / dest).is_file(), dest
    assert 'MIT License' in (VENDOR / 'blue_onnx' / 'LICENSE').read_text(encoding='utf-8')
    assert 'CC BY-SA 4.0' in (VENDOR / 'yi' / 'NOTICE').read_text(encoding='utf-8')
    style = (VENDOR / 'blue_onnx' / 'style.py').read_text(encoding='utf-8')
    assert 'def release_style_from_samples' in style and '\nimport librosa' not in style


@pytest.mark.parametrize('tag,expected', [
    ('he', ('he', '')), ('he-IL', ('he', 'il')), ('iw', ('he', '')), ('ji', ('yi', '')),
    ('es_MX', ('es', 'mx')), ('es-419', ('es', '419')), ('EN-us', ('en', 'us')), ('', ('', '')), (None, ('', '')),
])
def test_language_tags(tag, expected):
    assert voices.split_tag(tag) == expected


def test_spanish_variant_follows_the_region():
    assert voices.spanish_variant('') == voices.spanish_variant('es') == 'es'
    assert voices.spanish_variant('mx') == voices.spanish_variant('419') == 'es-419'


def test_the_hosts_rate_is_the_speed():
    assert addon._speed({}) == 1.0
    assert addon._speed({'rate': 50}) == 1.5
    assert addon._speed({'rate': -100}) == 0.5
    assert addon._speed({'rate': 'fast'}) == 1.0


def test_download_groups_follow_the_line():
    assert addon.groups_for('en', 'Hello.', False, 'accurate') == ['core']
    assert addon.groups_for('en', 'Hello.', True, 'accurate') == ['core', 'clone']
    assert addon.groups_for('he', 'שלום', False, 'accurate') == ['core', 'he']
    assert addon.groups_for('he', 'שלום', False, 'compact') == ['core', 'he-compact']
    assert addon.groups_for('en', 'He said שלום.', False, 'accurate') == ['core', 'he']
    assert addon.groups_for('yi', 'גוט מאָרגן', False, 'accurate') == ['core', 'yi']
