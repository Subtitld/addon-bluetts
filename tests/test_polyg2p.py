"""The installed polyg2p reads as the BlueV3 bundle's own wheel does.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

BlueV3 was trained on polyg2p's IPA, so a polyg2p that phonemizes differently
makes the model read worse. Where the bundle has no wheel (Windows), the
release builds one from source (tools/get_polyg2p.py); this checks it, and
every other platform's wheel, against what the bundle's Linux wheel gives on
100 Tatoeba sentences per language (tests/fixtures/polyg2p_parity.json). Needs
polyg2p's data, which downloads with the model (the core group, 428 MB).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bluetts_addon import __main__ as addon, download  # noqa: E402

pytestmark = pytest.mark.model
FIXTURE = json.loads((REPO_ROOT / 'tests' / 'fixtures' / 'polyg2p_parity.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def phonemizer():
    download.ensure(addon.pins('core'), lambda *_: None, lambda: False)
    os.environ['POLYG2P_EN_MODEL'] = 'large'
    from polyg2p import Phonemizer
    return Phonemizer(data_dir=str(addon.model_root() / 'frontend' / 'polyg2p_data'))


@pytest.mark.parametrize('language', sorted(FIXTURE['lines']))
def test_reads_as_the_bundle_wheel(phonemizer, language):
    lines = FIXTURE['lines'][language]
    differ = [(text, want, got) for text, want in lines if (got := phonemizer.phonemize(text, language)) != want]
    assert not differ, f'{len(differ)} of {len(lines)} lines differ, e.g. {differ[:3]}'
