#!/usr/bin/env python3
"""Copy upstream's inference code into bluetts_addon/vendor/, at pinned revisions.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

    python tools/vendor_upstream.py

Two pieces are vendored rather than installed:

- `blue_onnx`, the inference package of BlueTTS (MIT), from GitHub at
  BLUETTS_COMMIT. It is not on PyPI, and its own dependencies (librosa,
  soundfile, phonemizer, onnx) are only needed by paths the add-on never takes.
- The Yiddish front end of the BlueV3 release bundle (MIT, two lexicon tables
  CC BY-SA 4.0), from Hugging Face at BUNDLE_REVISION. The bundle ships it as
  Python files next to its model; the add-on carries the files and downloads
  only the model, so no code is ever fetched at run time.

Each file is checked against the sha256 in EXPECTED when one is recorded, so a
re-run that gets different bytes fails instead of silently changing the
add-on. Then the patches in tools/patches/ are applied, in name order.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'bluetts_addon' / 'vendor'
PATCHES = ROOT / 'tools' / 'patches'

BLUETTS_REPO = 'maxmelichov/BlueTTS'
BLUETTS_COMMIT = 'cbe51508f3f05207662bc714ad646da8191861d0'
BUNDLE_REPO = 'notmax123/BlueV3-onnx'
BUNDLE_REVISION = '39c4843019442bbf67e3980242549504b8583f2c'

BLUE_ONNX_FILES = ['__init__.py', 'ipa_text.py', 'text_norm.py', 'style.py', 'vocab.json']
YI_FILES = ['NOTICE', 'yi_align.py', 'yiddish_g2p.py', 'yiddish_labels.py', 'yiddish_renikud.py']
YI_LEXICONS = ['audio_endorsed_lk.py', 'audio_pe_lk.py', 'audio_vowel_lk.py', 'gold_lexicon.py',
               'homograph_lk.py', 'model_pointed_lk.py', 'sefaria_pointed_lk.py', 'stress_overrides.py']


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def sources() -> list[tuple[str, str]]:
    """(url, destination relative to vendor/) for every vendored file."""
    gh = f'https://raw.githubusercontent.com/{BLUETTS_REPO}/{BLUETTS_COMMIT}'
    hf = f'https://huggingface.co/{BUNDLE_REPO}/resolve/{BUNDLE_REVISION}'
    out = [(f'{gh}/LICENSE', 'blue_onnx/LICENSE')]
    out += [(f'{gh}/src/blue_onnx/{name}', f'blue_onnx/{name}') for name in BLUE_ONNX_FILES]
    out += [(f'{hf}/frontend/yi/{name}', f'yi/{name}') for name in YI_FILES]
    out += [(f'{hf}/frontend/yi/data/lexicons/{name}', f'yi/data/lexicons/{name}') for name in YI_LEXICONS]
    return out


def main() -> int:
    expected_file = ROOT / 'tools' / 'vendored.json'
    expected = json.loads(expected_file.read_text(encoding='utf-8')) if expected_file.is_file() else {}
    record = {'bluetts_commit': BLUETTS_COMMIT, 'bundle_revision': BUNDLE_REVISION, 'files': {}}
    if VENDOR.exists():
        shutil.rmtree(VENDOR)
    for url, dest in sources():
        data = get(url)
        digest = hashlib.sha256(data).hexdigest()
        want = expected.get('files', {}).get(dest)
        if want and want != digest and expected.get('bluetts_commit') == BLUETTS_COMMIT \
                and expected.get('bundle_revision') == BUNDLE_REVISION:
            raise SystemExit(f'{dest}: upstream bytes changed at the same revision ({digest} != {want})')
        path = VENDOR / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        record['files'][dest] = digest
    (VENDOR / '__init__.py').write_text('"""Upstream code, copied by tools/vendor_upstream.py: do not edit by hand."""\n',
                                        encoding='utf-8')
    for patch in sorted(PATCHES.glob('*.patch')):
        subprocess.run(['git', 'apply', '--directory', 'bluetts_addon/vendor', str(patch)], cwd=ROOT, check=True)
        print('applied', patch.name)
    expected_file.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    print(f"{len(record['files'])} files vendored")
    return 0


if __name__ == '__main__':
    sys.exit(main())
