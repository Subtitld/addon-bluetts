#!/usr/bin/env python3
"""Write bluetts_addon/pins.json: every file the add-on downloads, by size and sha256.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

    python tools/pin_models.py

Files are grouped by what needs them, and a group downloads the first time a
line needs it:

- core: the four synthesis graphs and their config, the 15 stock voices, and
  polyg2p's data (the G2P of English, German, Spanish, Italian and Russian,
  37 MB: small enough that every language gets it, which keeps the
  phonemizer from ever loading a partial data folder);
- clone: the reference encoder, for the speaker's own voice;
- he: RenikudPlus, the Hebrew G2P, as the release bundle ships it (1.2 GB);
- he-compact: the same model quantized to int8 (312 MB), from RenikudPlus's
  own repository at the revision the bundle's copy came from;
- yi: ReNikud-yi, the Yiddish context model. Its code ships in the add-on.

LFS files are pinned from Hugging Face's own metadata; small files are
downloaded and hashed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from vendor_upstream import BUNDLE_REPO, BUNDLE_REVISION  # noqa: E402

RENIKUD_REPO = 'notmax123/RenikudPlus'
RENIKUD_REVISION = '679c56ca449d41873fb8ff7711ddf7563d28198f'

ONNX = ['onnx/duration_predictor_style.onnx', 'onnx/text_encoder.onnx', 'onnx/vector_estimator.onnx',
        'onnx/vocoder.onnx', 'onnx/tts.json', 'onnx/vocab.json', 'onnx/stats.npz', 'onnx/uncond.npz']

# group -> (repo, revision, [paths or path prefixes ending in '/'])
GROUPS = {
    'core': (BUNDLE_REPO, BUNDLE_REVISION,
             ONNX + ['voices/', 'frontend/polyg2p_data/', 'LICENSE.OpenRAIL-M', 'LICENSES.md']),
    'clone': (BUNDLE_REPO, BUNDLE_REVISION, ['onnx/reference_encoder.onnx']),
    'he': (BUNDLE_REPO, BUNDLE_REVISION,
           ['frontend/renikud/model.onnx', 'frontend/renikud/LICENSE', 'frontend/renikud/README.md']),
    'he-compact': (RENIKUD_REPO, RENIKUD_REVISION, ['model_int8.onnx']),
    'yi': (BUNDLE_REPO, BUNDLE_REVISION,
           ['frontend/yi/onnx_renikud_yi/model.onnx', 'frontend/yi/onnx_renikud_yi/labels.json']),
}


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def tree(repo: str, revision: str) -> dict[str, dict]:
    items = json.loads(get(f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1'))
    return {item['path']: item for item in items if item.get('type') == 'file'}


def main() -> int:
    trees: dict[tuple[str, str], dict] = {}
    groups = {}
    for name, (repo, revision, wanted) in GROUPS.items():
        files_here = trees.setdefault((repo, revision), tree(repo, revision))
        paths = sorted(p for p in files_here
                       if any(p == w or (w.endswith('/') and p.startswith(w)) for w in wanted))
        missing = [w for w in wanted if not any(p == w or (w.endswith('/') and p.startswith(w)) for p in paths)]
        if missing:
            raise SystemExit(f'{name}: not in {repo}@{revision}: {missing}')
        files = []
        for path in paths:
            meta = files_here[path]
            lfs = meta.get('lfs') or {}
            sha = lfs.get('oid') or lfs.get('sha256')
            if not sha:
                data = get(f'https://huggingface.co/{repo}/resolve/{revision}/{path}')
                if len(data) != meta['size']:
                    raise SystemExit(f'{path}: got {len(data)} bytes, the tree says {meta["size"]}')
                sha = hashlib.sha256(data).hexdigest()
            files.append({'path': path, 'size': meta['size'], 'sha256': sha})
        groups[name] = {'repo': repo, 'revision': revision, 'files': files}
        print(f"{name}: {len(files)} files, {sum(f['size'] for f in files) / 1e6:,.0f} MB")
    out = ROOT / 'bluetts_addon' / 'pins.json'
    out.write_text(json.dumps({'groups': groups}, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
