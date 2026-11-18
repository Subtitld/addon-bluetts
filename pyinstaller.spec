# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld BlueTTS add-on.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Produces `dist/bluetts-addon/` (launcher + `_internal/`): ONNX Runtime, numpy,
polyg2p, RenikudPlus's runtime, the add-on and the upstream code it vendors.
No model is in it; the files download the first time a line needs them. The
release step copies the folder to `bin/` inside the archive, next to
manifest.json.

Run from the add-on root:

    pyinstaller pyinstaller.spec --noconfirm
    dist/bluetts-addon/bluetts-addon --self-test
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SPEC_ROOT = Path(SPECPATH).resolve()
PACKAGE = SPEC_ROOT / 'bluetts_addon'
VENDOR = PACKAGE / 'vendor'

# Fail the build, loudly, if the pins or the vendored code are missing.
# Otherwise the frozen add-on would answer the handshake and then fail the
# first line.
_REQUIRED = [PACKAGE / 'pins.json', VENDOR / 'blue_onnx' / '__init__.py', VENDOR / 'blue_onnx' / 'vocab.json',
             VENDOR / 'blue_onnx' / 'LICENSE', VENDOR / 'yi' / 'yiddish_labels.py', VENDOR / 'yi' / 'NOTICE',
             VENDOR / 'yi' / 'data' / 'lexicons' / 'gold_lexicon.py']
_MISSING = [str(p) for p in _REQUIRED if not p.is_file()]
if _MISSING:
    raise SystemExit(f'FATAL: missing {", ".join(_MISSING)}. Run tools/vendor_upstream.py.')

datas = [
    (str(PACKAGE / 'pins.json'), 'bluetts_addon'),
    (str(VENDOR / 'blue_onnx' / 'vocab.json'), 'bluetts_addon/vendor/blue_onnx'),
    (str(VENDOR / 'blue_onnx' / 'LICENSE'), 'bluetts_addon/vendor/blue_onnx'),
    # The Yiddish front end is plain files on disk: blue_onnx puts its folder on
    # sys.path, and it loads its lexicon tables by file path.
    (str(VENDOR / 'yi'), 'bluetts_addon/vendor/yi'),
]


# ---------------------------------------------------------------------------
# libexpat sanity check (from addon-whispercpp).
#
# PyInstaller bundles whatever expat it resolved on the build host, for
# CPython's pyexpat. If that host has an OLD expat shadowing the distro one
# (for example a hand-built 2.5.0 in /usr/local/lib), the frozen binary dies
# before reaching Python:
#     Failed to load libpython3.x.so: undefined symbol: XML_SetReparseDeferralEnabled
# So verify the bundled copy exports the symbol libpython needs, and substitute
# a known-good system copy if it does not.
# ---------------------------------------------------------------------------
_EXPAT_SYMBOL = 'XML_SetReparseDeferralEnabled'


def _exports_symbol(path: str, symbol: str) -> bool:
    try:
        out = subprocess.run(['nm', '-D', '--defined-only', path],
                             capture_output=True, text=True, timeout=30)
        return symbol in out.stdout
    except Exception:
        return True  # no nm (macOS/Windows): assume fine rather than break the build


def _fix_libexpat(entries):
    if not sys.platform.startswith('linux'):
        return entries
    out = []
    for entry in entries:
        name = entry[0] if entry else ''
        src = entry[1] if len(entry) > 1 else ''
        if not os.path.basename(name).startswith('libexpat.so') or _exports_symbol(src, _EXPAT_SYMBOL):
            out.append(entry)
            continue
        for cand in ('/usr/lib/x86_64-linux-gnu/libexpat.so.1', '/usr/lib64/libexpat.so.1',
                     '/lib/x86_64-linux-gnu/libexpat.so.1'):
            if os.path.exists(cand) and _exports_symbol(cand, _EXPAT_SYMBOL):
                print(f'WARNING: {src} lacks {_EXPAT_SYMBOL}; substituting {cand}')
                out.append((name, cand) + tuple(entry[2:]))
                break
        else:
            print(f'ERROR: bundled libexpat lacks {_EXPAT_SYMBOL}; the frozen binary will not start.')
            out.append(entry)
    return out


a = Analysis(
    [str(PACKAGE / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=[],
    datas=datas,
    # Imported inside functions, so the handshake does not wait for numpy;
    # and what the vendored Yiddish front end imports from the standard
    # library, which PyInstaller cannot see in files it does not analyse.
    hiddenimports=['bluetts_addon.engine', 'bluetts_addon.vendor.blue_onnx', 'bluetts_addon.vendor.blue_onnx.style',
                   'bluetts_addon.vendor.blue_onnx.ipa_text', 'bluetts_addon.vendor.blue_onnx.text_norm',
                   'renikud_onnx', 'hebrew_num2words', 'polyg2p', 'num2words',
                   'itertools', 'hashlib', 'functools', 'collections', 'unicodedata', 'importlib.util'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # num2words is LGPL-2.1: its modules stay readable, replaceable .py files
    # in _internal/num2words/ instead of entering the archive.
    module_collection_mode={'num2words': 'py'},
    excludes=[
        # ONNX Runtime's optional parts: training, the converters and tools,
        # and what only they import. Inference needs none of it.
        'onnxruntime.training', 'onnxruntime.tools', 'onnxruntime.transformers',
        'onnxruntime.quantization', 'onnx', 'sympy', 'coloredlogs', 'humanfriendly',
        # RenikudPlus's own downloader, which the add-on never calls: it
        # downloads the pinned weights itself. hf_xet is a native library.
        'huggingface_hub', 'hf_xet', 'httpx', 'httpx2', 'httpcore', 'httpcore2', 'anyio', 'tqdm', 'fsspec',
        'yaml', 'filelock', 'truststore',
        # What blue_onnx imports only on paths the add-on never takes: the
        # blue-onnx-v2 voice extractor, file reading, eSpeak.
        'librosa', 'soundfile', 'phonemizer', 'espeakng_loader',
        'torch', 'scipy', 'matplotlib', 'pandas', 'IPython',
        'PySide6', 'PyQt6', 'PyQt5', 'tkinter', 'Tkinter', '_tkinter',
        'pytest', '_pytest', 'pluggy', 'iniconfig', 'setuptools', 'pkg_resources',
        'unittest', 'pydoc_data',
        # Standard modules the add-on never calls, reached only through
        # optional imports. Each would bring a native library, and with it a
        # licence to ship. ssl and hashlib stay: the models download over
        # HTTPS and are checked by sha256.
        # decimal stays: num2words reads numbers with it.
        'bz2', '_bz2', 'lzma', '_lzma', 'compression', '_zstd',
        # GPL (readline) and ncurses: for interactive terminals, which the
        # add-on never has.
        'readline', 'curses', '_curses', '_curses_panel',
    ],
    noarchive=False,
)

a.binaries = _fix_libexpat(a.binaries)
a.binaries = [entry for entry in a.binaries if not os.path.basename(entry[0]).startswith('libmvec.so')]

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='bluetts-addon',
    debug=False,
    bootloader_ignore_signals=False,
    # strip=False: stripping the native libraries saves little and breaks
    # macOS code signatures.
    strip=False,
    upx=False,
    console=True,  # stdio is the protocol channel
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='bluetts-addon',
)
