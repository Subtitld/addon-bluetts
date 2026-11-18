#!/usr/bin/env python3
"""Install polyg2p 0.1.0, the G2P BlueV3 was trained with, for this machine.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

    python tools/get_polyg2p.py [--wheel-dir DIR]

polyg2p (Apache-2.0, Rust with Python bindings) is not on PyPI. The BlueV3
release bundle ships wheels for Linux x86-64 and macOS (arm64, x86-64); this
downloads the one for this machine at the bundle's pinned revision and checks
its sha256. Elsewhere (Windows) it builds the wheel from polyg2p's source at
the commit the bundle's wheels were built from, with maturin, which needs a
Rust toolchain. Either way the wheel is installed with pip and left in
--wheel-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vendor_upstream import BUNDLE_REPO, BUNDLE_REVISION  # noqa: E402

POLYG2P_REPO = 'https://github.com/maxmelichov/polyg2p'
POLYG2P_COMMIT = 'cbb613499adcda2d3039861c7d347c902e60fbd0'  # MANIFEST.json git.polyg2p_commit
MATURIN = 'maturin==1.16.0'

# (sys.platform, machine) -> (wheel in the bundle's frontend/, sha256)
WHEELS = {
    ('linux', 'x86_64'): ('polyg2p-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl',
                          'c32261120e83ed4d84cef433ae8bbb4cdab40e6e835eeea6483a0423c5193e37'),
    ('darwin', 'arm64'): ('polyg2p-0.1.0-cp39-abi3-macosx_11_0_arm64.whl',
                          '5e51041ea8724383444fa007b596cadacddde973e78b7aeba3096cbc3db8a7df'),
    ('darwin', 'x86_64'): ('polyg2p-0.1.0-cp39-abi3-macosx_10_12_x86_64.whl',
                           '1f6add2a4acc40cdae1e4236d7176b7f852b0fe2043301f5ffee897b20ef2a6a'),
}


def run(*cmd, cwd=None) -> None:
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def download(out: Path) -> Path | None:
    key = (sys.platform, platform.machine().lower().replace('amd64', 'x86_64'))
    if key not in WHEELS:
        return None
    name, sha = WHEELS[key]
    url = f'https://huggingface.co/{BUNDLE_REPO}/resolve/{BUNDLE_REVISION}/frontend/{name}'
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f'{name}: sha256 does not match the pin')
    path = out / name
    path.write_bytes(data)
    print(f'downloaded {name} ({len(data):,} bytes)')
    return path


def build(out: Path) -> Path:
    out = out.resolve()  # maturin runs in the bindings folder: a relative --out would land there
    with tempfile.TemporaryDirectory(prefix='polyg2p-') as tmp:
        src = Path(tmp) / 'polyg2p'
        run('git', 'init', '-q', src)
        run('git', 'fetch', '-q', '--depth', '1', POLYG2P_REPO, POLYG2P_COMMIT, cwd=src)
        run('git', 'checkout', '-q', 'FETCH_HEAD', cwd=src)
        run(sys.executable, '-m', 'pip', 'install', MATURIN)
        run(sys.executable, '-m', 'maturin', 'build', '--release', '--locked', '--out', out,
            cwd=src / 'bindings' / 'python')
    [wheel] = sorted(out.glob('polyg2p-0.1.0-*.whl'))
    return wheel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--wheel-dir', type=Path, default=Path('build') / 'polyg2p')
    args = parser.parse_args()
    args.wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel = download(args.wheel_dir) or build(args.wheel_dir)
    run(sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', wheel)
    return 0


if __name__ == '__main__':
    sys.exit(main())
