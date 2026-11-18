#!/usr/bin/env python3
"""Write the notices of polyg2p and the Rust crates its extension links statically.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

cargo-about (https://github.com/EmbarkStudios/cargo-about) reads the Cargo.lock
of polyg2p's Python bindings at the pinned commit and gathers each crate's
licence text, for the targets the add-on ships (tools/cargo-about/about.toml).
A crate under a licence not accepted there fails the run. polyg2p's own
LICENSE and NOTICE are copied beside it.

The result is committed in THIRD_PARTY_LICENSES/polyg2p/; run this again
whenever tools/get_polyg2p.py moves to another commit.

    cargo install --locked cargo-about --features cli
    git clone https://github.com/maxmelichov/polyg2p /tmp/polyg2p
    git -C /tmp/polyg2p checkout <POLYG2P_COMMIT>
    python tools/crate_licences.py /tmp/polyg2p
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ABOUT = REPO_ROOT / 'tools' / 'cargo-about'
OUT = REPO_ROOT / 'THIRD_PARTY_LICENSES' / 'polyg2p'


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('upstream', type=Path, help='a checkout of maxmelichov/polyg2p at the pinned commit')
    args = parser.parse_args(argv)

    commit = runpy.run_path(str(REPO_ROOT / 'tools' / 'get_polyg2p.py'), run_name='crate_licences')['POLYG2P_COMMIT']
    head = subprocess.run(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != commit:
        raise SystemExit(f'FATAL: {args.upstream} is at {head}, expected {commit}')

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        template = Path(tmp) / 'about.hbs'
        template.write_text((ABOUT / 'about.hbs').read_text(encoding='utf-8').replace('@COMMIT@', commit),
                            encoding='utf-8')
        subprocess.run(['cargo', 'about', 'generate', '--locked', '--fail', '-c', str(ABOUT / 'about.toml'),
                        str(template), '-o', str(OUT / 'CRATES.txt')],
                       cwd=args.upstream / 'bindings' / 'python', check=True)
    for name in ('LICENSE', 'NOTICE'):
        shutil.copyfile(args.upstream / name, OUT / name)
    print(f"wrote {OUT} (CRATES.txt {(OUT / 'CRATES.txt').stat().st_size:,} bytes)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
