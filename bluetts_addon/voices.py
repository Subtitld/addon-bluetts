"""The voices the add-on offers, and the languages BlueTTS speaks.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

- `bluetts-clone` clones the speaker from the video: the host hands over a clip
  of their speech (or Configure names one), and BlueV3's reference encoder
  turns it into a voice.
- The 15 stock voices are the release bundle's voice files, made from
  LibriTTS-R speakers (CC-BY-4.0). They download with the model. Every voice
  speaks every language, each at its own calibrated pace.
"""

from __future__ import annotations

from dataclasses import dataclass

CLONE = 'bluetts-clone'
LANGUAGES = ('he', 'en', 'de', 'es', 'it', 'ru', 'yi')
LANGUAGE_NAMES = {'he': 'Hebrew', 'en': 'English', 'de': 'German', 'es': 'Spanish', 'it': 'Italian',
                  'ru': 'Russian', 'yi': 'Yiddish'}
# Older ISO 639-1 codes for Hebrew and Yiddish, still in use.
ALIASES = {'iw': 'he', 'ji': 'yi'}
# Spanish regions read as Castilian; any other region is Latin American (seseo, yeísmo).
CASTILIAN_REGIONS = {'', 'es', 'ad', 'gq'}

SAMPLE_RATE = 44100

_STOCK = (
    ('noa', 'woman'), ('maya', 'woman'), ('lily', 'woman'), ('tamar', 'woman'), ('shira', 'woman'),
    ('yael', 'woman'), ('emma', 'woman'), ('sara', 'woman'),
    ('adam', 'man'), ('daniel', 'man'), ('david', 'man'), ('jonathan', 'man'), ('ethan', 'man'),
    ('michael', 'man'), ('ariel', 'man'),
)


@dataclass(frozen=True)
class Voice:
    id: str
    display_name: str
    file: str = ''   # voices/<file> in the release bundle, empty for the clone


def stock() -> tuple[Voice, ...]:
    return tuple(Voice(f'bluetts-{name}', f'{name.capitalize()} ({kind})', f'{name}.json') for name, kind in _STOCK)


def all_voices() -> tuple[Voice, ...]:
    return (Voice(CLONE, "Speaker's own voice (cloned from the video)"), *stock())


def by_id(voice_id: str) -> Voice | None:
    return next((v for v in all_voices() if v.id == voice_id), None)


def manifest_voices() -> list[dict]:
    """The voice list as manifest.json and the hello frame carry it."""
    out = []
    for voice in all_voices():
        entry = {'id': voice.id, 'language': '*', 'display_name': voice.display_name, 'sample_rate': SAMPLE_RATE}
        if voice.id == CLONE:
            entry['requires'] = ['voice_ref_audio']
        out.append(entry)
    return out


def split_tag(tag) -> tuple[str, str]:
    """`es-MX` -> ('es', 'mx'); `iw` -> ('he', ''). Lower-cased; ('', '') when there is none."""
    parts = str(tag or '').strip().lower().replace('_', '-').split('-')
    primary = ALIASES.get(parts[0], parts[0])
    region = next((p for p in parts[1:] if len(p) == 2 or p.isdigit()), '')
    return primary, region


def spanish_variant(region: str) -> str:
    """polyg2p's Spanish: `es` (Castilian) for Spain, `es-419` for the Americas."""
    return 'es' if region in CASTILIAN_REGIONS else 'es-419'
