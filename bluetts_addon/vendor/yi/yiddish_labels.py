from __future__ import annotations
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent

def _find_engine() -> Path:
    for cand in (_HERE, _HERE.parent):
        if (cand / 'yiddish_g2p.py').exists() and (cand / 'data').is_dir():
            return cand
    raise RuntimeError(f'cannot locate yiddish_g2p.py next to a data/ directory (looked in {_HERE} and {_HERE.parent})')
ENGINE_DIR = _find_engine()
for _p in (str(ENGINE_DIR), str(_HERE)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import yiddish_g2p as _g2p
_EXPECTED = {'GOLD_LEXICON': 502, '_AUDIO_PE': 77, '_AUDIO_VOWEL': 42, '_AUDIO_ENDORSED': 100, '_HOMOGRAPH_LK': 200, '_SEFARIA_POINTED': 3000, '_MODEL_POINTED': 7000}
_CANARIES = {'פעקל': 'pɛkl', 'יארצייט': 'jˈurʦajt', 'האף': 'huf', 'שבת': 'ʃˈabəs', 'א פאר יאר': 'a pˈur jur'}

def verify(strict: bool=True) -> dict:
    problems: list[str] = []
    sizes: dict[str, int] = {}
    for name, want in _EXPECTED.items():
        got = len(getattr(_g2p, name, {}) or {})
        sizes[name] = got
        if got < want:
            problems.append(f'{name}: {got} entries, expected >= {want}' + (' -- table missing from data/' if got == 0 else ''))
    readings: dict[str, str] = {}
    for word, want in _CANARIES.items():
        got = _g2p.hebrew_to_ipa(word, stress=True)
        readings[word] = got
        if got != want:
            problems.append(f'{word}: reads {got!r}, expected {want!r}')
    report = {'engine_dir': str(ENGINE_DIR), 'sizes': sizes, 'canaries': readings, 'problems': problems}
    if problems and strict:
        raise RuntimeError('Yiddish label stack is not correctly deployed:\n  ' + '\n  '.join(problems) + f'\n\nEngine loaded from {ENGINE_DIR}. The data/*.py tables must sit beside yiddish_g2p.py; without them the engine silently drops every native verdict and audio correction.')
    return report
verify()

def _install_context_reader() -> str:
    import os
    if os.environ.get('PHONIKUD_YI_RENIKUD', '1') == '0':
        return 'disabled (PHONIKUD_YI_RENIKUD=0)'
    try:
        import yiddish_renikud
    except Exception as exc:
        return f'unavailable ({exc!r})'
    if not yiddish_renikud.available():
        return 'no export found (onnx_renikud_yi/ beside yiddish_renikud.py)'
    _g2p.set_context_reader(yiddish_renikud.context_reader)
    return f'installed ({yiddish_renikud.model_dir()})'
CONTEXT_READER = _install_context_reader()

def text_to_ipa(text: str) -> str:
    return _g2p.hebrew_to_ipa(text, stress=True)

def token_detail(word: str) -> dict:
    return _g2p.g2p_token(word)

def text_to_nikud(text: str) -> str:
    from yiddish_nikud import add_nikud
    return add_nikud(text)

def text_to_nikud_batch(texts: list[str]) -> list[str]:
    from yiddish_nikud import add_nikud_batch
    return add_nikud_batch(texts)
if __name__ == '__main__':
    import json
    print(json.dumps(verify(strict=False), ensure_ascii=False, indent=1))
