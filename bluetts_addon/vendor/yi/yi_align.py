from __future__ import annotations
import functools
import sys
import unicodedata
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
CONSONANTS: tuple[str, ...] = ('b', 'd', 'f', 'ɡ', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'z', 'x', 'ʃ', 'ʒ', 'ʦ', 'ʧ', 'ʤ', 'ŋ')
VOWELS: tuple[str, ...] = ('aː', 'ej', 'aj', 'ɔj', 'oʊ', 'a', 'ɛ', 'ə', 'i', 'u', 'ɔ')
STRESS = 'ˈ'
MARKERS = ("'", '"', '-')
LETTER: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {'א': (('',), VOWELS + ('',)), 'ב': (('b', 'v', 'p'), VOWELS + ('',)), 'ג': (('ɡ', 'k'), VOWELS + ('',)), 'ד': (('d', 't', ''), VOWELS + ('',)), 'ה': (('h', ''), VOWELS + ('',)), 'ו': (('v', 'f', ''), VOWELS + ('',)), 'ז': (('z', 's', 'ʒ'), VOWELS + ('',)), 'ח': (('x',), VOWELS + ('',)), 'ט': (('t',), VOWELS + ('',)), 'י': (('j', ''), VOWELS + ('',)), 'כ': (('k', 'x'), VOWELS + ('',)), 'ל': (('l', ''), VOWELS + ('',)), 'מ': (('m',), VOWELS + ('',)), 'נ': (('n', 'ŋ'), VOWELS + ('',)), 'ס': (('s', 'z'), VOWELS + ('',)), 'ע': (('',), VOWELS + ('',)), 'פ': (('f', 'p'), VOWELS + ('',)), 'צ': (('ʦ', 'ʧ'), VOWELS + ('',)), 'ק': (('k',), VOWELS + ('',)), 'ר': (('r', ''), VOWELS + ('',)), 'ש': (('ʃ', 's', 'ʒ'), VOWELS + ('',)), 'ת': (('t', 's'), VOWELS + ('',))}
_FINALS = {'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}
_LIGATURES = {'ײ': 'יי', 'ױ': 'וי', 'װ': 'וו'}
DIGRAPHS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {'דזש': (('ʤ',), ('',)), 'וו': (('v',), VOWELS + ('',)), 'יי': (('',), ('aj', 'ej', 'aː', 'i')), 'וי': (('',), ('ɔj', 'oʊ', 'u')), 'זש': (('ʒ',), VOWELS + ('',)), 'טש': (('ʧ',), VOWELS + ('',)), 'תש': (('ʧ',), VOWELS + ('',)), 'טס': (('ʦ',), VOWELS + ('',)), 'טצ': (('ʦ',), VOWELS + ('',))}

def normalize_word(word: str) -> str:
    w = unicodedata.normalize('NFD', word)
    w = ''.join((ch for ch in w if unicodedata.category(ch) != 'Mn'))
    w = unicodedata.normalize('NFC', w)
    w = w.translate(str.maketrans(_FINALS)).translate(str.maketrans(_LIGATURES))
    return w.replace('׳', "'").replace('״', '"')

def _chunks_for(cons: tuple[str, ...], vows: tuple[str, ...], rest: str):
    for c in cons:
        if c and (not rest.startswith(c)):
            continue
        n = len(c)
        for stressed in (True, False):
            s = 0
            if stressed:
                if rest[n:n + 1] != STRESS:
                    continue
                s = 1
            for v in vows:
                if v:
                    if not rest[n + s:].startswith(v):
                        continue
                    L = n + s + len(v)
                elif stressed:
                    continue
                else:
                    L = n
                yield (rest[:L], L)
    for c in cons:
        if not c:
            continue
        for stressed in (True, False):
            s = 1 if stressed else 0
            if stressed and (not rest.startswith(STRESS)):
                continue
            for v in vows:
                if not v or not rest[s:].startswith(v):
                    continue
                L = s + len(v)
                if rest[L:].startswith(c):
                    yield (rest[:L + len(c)], L + len(c))

def align_word(word: str, ipa: str) -> list[tuple[str, str]] | None:
    w = normalize_word(word)
    if ' ' in ipa.strip():
        return None
    ipa = ipa.replace(' ', '').replace('-', '')

    @functools.lru_cache(maxsize=None)
    def search(h: int, i: int):
        if h == len(w) and i == len(ipa):
            return ()
        if h == len(w):
            return None
        ch = w[h]
        if ch in MARKERS:
            r = search(h + 1, i)
            return ((ch, ''),) + r if r is not None else None
        rest = ipa[i:]
        for dg, (cons, vows) in DIGRAPHS.items():
            if w.startswith(dg, h):
                for chunk, L in _chunks_for(cons, vows, rest):
                    r = search(h + len(dg), i + L)
                    if r is not None:
                        return ((dg[0], chunk),) + tuple(((c, '') for c in dg[1:])) + r
        if ch not in LETTER:
            return None
        cons, vows = LETTER[ch]
        for chunk, L in _chunks_for(cons, vows, rest):
            r = search(h + 1, i + L)
            if r is not None:
                return ((ch, chunk),) + r
        return None
    if len(w) > 40 or len(ipa) > 80:
        return None
    try:
        out = search(0, 0)
    except RecursionError:
        return None
    return list(out) if out is not None else None

def parse_chunk(chunk: str) -> tuple[str, str, int]:
    stress = int(STRESS in chunk)
    c = chunk.replace(STRESS, '')
    cons = ''
    for k in sorted(CONSONANTS, key=len, reverse=True):
        if c.startswith(k):
            cons = k
            c = c[len(k):]
            break
    if not cons:
        for k in sorted(CONSONANTS, key=len, reverse=True):
            if c.endswith(k) and c[:-len(k)] in VOWELS:
                cons = k
                c = c[:-len(k)]
                break
    vowel = c
    if vowel and vowel not in VOWELS:
        raise ValueError(f'bad chunk {chunk!r}')
    return (cons, vowel, stress)
