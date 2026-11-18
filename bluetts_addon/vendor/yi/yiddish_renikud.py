from __future__ import annotations
import itertools
import json
import os
import re
import sys
import unicodedata as ud
from pathlib import Path
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / 'scripts'):
    if (_p / 'yi_align.py').exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
_CANDIDATES = (os.environ.get('PHONIKUD_YI_RENIKUD_MODEL', ''), str(_HERE / 'onnx_renikud_yi'), str(_HERE.parent / 'models' / 'renikud_yi_audio' / 'onnx_int8'), str(_HERE.parent / 'models' / 'renikud_yi_audio' / 'onnx'))
_OPEN_CLASSES = (frozenset({'a', 'ɔ', 'u', 'aː'}), frozenset({'f', 'p'}), frozenset({'aj', 'aː', 'ej'}), frozenset({'ɔj', 'oʊ'}), frozenset({'i', 'u'}), frozenset({'ɛ', 'ə'}))
_FINAL_DEVOICE = {'b': 'p', 'd': 't', 'ɡ': 'k', 'v': 'f', 'z': 's'}
_MAX_CANDIDATES = 64
_MULTI = ('aː', 'ej', 'aj', 'ɔj', 'oʊ')
_WS = re.compile('(\\s+)')

def model_dir() -> str | None:
    for cand in _CANDIDATES:
        if cand and os.path.exists(os.path.join(cand, 'model.onnx')):
            return cand
    return None

def available() -> bool:
    return model_dir() is not None

def _has_marks(word: str) -> bool:
    return any((ud.category(c) == 'Mn' for c in ud.normalize('NFKC', word)))

def _base_text(text: str) -> str:
    return ''.join((c for c in ud.normalize('NFKC', text) if ud.category(c) != 'Mn')).lower()

def graph_candidates(reading: list[str], max_candidates: int=_MAX_CANDIDATES) -> list[list[str]]:
    slots: list[list[str]] = []
    n = len(reading)
    for i, p in enumerate(reading):
        alts = {p}
        for cls in _OPEN_CLASSES:
            if p in cls:
                alts |= cls
        if i == n - 1 and p in _FINAL_DEVOICE:
            alts.add(_FINAL_DEVOICE[p])
        slots.append(sorted(alts, key=lambda x: (x != p, x)))
    out: list[list[str]] = []
    for combo in itertools.product(*slots):
        out.append(list(combo))
        if len(out) >= max_candidates:
            break
    return out

def restress(engine_ipa: str, phones: list[str], inventory: set[str]) -> str:
    idx = None
    seen = 0
    s = engine_ipa.replace(' ', '')
    i = 0
    while i < len(s):
        if s[i] == 'ˈ':
            idx = seen
            i += 1
            continue
        for m in _MULTI:
            if s.startswith(m, i):
                seen += 1
                i += len(m)
                break
        else:
            if s[i] in inventory:
                seen += 1
            i += 1
    out = []
    for k, p in enumerate(phones):
        if idx is not None and k == idx:
            out.append('ˈ')
        out.append(p)
    return ''.join(out)

class ReNikudYi:

    def __init__(self, model_dir_: str | None=None, providers: list[str] | None=None, threads: int | None=None):
        import onnxruntime as ort
        self.model_dir = model_dir_ or model_dir()
        if not self.model_dir:
            raise RuntimeError('no ReNikud-yi export found (model.onnx in: ' + ', '.join((c for c in _CANDIDATES if c)) + ')')
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self._session = ort.InferenceSession(os.path.join(self.model_dir, 'model.onnx'), so, providers=providers or ['CPUExecutionProvider'])
        meta = self._session.get_modelmeta().custom_metadata_map
        missing = {'vocab', 'consonants', 'vowels', 'cls_id', 'sep_id', 'pad_id', 'unk_id'} - set(meta)
        if missing:
            raise RuntimeError(f'{self.model_dir}/model.onnx lacks runtime metadata {sorted(missing)}; re-export with scripts/export_renikud_onnx.py')
        self._vocab = {c: i for i, c in enumerate(json.loads(meta['vocab']))}
        self.consonants: list[str] = json.loads(meta['consonants'])
        self.vowels: list[str] = json.loads(meta['vowels'])
        self._c2i = {c: i for i, c in enumerate(self.consonants)}
        self._v2i = {v: i for i, v in enumerate(self.vowels)}
        self.inventory = {p for p in self.consonants + self.vowels if p}
        self._cls, self._sep, self._pad, self._unk = (int(meta[k]) for k in ('cls_id', 'sep_id', 'pad_id', 'unk_id'))
        self._chunk = int(meta.get('max_len', 512)) - 8
        self.providers = self._session.get_providers()

    def _split(self, base: str) -> list[str]:
        if len(base) <= self._chunk:
            return [base]
        out: list[str] = []
        cur = ''
        for piece in _WS.split(base):
            if cur and len(cur) + len(piece) > self._chunk:
                out.append(cur)
                cur = piece
            else:
                cur += piece
        if cur:
            out.append(cur)
        return out

    def logprobs(self, base: str):
        import numpy as np
        segs = self._split(base)
        ids = [[self._cls] + [self._vocab.get(c, self._unk) for c in s] + [self._sep] for s in segs]
        width = max((len(r) for r in ids))
        input_ids = np.full((len(ids), width), self._pad, dtype=np.int64)
        attention = np.zeros((len(ids), width), dtype=np.int64)
        for r, row in enumerate(ids):
            input_ids[r, :len(row)] = row
            attention[r, :len(row)] = 1
        cons, vow, st = self._session.run(None, {'input_ids': input_ids, 'attention_mask': attention})
        lc = np.concatenate([_log_softmax(cons[r, 1:1 + len(s)]) for r, s in enumerate(segs)])
        lv = np.concatenate([_log_softmax(vow[r, 1:1 + len(s)]) for r, s in enumerate(segs)])
        ls = np.concatenate([_log_softmax(st[r, 1:1 + len(s)])[:, 1] for r, s in enumerate(segs)])
        return (lc, lv, ls)

    def tokenize(self, ipa: str) -> list[str]:
        out: list[str] = []
        s = ipa.replace(' ', '')
        i = 0
        while i < len(s):
            ch = s[i]
            if ch in 'ˈˌ':
                i += 1
                continue
            for m in _MULTI:
                if s.startswith(m, i):
                    out.append(m)
                    i += len(m)
                    break
            else:
                if ch in self.inventory:
                    out.append(ch)
                i += 1
        return out

    def score(self, word: str, phones: list[str], start: int, lc, lv, ls=None) -> tuple[float, int | None]:
        from yi_align import align_word, parse_chunk
        aligned = align_word(word, ''.join(phones))
        if aligned is None or len(aligned) != len(word):
            return (float('-inf'), None)
        total = 0.0
        n_v = 0
        full: list[tuple[int, float]] = []
        for k, (_, chunk) in enumerate(aligned):
            cons, vowel, _ = parse_chunk(chunk) if chunk else ('', '', 0)
            total += float(lc[start + k, self._c2i.get(cons, 0)]) + float(lv[start + k, self._v2i.get(vowel, 0)])
            if vowel:
                if vowel != 'ə' and ls is not None:
                    full.append((n_v, float(ls[start + k])))
                n_v += 1
        return (total, max(full, key=lambda x: x[1])[0] if len(full) >= 2 else None)

    def place_stress(self, phones: list[str], ordinal: int | None) -> str:
        out, n_v = ([], 0)
        for p in phones:
            if p in self._v2i and p:
                if ordinal is not None and n_v == ordinal:
                    out.append('ˈ')
                n_v += 1
            out.append(p)
        return ''.join(out)

    def free_reading(self, start: int, end: int, lc, lv) -> list[str]:
        out: list[str] = []
        for k in range(start, end):
            c = self.consonants[int(lc[k].argmax())]
            v = self.vowels[int(lv[k].argmax())]
            if c:
                out.append(c)
            if v:
                out.append(v)
        return out

    def read_records(self, text: str, records: list[dict]) -> list[dict]:
        base = _base_text(text)
        if not any((r.get('route') == 'rule' and r.get('ipa_primary') for r in records)):
            return records
        lc, lv, ls = self.logprobs(base)
        cursor = 0
        out = list(records)
        for i, rec in enumerate(records):
            w = _base_text(str(rec.get('word') or ''))
            if not w:
                continue
            pos = base.find(w, cursor)
            if pos < 0:
                continue
            cursor = pos + len(w)
            if rec.get('route') != 'rule' or not rec.get('ipa_primary') or ' ' in w:
                continue
            if _has_marks(str(rec.get('word') or '')):
                continue
            eng_ipa = str(rec['ipa_primary'])
            eng = self.tokenize(eng_ipa)
            if not eng:
                continue
            cands = graph_candidates(eng)
            own = self.free_reading(pos, pos + len(w), lc, lv)
            if own and own not in cands:
                cands.append(own)
            scored = sorted(((*self.score(w, c, pos, lc, lv, ls), c) for c in cands), key=lambda x: -x[0])
            best_score, k, best = scored[0]
            if best_score == float('-inf'):
                continue
            if k is not None:
                ipa = self.place_stress(best, k)
            else:
                ipa = restress(eng_ipa, best, self.inventory) if len(best) == len(eng) else ''.join(best)
            if ipa == eng_ipa.replace(' ', ''):
                continue
            new = dict(rec)
            new['ipa_primary'] = ipa
            new['layer'] = 'R'
            new['reason'] = f"ReNikud-yi, graph-constrained{(' + stress head' if k is not None else '')}: engine read {eng_ipa}"
            out[i] = new
        return out

def _log_softmax(x):
    import numpy as np
    x = x.astype(np.float32)
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return x - m - np.log(e.sum(-1, keepdims=True))
_INSTANCE: ReNikudYi | None = None

def context_reader(text: str, records: list[dict]) -> list[dict]:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ReNikudYi()
    return _INSTANCE.read_records(text, records)
