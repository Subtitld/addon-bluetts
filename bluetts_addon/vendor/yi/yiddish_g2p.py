from __future__ import annotations
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path
_HEBREW_CHAR = re.compile('[\\u0590-\\u05FF]')

def _strip_points(text: str) -> str:
    decomposed = unicodedata.normalize('NFD', text)
    stripped = ''.join((c for c in decomposed if unicodedata.category(c) != 'Mn'))
    return unicodedata.normalize('NFC', stripped)
PHONE_VOWELS = ('aː', 'ej', 'aj', 'ɔj', 'oʊ', 'a', 'ɛ', 'ə', 'i', 'u', 'ɔ')
PHONE_CONSONANTS = frozenset('bdfɡhjklmnprstvzxʃʒʦʧʤŋ')
PHONE_MARKS = frozenset('ˈ')
PHONE_SYMBOLS = PHONE_VOWELS + tuple(sorted(PHONE_CONSONANTS)) + tuple(PHONE_MARKS)
PHONE_INVENTORY = frozenset(''.join(PHONE_SYMBOLS))
SEPARATORS = ' -'

def ipa_phone_violations(ipa: str) -> list[str]:
    bad: set[str] = set()
    i = 0
    while i < len(ipa):
        if ipa[i] in SEPARATORS:
            i += 1
            continue
        for sym in PHONE_SYMBOLS:
            if ipa.startswith(sym, i):
                i += len(sym)
                break
        else:
            bad.add(ipa[i])
            i += 1
    return sorted(bad)

def vowel_consonant_counts(ipa: str) -> tuple[int, int]:
    vowels = consonants = 0
    i = 0
    while i < len(ipa):
        if ipa[i] in SEPARATORS:
            i += 1
            continue
        for sym in PHONE_SYMBOLS:
            if ipa.startswith(sym, i):
                if sym in PHONE_VOWELS:
                    vowels += 1
                elif sym in PHONE_CONSONANTS:
                    consonants += 1
                i += len(sym)
                break
        else:
            i += 1
    return (vowels, consonants)

def max_consonant_run(ipa: str) -> int:
    longest = run = 0
    i = 0
    while i < len(ipa):
        if ipa[i] in SEPARATORS:
            run = 0
            i += 1
            continue
        for sym in PHONE_SYMBOLS:
            if ipa.startswith(sym, i):
                if sym in PHONE_VOWELS:
                    run = 0
                elif sym in PHONE_CONSONANTS:
                    run += 1
                    longest = max(longest, run)
                i += len(sym)
                break
        else:
            i += 1
    return longest

def violates_vowel_ratio(ipa: str) -> bool:
    for word in re.split('[ -]+', ipa):
        if not word.strip():
            continue
        vowels, consonants = vowel_consonant_counts(word)
        if consonants and vowels == 0:
            return True
        if max_consonant_run(word) > 4:
            return True
    return False
_FINAL_FOLD = {'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}
_FINAL_FOLD_TABLE = str.maketrans(_FINAL_FOLD)
_LIGATURE_FOLD = {'ײ': 'יי', 'ױ': 'וי', 'װ': 'וו'}
_LIGATURE_FOLD_TABLE = str.maketrans(_LIGATURE_FOLD)
_GERESH_CHARS = re.compile('[׳ʼ‘’`]')
_GERSHAYIM_CHARS = re.compile('[״“”]')
_DASH_CHARS = re.compile('[־‐‑‒–—]')
_EDGE_PUNCT = ' \t\n\r.,!?;:()[]{}<>«»„‚‹›…׃״“”"*/\\|-'

def normalize_surface(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = _DASH_CHARS.sub('-', text)
    text = _GERESH_CHARS.sub("'", text)
    return _GERSHAYIM_CHARS.sub('"', text)

def split_affixes(token: str) -> tuple[str, str, str]:
    core = token.strip(_EDGE_PUNCT)
    if not core:
        return ('', token, '')
    start = token.index(core)
    return (token[:start], core, token[start + len(core):])

def lexicon_key(word: str) -> str:
    bare = _strip_points(normalize_surface(word))
    return bare.translate(_FINAL_FOLD_TABLE).translate(_LIGATURE_FOLD_TABLE)
DAGESH = 'ּ'
RAFE = 'ֿ'
SHIN_DOT = 'ׁ'
SIN_DOT = 'ׂ'
SHEVA = 'ְ'
HIRIQ = 'ִ'
TSERE = 'ֵ'
PATAH = 'ַ'
QAMATS = 'ָ'
HOLAM = 'ֹ'
QUBUTS = 'ֻ'
_POINT_TO_LATIN: dict[str, str] = {SHEVA: '', 'ֱ': 'e', 'ֲ': 'a', 'ֳ': 'oo', HIRIQ: 'i', TSERE: 'ey', 'ֶ': 'e', PATAH: 'a', QAMATS: 'oo', HOLAM: 'oy', QUBUTS: 'u', 'ׇ': 'oo'}
_MARKS_CLASS = '[֑-ׇֽֿׁׂׅׄ]*'

def _tolerant(word: str) -> str:
    return ''.join((re.escape(c) + _MARKS_CLASS for c in _strip_points(word)))

def _split_units(word: str) -> list[tuple[str, str]]:
    units: list[list[str]] = []
    for ch in unicodedata.normalize('NFD', word):
        if unicodedata.category(ch) == 'Mn':
            if units:
                units[-1][1] += ch
        else:
            units.append([ch, ''])
    return [(base, marks) for base, marks in units]

def _vowel_point(marks: str) -> str:
    for mark in marks:
        if mark in _POINT_TO_LATIN:
            return mark
    return ''
_LOSHN_KOYDESH = {'שבת': 'שאַבעס', 'שבתים': 'שאַבאָסים', 'מעשה': 'מײַסע', 'גמרא': 'געמאָרע', 'פרשת': 'פּאַרשעס', 'בית המדרש': 'ביסמעדרעש', 'בית מדרש': 'ביסמעדרעש', 'בית דין': 'בעסדין', 'בית עולם': 'בעסוילעם', 'בית הכנסת': 'בעסאַקנעסעס', 'בית': 'בייס', 'עולם': 'אוילעם', 'רבי': 'רעבע', 'פסוק': 'פּאָסיק', 'יצחק': 'ייִצכאָק', 'תשובה': 'טשיווע', 'שלום': 'שאָלעם', 'ברכה': 'בראָכע', 'נשמה': 'נעשאָמע', 'ספרים': 'ספֿאָרים', 'חסידים': 'כאַסידעם', 'כבוד': 'קאָוועד', 'חכמה': 'כאָכמע', 'מנהג': 'מינהעג', 'משפחה': 'מישפּאָכע', 'את': 'עס', 'מצוה': 'מיצווע', 'שמחה': 'סימכע', 'תפלה': 'טפֿילע', 'חיים': 'כײים', 'יום-טובֿ': 'יאנטעוו', 'יום-טובֿים': 'יאָנטויווים', 'פּסח': 'פּייסעך', 'חנוכּה': 'כאַניקע', 'פורים': 'פּורים', 'ראש-השנה': 'ראָשעשאָנע', 'יום-כּיפּור': 'יאָנקיפּער', 'סוכּות': 'סוקעס', 'שבועות': 'שוווּעס', 'תּישעה-באָב': 'טישעבאָוו', 'חודש': 'כוידעש', 'חדשים': 'כאַדאָשים', 'אדר': 'אָדער', 'תּמיד': 'טאָמיד', 'זמן': 'זמאַן', 'שעה': 'שאָ', 'שעות': 'שאָעס', 'משפּחה': 'מישפּאָכע', 'חתן': 'כאָסן', 'כּלה': 'קאַלע', 'מחותּן': 'מעכוטן', 'מחותּנים': 'מעכוטאָנים', 'חבֿר': 'כאַווער', 'חבֿרים': 'כאַוויירים', 'בעל-הבית': 'באַלעבאָס', 'בעלת-הבית': 'באַלעבאָסטע', 'רב': 'ראָוו', 'רבנים': 'ראַבאָנים', 'רבי': 'רעבע', 'צדיק': 'צאַדיק', 'תּלמיד': 'טאַלמיד', 'חכם': 'כאָכעם', 'חכמים': 'כאַכאָמים', 'גנבֿ': 'גאַנעוו', 'גנבֿים': 'גאַנאָווים', 'שדכן': 'שאַדכן', 'שוחט': 'שויכעט', 'קצב': 'קאַצעוו', 'רוצח': 'ראָטסייעך', 'נבֿיא': 'נאָווי', 'עולם': 'אוילעם', 'תּורה': 'טוירע', 'ספֿר': 'סייפער', 'ספֿרים': 'ספֿאָרים', 'מעשׂה': 'מײַסע', 'מעשׂיות': 'מײַסעס', 'פּירוש': 'פּיירעש', 'הגדה': 'האַגאָדע', 'תּפֿילה': 'טפֿילע', 'ברכה': 'בראָכע', 'ברכות': 'בראָכעס', 'מצווה': 'מיצווע', 'עבֿירה': 'אַוויירע', 'חתונה': 'כאַסענע', 'לוויה': 'לעווײַע', 'ברית': 'בריס', 'בית-מדרש': 'בעסמעדרעש', 'כּשר': 'קאָשער', 'טריף': 'טרייף', 'משיח': 'מאָשיעך', 'גלות': 'גאָלעס', 'נשמה': 'נעשאָמע', 'אמת': 'עמעס', 'שׂכל': 'סייכל', 'חכמה': 'כאָכמע', 'פּנים': 'פּאָנעם', 'חן': 'כיין', 'סוד': 'סאָד', 'סודות': 'סוידעס', 'קול': 'קאָל', 'קולות': 'קוילעס', 'חלום': 'כאָלעם', 'מזל': 'מאַזל', 'סכּנה': 'סאַקאָנע', 'טעות': 'טאָעס', 'כּוח': 'קויעך', 'מלחמה': 'מילכאָמע', 'שלום': 'שאָלעם', 'חוצפּה': 'כוטספּע', 'טעם': 'טאַם', 'כּעס': 'קאַס', 'ספֿק': 'סאָפֿעק', 'תּכלית': 'טאַכלעס', 'הצלחה': 'האַצלאָכע', 'פּרנסה': 'פּאַרנאָסע', 'חורבן': 'כורבן', 'צורה': 'צורע', 'טבֿע': 'טעווע', 'רחמנות': 'ראַכמאָנעס', 'אפֿילו': 'אַפֿילע', 'ודאַי': 'וואַדע', 'דווקא': 'דאַווקע', 'כּמעט': 'קימאַט', 'אפֿשר': 'עפֿשער', 'למשל': 'לעמאָשל', 'פּשוט': 'פּושעט', 'ממש': 'מאַמעש', 'תּיכּף': 'טייקעף', 'סך-הכּל': 'סאַכאַקל', 'מזל-טובֿ': 'מאַזלטאָוו', 'שאַקי': 'שייקי', 'שאקי': 'שייקי', 'קידוש': 'קידעש', 'חמש': 'כימעש', 'עדות': 'אײדעס', 'שידוך': 'שידעך', 'תענית': 'טאָנעס', 'מתיבתא': 'מעסיפֿטע', 'מסיבתא': 'מעסיפֿטע', 'בוחר': 'באָכער', 'בחורים': 'באַכירעם', 'בעלי-בתים': 'באַלעבאַטים', 'ישראל': 'ייִסראָעל', 'מלך': 'מיילעך', 'שם': 'שעם', 'מדרש': 'מעדרעש', 'שלמה': 'שלוימע', 'לברכה': 'ליווראָכע', 'חלילה': 'כאַלילע', 'עבודה': 'אַוווידע', 'אמונה': 'עמונע', 'שירה': 'שירע', 'רגע': 'רעגע'}
_WH_WHEN_POINTED = frozenset({'מלך'})
_LK_POINTED: list[tuple[re.Pattern, str]] = [(re.compile('(?<![\\w\u0590-\u05ff])קִדּ?וּ?שׁ?(?![\\w\u0590-\u05ff])'), 'קידעש')]
_LK_BARE: dict[str, str] = {_strip_points(_k): _v for _k, _v in _LOSHN_KOYDESH.items()}
_LK_PATTERN = re.compile('(?<![\\w\\u0590-\\u05FF])(' + '|'.join((_tolerant(k) for k in sorted(_LK_BARE, key=len, reverse=True))) + ')(?![\\w\\u0590-\\u05FF])')

def _lk_replace(match: re.Match) -> str:
    raw = match.group(1)
    bare = _strip_points(raw)
    if bare in _WH_WHEN_POINTED and _vowel_point(raw):
        return raw
    return _LK_BARE.get(bare, raw)
_WORD_LATIN: dict[str, str] = {'וואס': 'voos', 'דאס': 'doos', 'דא': 'doo', 'נאר': 'nor', 'נאך': 'nookh', 'נאכדעם': 'nookhdeem', 'נאכער': 'nookher', 'נאכאמאל': 'nookhamool', 'זאל': 'zool', 'זאלן': 'zooln', 'זאלסט': 'zoolst', 'זאג': 'zoog', 'זאגן': 'zoogn', 'זאגט': 'zoogt', 'זאגסט': 'zoogst', 'געזאגט': 'gezoogt', 'אנזאגן': 'oonzoogn', 'יא': 'yoo', 'יאר': 'yoor', 'יארן': 'yoorn', 'האב': 'hob', 'האבן': 'hobn', 'האסט': 'host', 'וואלט': 'volt', 'וואלטן': 'voltn', 'מאל': 'mool', 'אמאל': 'amool', 'קיינמאל': 'keynmool', 'געווארן': 'gevorn', 'פארוואס': 'farvoos', 'שלאפן': 'shloofn', 'שלאף': 'shloof', 'גארנישט': 'goornisht', 'לאמיר': 'lomir', 'אווענט': 'oovnt', 'אוונט': 'oovnt', 'אוונטן': 'oovntn', 'טאג': 'toog', 'פרייטאג': 'fraytoog', 'דאך': 'dokh', 'שווער': 'shvir', 'שווערער': 'shvirer', 'הערן': 'hirn', 'געהערט': 'gehirt', 'דערהערט': 'derhirt', 'ווערט': 'virt', 'ווערן': 'virn', 'זען': 'zeen', 'זעהן': 'zeen', 'זעה': 'zee', 'זעהט': 'zeet', 'זעט': 'zeet', 'געזען': 'gezeen', 'וועג': 'veeg', 'וועגן': 'veegn', 'וועגס': 'veegs', 'טעג': 'teeg', 'מעג': 'meeg', 'געבן': 'geebn', 'בעטן': 'beetn', 'יעדן': 'yeedn', 'ברענגען': 'breengen', 'קעגן': 'keegn', 'לערנען': 'lirnen', 'לערנט': 'lirnt', 'לערנן': 'lirnen', 'געלערנט': 'gelirnt', 'לעבן': 'leebn', 'לעבט': 'leebt', 'מעגליך': 'meeglekh', 'שפעטער': 'shpeeter', 'געווען': 'geveen', 'דעם': 'deem', 'אים': 'eem', 'איהם': 'im', 'עם': 'eem', 'אבער': 'ober', 'אדער': 'oder', 'דארט': 'dort', 'דארטן': 'dortn', 'וואך': 'vokh', 'וואכן': 'vokhn', 'האט': 'hot', 'געוואלט': 'gevolt', 'אראפ': 'aroop', 'פארט': 'fort', 'אפ': 'op', 'אפט': 'oft', 'מארגן': 'morgn', 'קאפ': 'kop', 'גאט': 'got', 'אקס': 'oks', 'נאז': 'noz', 'גראב': 'grob', 'אוודאי': 'avade', 'גאר': 'goor', 'ווארט': 'vort', 'ווערטער': 'verter', 'נאמען': 'noomen', 'אפאר': 'apoor', 'אביסל': 'abisl', 'ביים': 'bam', 'יוסף': 'yosef', 'יום': 'yoym', 'על': 'al', 'דורך': 'dorekh', 'עפעס': 'epes', 'הויז': 'houz', 'הויזער': 'houzer', 'הויט': 'hout', 'בויך': 'boukh', 'מויל': 'moul', 'מויז': 'mouz', 'טויזנט': 'touznt', 'טויזנטער': 'touznter', 'זויער': 'zouer', 'בויען': 'bouen', 'געבויט': 'gebout', 'דרויסן': 'drousn', 'קרויט': 'krout', 'שטוינט': 'shtount', 'פויער': 'pouer', 'לויט': 'lout', 'מויער': 'mouer', 'לויז': 'louz', 'זויגן': 'zougn', 'פויסט': 'foust', 'זויפן': 'zoufn', 'טרויער': 'trouer', 'הויפן': 'houfn', 'ברויך': 'broukh', 'ברויכן': 'broukhn', 'ברויכט': 'broukht', 'ברויכסט': 'broukhst', 'טרויעריק': 'trouerik', 'טרויעריג': 'trouerig', 'טרויערט': 'trouert', 'געטרויערט': 'getrouert', 'אינדרויסן': 'indrousn', 'בוי': 'bou', 'בויט': 'bout', 'בויער': 'bouer', 'טויזנטסטע': 'touzntste', 'גאסטהויז': 'gasthouz', 'הויזפול': 'houzfil', 'פויל': 'foul', 'פוילער': 'fouler', 'פארפוילט': 'farfoult', 'פוילן': 'poyln', 'פוילין': 'poylin', 'פויליש': 'poylish', 'פוילישע': 'poylishe', 'פוילישער': 'poylisher', 'דערויף': 'derouf', 'זיין': 'zayn', 'ביי': 'bay', 'ביידע': 'bayde', 'ביידן': 'baydn', 'דריי': 'dray', 'היינט': 'haynt', 'ווייל': 'vayl', 'צייט': 'tsayt', 'צייטן': 'tsaytn', 'אריין': 'arayn', 'מיין': 'mayn', 'מיינע': 'mayne', 'דיין': 'dayn', 'דיינע': 'dayne', 'זיינע': 'zayne', 'זייט': 'zayt', 'סיי': 'say', 'מייל': 'mayl', 'ווייטער': 'vayter', 'ניי': 'nay', 'נייע': 'naye', 'ביישפיל': 'bayshpil', 'פיין': 'fayn', 'וויין': 'vayn', 'גלייך': 'glaykh', 'ניין': 'nayn', 'שרייבן': 'shraybn', 'שרייבט': 'shraybt', 'געשריבן': 'geshribn', 'ווייט': 'vayt', 'ווייטע': 'vayte', 'ווייב': 'vayb', 'ווייבל': 'vaybl', 'ווייבער': 'vayber', 'בלייבן': 'blaybn', 'בלייבט': 'blaybt', 'בלייבסט': 'blaybst', 'געבליבן': 'geblibn', 'גלייבן': 'glaybn', 'גלייבט': 'glaybt', 'געגלייבט': 'geglaybt', 'הייזער': 'hayzer', 'איד': 'yid', 'אידן': 'yidn', 'אידיש': 'yidish', 'אידישע': 'yidishe', 'אידישער': 'yidisher', 'אידישן': 'yidishn', 'אידישקייט': 'yidishkayt', 'פרעזידענט': 'prezident', 'פראצענט': 'protsent', 'פראבלעם': 'problem', 'פראבלעמען': 'problemen', 'פלאץ': 'plats', 'פונקט': 'punkt', 'פרעסע': 'prese', 'פאליטיק': 'politik', 'פאליטישע': 'politishe', 'פארטיי': 'partey', 'פרובירן': 'prubirn', 'פרובירט': 'prubirt', 'פלוצלינג': 'plutsling', 'פראגראם': 'program', 'מאדעל': 'model', 'פאליציי': 'politsey', 'פלאן': 'plan', 'פאפיר': 'papir', 'פארק': 'park', 'פאסט': 'post', 'טראמפ': 'tromp', 'אבאמא': 'obama', 'קאפי': 'kofi', 'שאפ': 'shop', 'אודיאו': 'odyo', 'אמעריקע': 'amerike', 'אמעריקאנער': 'amerikaner', 'דעמאקראטן': 'demokratn', 'דעמאקראטיש': 'demokratish', 'רעפובליקאנער': 'republikaner', 'קאנגרעס': 'kongres', 'יארק': 'york', 'דאלאר': 'dolar', 'מיליאן': 'milyon', 'ביליאן': 'bilyon', 'סאו': 'so', 'אקעי': 'okey', 'ניו': 'nyu', 'בערך': 'berekh', 'בכלל': 'bikhlal', 'חלק': 'kheylek', 'מסביר': 'masbir', 'כדי': 'kedey', 'בעצם': 'betsem', 'ממילא': 'memeyle', 'מסתמא': 'mistame', 'אגב': 'agev', 'בקיצור': 'bekitser', 'חוץ': 'khuts', 'שייך': 'shayekh', 'ענין': 'inyen', 'עניינים': 'inyonim', 'פשוטע': 'pushete', 'לבנה': 'levone', 'כח': 'koyekh', 'כוחות': 'koykhes', 'כלל': 'klal', 'כל': 'kool', 'הכל': 'hakl', 'סך': 'sakh', 'מח': 'moyekh', 'בדרך': 'bederekh', 'דרך': 'derekh', 'מזרח': 'mizrekh', 'מערב': 'mayrev', 'משה': 'moyshe', 'במשך': 'bemeshekh', 'משך': 'meshekh', 'מצב': 'matsev', 'חברה': 'khevre', 'שטח': 'shetekh', 'שטחים': 'shtokhim', 'גוף': 'guf', 'רוב': 'rov', 'מהלך': 'mahalekh', 'נפש': 'nefesh', 'סוף': 'sof', 'צד': 'tsad', 'צדדים': 'tsdodim', 'שכל': 'seykhl', 'תוך': 'tokh', 'פרשה': 'parshe', 'סדר': 'seyder', 'הסבר': 'hesber', 'משל': 'moshl', 'לכל': 'lekool', 'אוו': 'ov', 'וויבאלד': 'vibald', 'אויך': 'oykh', 'אויף': 'ouf', 'אויפן': 'afn', 'יעקב': 'yankev', 'חשוון': 'khezhvn', 'ישיבה': 'yeshive', 'סייפער': 'seefer', 'יאנטעוו': 'yontev'}
_CLASS25_PARADIGM: dict[str, str] = {}
_INFLECTIONS: list[tuple[str, str]] = [('ן', 'n'), ('ט', 't'), ('סט', 'st'), ('נס', 'ns'), ('ס', 's'), ('ע', 'e'), ('טע', 'te'), ('נדיק', 'ndik'), ('עדיק', 'edik'), ('נען', 'nen'), ('נט', 'nt'), ('נסט', 'nst')]
for _stem, _latin in _CLASS25_PARADIGM.items():
    for _heb_end, _lat_end in [('', '')] + _INFLECTIONS:
        _WORD_LATIN.setdefault(_stem + _heb_end, _latin + _lat_end)
_STEM_SUBS: list[tuple[str, str]] = [('חסיד', 'כאָסיד'), ('שבת', 'שאַבעס'), ('אמת', 'עמעס'), ('חן', 'כיין'), ('פסק', 'פּאַסק'), ('פטר', 'פּאַטער'), ('הרג', 'האַרג')]
_STEM_MARKS = '[֑-ׇֽׁׅׄ]*'
_STEM_MARKS_NO_DAGESH = '[֑-ׇֻֽׁׅׄ]*'

def _stem_pattern(stem: str) -> re.Pattern:
    if len(stem) < 3:
        return re.compile(re.escape(stem))
    return re.compile(''.join((re.escape(c) + (_STEM_MARKS_NO_DAGESH if c == 'פ' else _STEM_MARKS) for c in stem)))
_STEM_SUB_RE: list[tuple[re.Pattern, str]] = [(_stem_pattern(stem), repl) for stem, repl in _STEM_SUBS]

def _apply_stem_subs(core: str) -> str:
    for pat, repl in _STEM_SUB_RE:
        core = pat.sub(repl, core)
    return core

def _has_stem_sub(core: str) -> bool:
    return any((pat.search(core) for pat, _ in _STEM_SUB_RE))
_SINGLE_MAP: dict[str, str] = {'ב': 'b', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ז': 'z', 'ח': 'kh', 'ט': 't', 'כ': 'kh', 'ך': 'kh', 'ל': 'l', 'מ': 'm', 'ם': 'm', 'נ': 'n', 'ן': 'n', 'ס': 's', 'ע': 'e', 'פ': 'f', 'ף': 'f', 'צ': 'ts', 'ץ': 'ts', 'ק': 'k', 'ר': 'r', 'ש': 'sh', 'ת': 's'}
_LATIN_VOWELS = set('aeiou')
_NUCLEUS_START = set('ויױײ')
_CLUSTERS: list[tuple[str, str]] = [('דזש', 'dzh'), ('דז', 'dz'), ('זש', 'zh'), ('טש', 'tsh')]
_DAGESH_PAIRS: dict[str, tuple[str, str, str]] = {'ב': ('b', 'v', 'b'), 'כ': ('k', 'kh', 'kh'), 'ך': ('k', 'kh', 'kh'), 'פ': ('p', 'f', 'f'), 'ף': ('p', 'f', 'f'), 'ת': ('t', 's', 's')}

def _consonant(ch: str, marks: str) -> str:
    if ch == 'ש':
        return 's' if SIN_DOT in marks else 'sh'
    pair = _DAGESH_PAIRS.get(ch)
    if pair is not None:
        hard, soft, bare = pair
        if RAFE in marks:
            return soft
        if DAGESH in marks:
            return hard
        return bare
    return _SINGLE_MAP.get(ch, '')

def _is_feminine_ending(units: list[tuple[str, str]], i: int) -> bool:
    ch, marks = units[i] if 0 <= i < len(units) else ('', '')
    return ch in ('א', 'ע', 'ו') and _vowel_point(marks) in (QAMATS, PATAH) and (i + 2 == len(units)) and (letter_at(units, i + 1) == 'ה') and (not units[i + 1][1])

def _spells_own_vowel(units: list[tuple[str, str]], i: int) -> bool:
    ch, marks = units[i] if 0 <= i < len(units) else ('', '')
    if not ch:
        return False
    if ch in ('א', 'ע'):
        return bool(_vowel_point(marks)) and (not _is_feminine_ending(units, i))
    if ch in ('ײ', 'ױ'):
        return True
    if ch == 'י':
        return letter_at(units, i + 1) == 'י' or bool(_vowel_point(marks))
    if ch == 'ו':
        if letter_at(units, i + 1) == 'ו':
            return False
        if _vowel_point(marks) and _vowel_point(marks) not in (HOLAM, QUBUTS):
            return False
        return bool(marks) or letter_at(units, i + 1) == 'י'
    return False

def _word_to_latin(word: str) -> str:
    units = _split_units(word)
    out: list[str] = []
    prev_point = ''
    prev_consonant_point = ''
    i = 0
    n = len(units)

    def emitted_any() -> bool:
        return any((t for t in out))

    def last_char() -> str:
        for t in reversed(out):
            if t:
                return t[-1]
        return ''

    def letter(idx: int) -> str:
        return units[idx][0] if 0 <= idx < n else ''

    def marks_of(idx: int) -> str:
        return units[idx][1] if 0 <= idx < n else ''
    while i < n:
        ch, marks = units[i]
        if not _HEBREW_CHAR.match(ch):
            out.append(' ' if ch == '-' else ch)
            prev_point = prev_consonant_point = ''
            i += 1
            continue
        bare_run = ''.join((letter(j) for j in range(i, n)))
        cluster = next((c for c in _CLUSTERS if bare_run.startswith(c[0])), None)
        if cluster is not None:
            out.append(cluster[1])
            prev_point = prev_consonant_point = ''
            i += len(cluster[0])
            continue
        if ch == 'ו' and letter(i + 1) == 'ו':
            out.append('v')
            if DAGESH in marks_of(i + 1):
                out.append('u')
                prev_point, prev_consonant_point = (DAGESH, '')
                i += 2
                continue
            point = _vowel_point(marks) or _vowel_point(marks_of(i + 1))
            i += 2
            prev_point = prev_consonant_point = ''
            if point and (not _spells_own_vowel(units, i)):
                out.append(_POINT_TO_LATIN[point])
                if _POINT_TO_LATIN[point]:
                    prev_point = prev_consonant_point = point
            elif point:
                prev_consonant_point = point
            continue
        if ch == 'װ':
            out.append('v')
            prev_point = prev_consonant_point = ''
            i += 1
            continue
        nucleus = _nucleus(units, i, emitted_any(), last_char(), prev_point, prev_consonant_point)
        if nucleus is not None:
            latin, size, point_used = nucleus
            if latin and last_char() == 'e' and latin.startswith('e'):
                out.append("'")
            out.append(latin)
            prev_point = point_used if latin else ''
            prev_consonant_point = ''
            i += size
            continue
        latin_c = _consonant(ch, marks)
        if latin_c == 'f' and RAFE not in marks and (letter(i - 1) == 'ש') and (SIN_DOT not in marks_of(i - 1)):
            latin_c = 'p'
        if latin_c.startswith('h') and last_char() and (last_char() in 'sztk'):
            out.append("'")
        if latin_c == 'b' and DAGESH not in marks and (i > 0) and (units[i - 1][0] not in 'אעיוײױ') and _HEBREW_CHAR.match(units[i - 1][0]) and _vowel_point(units[i - 1][1]):
            latin_c = 'v'
        if latin_c == 'b' and DAGESH not in marks and _POINT_TO_LATIN.get(_vowel_point(marks)) and (letter(i + 1) not in 'אעיוײױ'):
            latin_c = 'v'
        point = _vowel_point(marks)
        if i == n - 1 and ch in ('ח', 'ע', 'ה') and (point in (PATAH, 'ֲ')) and emitted_any() and (prev_point not in (PATAH, QAMATS, 'ׇ', 'ֲ')) and (last_char() in _LATIN_VOWELS or last_char() == 'y'):
            out.append('e')
            out.append(latin_c)
            prev_point = prev_consonant_point = point
            i += 1
            continue
        out.append(latin_c)
        prev_point = prev_consonant_point = ''
        if point and (not _restates_point(units, i + 1, point)):
            vowel = _POINT_TO_LATIN[point]
            if point in (QAMATS, PATAH) and letter(i + 1) == 'ה' and (not marks_of(i + 1)) and (i + 2 == n):
                vowel = 'e'
            out.append(vowel)
            if vowel:
                prev_point = prev_consonant_point = point
        elif point:
            prev_consonant_point = point
        i += 1
    return ''.join(out)
_MATRES_FOR = {'י': (HIRIQ, TSERE), 'ו': (HOLAM, QUBUTS, DAGESH)}

def _restates_point(units: list[tuple[str, str]], idx: int, point: str) -> bool:
    return _spells_own_vowel(units, idx)

def _nucleus(units: list[tuple[str, str]], i: int, emitted: bool, prev_latin: str, prev_point: str, prev_consonant_point: str='') -> tuple[str, int, str] | None:
    n = len(units)
    ch, marks = units[i]
    nxt, nxt_marks = units[i + 1] if i + 1 < n else ('', '')
    point = _vowel_point(marks)
    if ch == 'ײ':
        if PATAH in marks or prev_consonant_point == PATAH:
            return ('ay', 1, PATAH)
        if i == 0:
            return ('yi', 1, HIRIQ)
        return ('ey', 1, '')
    if ch == 'ױ':
        return ('oy', 1, '')
    if ch == 'י':
        if nxt == 'י':
            if HIRIQ in marks or HIRIQ in nxt_marks:
                return ('yi', 2, HIRIQ)
            if PATAH in marks or PATAH in nxt_marks or prev_consonant_point == PATAH:
                return ('ay', 2, PATAH)
            if i == 0:
                return ('yi', 2, HIRIQ)
            return ('ey', 2, '')
        if point:
            if not emitted and nxt in 'אע' and (not _vowel_point(nxt_marks)):
                return ('y' + _POINT_TO_LATIN[point], 2, point)
            if not emitted:
                return ('y' + _POINT_TO_LATIN[point], 1, point)
            return (_POINT_TO_LATIN[point], 1, point)
        if prev_point in _MATRES_FOR['י']:
            return ('', 1, '')
        if not emitted and nxt in 'אעו':
            return ('y', 1, '')
        if prev_latin in _LATIN_VOWELS:
            return ('y', 1, '')
        if nxt == 'ו' and letter_at(units, i + 2) != 'ו':
            return ('y', 1, '')
        return ('i', 1, '')
    if ch == 'ו':
        yud_follows = nxt == 'י' and (not _vowel_point(nxt_marks))
        if HOLAM in marks:
            return ('oy', 2 if yud_follows else 1, HOLAM)
        if DAGESH in marks or QUBUTS in marks:
            return ('u', 1, DAGESH if DAGESH in marks else QUBUTS)
        if _is_feminine_ending(units, i):
            return ('ve', 1, '')
        if point:
            return ('v' + _POINT_TO_LATIN[point], 1, point)
        if yud_follows:
            return ('oy', 2, '')
        if prev_point in _MATRES_FOR['ו']:
            return ('', 1, '')
        return ('u', 1, '')
    if ch in ('א', 'ע'):
        if _is_feminine_ending(units, i):
            return ('e', 1, '')
        nxt2 = letter_at(units, i + 2)
        if ch == 'א' and (nxt == 'י' and nxt2 == 'י' or nxt in ('ײ', 'ױ') or (nxt == 'ו' and nxt2 == 'י')):
            return ('', 1, '')
        if point:
            return (_POINT_TO_LATIN[point], 1, point)
        if prev_consonant_point:
            return ('', 1, '')
        if ch == 'א' and nxt == 'ו' and _opens_nucleus(units, i + 1):
            return ('', 1, '')
        if not emitted and _opens_nucleus(units, i + 1) and (ch == 'א' or nxt == 'ו'):
            return ('', 1, '')
        return ('a' if ch == 'א' else 'e', 1, '')
    if ch == 'ה' and (not marks) and (prev_latin in _LATIN_VOWELS) and (i == n - 1 or letter_at(units, i + 1) not in 'אעיוײױ'):
        return ('', 1, '')
    if ch == 'ה' and (not marks) and (i == n - 1) and emitted and (prev_latin not in _LATIN_VOWELS):
        return ('e', 1, '')
    return None

def letter_at(units: list[tuple[str, str]], idx: int) -> str:
    return units[idx][0] if 0 <= idx < len(units) else ''

def _opens_nucleus(units: list[tuple[str, str]], idx: int) -> bool:
    ch = letter_at(units, idx)
    if ch not in _NUCLEUS_START:
        return False
    if ch == 'ו' and letter_at(units, idx + 1) == 'ו':
        return False
    return True
_TAG_PATTERN = re.compile('<\\s*[a-zA-Z]+\\s*>')
_PUNCT_SPLIT = re.compile('^([^\\w\\u0590-\\u05FF]*)([\\s\\S]*?)([^\\w\\u0590-\\u05FF]*)$')

def strip_tags(text: str) -> str:
    text = _TAG_PATTERN.sub(' ', text)
    return re.sub('\\s+', ' ', text).strip()

def _preprocess_hebrew(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = re.sub('[\\u05BE\\u2010\\u2011\\u2012\\u2013\\u2014]', '-', text)
    text = re.sub('[\\u05f3\\u02bc\\u2018\\u2019`]', "'", text)
    text = re.sub('[\\u05f4\\u201c\\u201d]', '"', text)
    text = re.sub('(?<![\\u0590-\\u05ff])"(?=[\\u0590-\\u05ff])', '', text)
    text = re.sub('(?<=[\\u0590-\\u05ff])"(?![\\u0590-\\u05ff])', '', text)
    text = re.sub(_tolerant('ס') + "'" + _tolerant('איז'), 'סיז', text)
    text = re.sub(_tolerant('כ') + "'", 'איך ', text)
    text = re.sub(_tolerant('מ') + "'", 'מע ', text)
    text = re.sub(_tolerant('ס') + "'", 'עס ', text)
    text = re.sub('(?<![\\u0590-\\u05FF])' + _tolerant('ר') + "'", 'רעב', text)
    text = re.sub('(?<![\\u0590-\\u05FF])' + _tolerant('ה') + "'", 'האשעם', text)
    text = re.sub("(?<=[\\u0590-\\u05FF])'|'(?=[\\u0590-\\u05FF])", '', text)
    text = re.sub(f'([אעיו]{_MARKS_CLASS})ה{_MARKS_CLASS}(?=ן{_MARKS_CLASS}\\b|ס{_MARKS_CLASS}ט{_MARKS_CLASS}\\b|ט{_MARKS_CLASS}\\b)', '\\1', text)
    for pattern, repl in _LK_POINTED:
        text = pattern.sub(repl, text)
    text = _LK_PATTERN.sub(_lk_replace, text)
    return text
_CLASS54_PREFIXES = (('aroys', 'arous'), ('aroyf', 'arouf'), ('oys', 'ous'), ('areyn', 'arayn'), ('afge', 'opge'), ('araf', 'aroop'))
_KAAT_SUFFIX = re.compile('(?<=.)([kh])eyt(n|s|en|er)?$')
_LEKH_SUFFIX = re.compile('(?<=[^aeiou])likh(e|en|er|es|n|s|st)?$')

def _class54_prefix(latin: str) -> str:
    for src, dst in _CLASS54_PREFIXES:
        if latin.startswith(src):
            latin = dst + latin[len(src):]
            break
    latin = _KAAT_SUFFIX.sub(lambda m: m.group(1) + 'ayt' + (m.group(2) or ''), latin)
    match = _LEKH_SUFFIX.search(latin)
    if match and _nuclei_spans(latin[:match.start()]):
        latin = latin[:match.start()] + 'lekh' + (match.group(1) or '')
    return latin

def hebrew_to_latin(text: str, stem_subs: bool=True) -> str:
    tokens = text.split()
    out_tokens: list[str] = []
    for tok in tokens:
        parts = tok.split('-')
        latin_parts: list[str] = []
        for part in parts:
            if not part:
                continue
            m = _PUNCT_SPLIT.match(part)
            lead, core, trail = (m.group(1), m.group(2), m.group(3))
            bare = _strip_points(core)
            if bare in _WORD_LATIN:
                latin = _WORD_LATIN[bare]
            else:
                if stem_subs:
                    core = _apply_stem_subs(core)
                latin = _word_to_latin(core)
                latin = _class54_prefix(latin)
            latin_parts.append(lead + latin + trail)
        if latin_parts:
            out_tokens.append(' '.join(latin_parts))
    return re.sub('\\s+', ' ', ' '.join(out_tokens)).strip()
STRESS = 'ˈ'
_NUCLEI = ('ee', 'ey', 'ay', 'oy', 'ou', 'oo', 'a', 'e', 'i', 'o', 'u')
_UNSTRESSED_PREFIXES = ('ge', 'be', 'der', 'far', 'tse', 'ba')
_CLITICS = frozenset({'a', 'an', 'di', 'der', 'dos', 'doos', 'dem', 'den', 'de', 'in', 'im', 'un', 'az', 'tsu', 'mit', 'fun', 'far', 'bay', 'ba', 'oyf', 'es', 'zi', 'er', 'ix', 'mir', 'dir', 'zix', 'ze', 'do', 'doo', 'vi', 'nor', 'noor', 'ven'})
_TONIC_SUFFIXES = ('tsyes', 'tsees', 'tsye', 'tsee', 'izm', 'ist', 'ent', 'ant', 'ur', 'ir')
_PRETONIC_SUFFIXES = ('aner', 'iye')
_NEUTRAL_SUFFIXES = ('ndik', 'shaft', 'kayt', 'hayt', 'keyt', 'heyt', 'lekh', 'dik', 'ung', 'es', 'en', 'er', 'l', 's')
_STRESS_OVERRIDES: dict[str, int] = {'mishpookhe': 1, 'tsedooke': 1, 'meshuge': 1, 'mekhutn': 1, 'rebetsn': 0, 'baleboos': 0, 'balebooste': 0, 'yeshive': 1, 'khevre': 0, 'shabes': 0, 'yontev': 0, 'khasene': 0, 'mazltoov': 0, 'seykhl': 0, 'khoydesh': 0, 'khoolem': 0, 'afile': 1, 'efsher': 0, 'asakh': 0, 'neshoome': 1, 'khasidem': 1, 'shaboosim': 1, 'bishas': 1, 'beemes': 1, 'stam': 0, 'mesifte': 1, 'bakhirem': 1, 'balebatim': 2, 'hashem': 1, 'masbir': 1, 'avoyde': 1, 'emune': 1, 'khalile': 1, 'livrookhe': 1, 'azoy': 1, 'viazoy': 2, 'aleyn': 1, 'azoyne': 1, 'azelkhe': 1, 'avade': 1, 'akegn': 1, 'arum': 1, 'tsurik': 1, 'arous': 1, 'arayn': 1, 'avek': 1, 'anider': 1, 'amool': 1, 'aza': 1, 'arop': 1, 'aroop': 1, 'aheym': 1, 'arouf': 1, 'arayf': 1, 'aroyf': 1, 'abisl': 1, 'apoor': 1, 'aleyns': 1, 'ahin': 1, 'aher': 1, 'atsind': 1, 'arunter': 1, 'adurkh': 1, 'arouf': 1, 'beser': 0, 'bese': 0, 'derekh': 0, 'beged': 0, 'berye': 0, 'amerike': 1, 'iran': 1, 'iraner': 1, 'kongres': 1, 'kangres': 1, 'politishe': 1, 'politish': 1, 'politik': 1, 'republikaner': 3, 'demokratn': 2, 'demakrotn': 2, 'milyon': 1, 'protsent': 1}

def _nuclei_spans(word: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(word)
    while i < n:
        for nuc in _NUCLEI:
            if word.startswith(nuc, i):
                spans.append((i, i + len(nuc)))
                i += len(nuc)
                break
        else:
            i += 1
    return spans

def _syllable_count(word: str) -> int:
    return len(_nuclei_spans(word))
_SEPARABLE_PREFIXES: dict[str, int] = {'arous': 1, 'arayn': 1, 'anider': 1, 'arunter': 1, 'aruf': 1, 'arouf': 1, 'arum': 1, 'aroop': 1, 'arop': 1, 'aroyf': 1, 'avek': 1, 'aheym': 1, 'ahin': 1, 'aher': 1, 'adurkh': 1, 'arayf': 1, 'tsurik': 1, 'unter': 0, 'iber': 0, 'mit': 0, 'ous': 0, 'oon': 0, 'on': 0, 'uf': 0, 'oyf': 0, 'ayn': 0, 'tsuzamen': 1, 'farbay': 1, 'antkegn': 1}
_CIRCUMFIX_ENDINGS = {'be': ('n', 't', 'ndik'), 'der': ('n', 't', 'ndik')}
_ILLEGAL_INTERNAL = re.compile('(kh|sh|ts|s|f|kh|b|d|g|k|p|t|v|z)(g|b|d|k|p|t|m|v|z|sh|ts|kh)')
_COMPOUND_SEAMS = ('khg', 'khb', 'khm', 'khsh', 'sts', 'shts', 'fm', 'fb', 'fg', 'sg', 'sb', 'tsg', 'tsb', 'tsm', 'khts', 'shg', 'shb', 'zg', 'zb', 'pg', 'pb', 'kg')

def _compound_split(word: str) -> int | None:
    for seam in _COMPOUND_SEAMS:
        pos = word.find(seam, 1)
        if pos > 0 and _nuclei_spans(word[:pos]) and _nuclei_spans(word[pos:]):
            return pos
    return None

def _splits_nucleus(spans: list[tuple[int, int]], boundary: int) -> bool:
    return any((start < boundary < end for start, end in spans))

def _suffix_stress(stem: str, spans: list[tuple[int, int]]) -> int | None:
    for suf in _PRETONIC_SUFFIXES + _TONIC_SUFFIXES:
        if not stem.endswith(suf) or len(stem) == len(suf):
            continue
        start = len(stem) - len(suf)
        if any((nuc_end == start for _, nuc_end in spans)):
            continue
        for i, (nuc_start, _) in enumerate(spans):
            if nuc_start >= start:
                return i
    return None

def _strip_neutral(word: str) -> str:
    for suf in _NEUTRAL_SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf):
            stem = word[:-len(suf)]
            if _nuclei_spans(stem):
                return stem
    return word

def _stressed_syllable(word: str, count: int) -> int:
    spans = _nuclei_spans(word)
    for pre, pre_idx in _SEPARABLE_PREFIXES.items():
        if word.startswith(pre) and len(word) > len(pre) and (not _splits_nucleus(spans, len(pre))):
            tail = word[len(pre):]
            pre_spans = _nuclei_spans(pre)
            if _nuclei_spans(tail) and pre_spans:
                return min(pre_idx, len(pre_spans) - 1)
    seam = _compound_split(word)
    if seam is not None:
        head = word[:seam]
        if _nuclei_spans(head):
            return 0
    idx = _suffix_stress(word, spans)
    if idx is not None:
        return min(idx, count - 1)
    stem = _strip_neutral(word)
    if stem != word:
        idx = _suffix_stress(stem, spans)
        if idx is not None:
            return min(idx, count - 1)
    consumed = 0
    rest = word
    while True:
        for pre in _UNSTRESSED_PREFIXES:
            if rest.startswith(pre):
                if _splits_nucleus(_nuclei_spans(rest), len(pre)):
                    continue
                tail = rest[len(pre):]
                required = _CIRCUMFIX_ENDINGS.get(pre)
                if required and (not word.endswith(required)):
                    continue
                if _nuclei_spans(tail):
                    consumed += len(_nuclei_spans(pre))
                    rest = tail
                    break
        else:
            break
    return min(consumed, count - 1)

def add_stress(latin: str, penult: bool=False) -> str:
    out: list[str] = []
    for token in re.split('(\\s+)', latin):
        if not token or token.isspace():
            out.append(token)
            continue
        core = token.strip('.,!?;:"()\'-')
        lead = token[:len(token) - len(token.lstrip('.,!?;:"()\'-'))]
        trail = token[len(lead) + len(core):]
        spans = _nuclei_spans(core)
        lowered = core.lower()
        if not core or not spans or _syllable_count(lowered) == 1 or (lowered in _CLITICS):
            out.append(token)
            continue
        idx = _STRESS_OVERRIDES.get(lowered)
        if idx is None and penult:
            stem_spans = _nuclei_spans(_strip_neutral(lowered)) or spans
            idx = max(len(stem_spans) - 2, 0)
        if idx is None:
            idx = _stressed_syllable(lowered, len(spans))
        idx = min(max(idx, 0), len(spans) - 1)
        at = spans[idx][0]
        out.append(lead + core[:at] + STRESS + core[at:] + trail)
    return ''.join(out)
_LATIN_TO_IPA: list[tuple[str, str]] = [('tsh', 'ʧ'), ('dzh', 'ʤ'), ('ts', 'ʦ'), ('kh', 'x'), ('sh', 'ʃ'), ('zh', 'ʒ'), ('ee', 'ej'), ('ey', 'aj'), ('ay', 'aː'), ('oy', 'ɔj'), ('ou', 'oʊ'), ('oo', 'u'), ('a', 'a'), ('o', 'ɔ'), ('e', 'ɛ'), ('u', 'i'), ('i', 'i'), ('b', 'b'), ('v', 'v'), ('d', 'd'), ('h', 'h'), ('z', 'z'), ('t', 't'), ('l', 'l'), ('m', 'm'), ('n', 'n'), ('s', 's'), ('f', 'f'), ('p', 'p'), ('k', 'k'), ('g', 'ɡ'), ('r', 'r'), ('y', 'j'), ('-', ' ')]
_APOSTROPHE = re.compile("'")
_AFFRICATE_DECOMPOSE = [(re.compile('t͡s', re.I), 'ʦ'), (re.compile('t͡ʃ', re.I), 'ʧ'), (re.compile('d͡ʒ', re.I), 'ʤ'), (re.compile('d͡z', re.I), 'dz')]

def latin_to_ipa(latin: str) -> str:
    out: list[str] = []
    i = 0
    n = len(latin)
    while i < n:
        ch = latin[i]
        if ch in ' \t\n\r':
            out.append(' ')
            i += 1
            continue
        if ch == "'":
            i += 1
            continue
        if ch in '.,!?;:"()':
            out.append(ch)
            i += 1
            continue
        matched = False
        for src, dst in _LATIN_TO_IPA:
            if latin.startswith(src, i):
                out.append(dst)
                i += len(src)
                matched = True
                break
        if not matched:
            out.append(ch)
            i += 1
    ipa = ''.join(out)
    ipa = re.sub('(\\S{2,})ɛ\\b', '\\1ə', ipa)
    return ipa
_DEVOICE = {'b': 'p', 'd': 't', 'ɡ': 'k', 'v': 'f', 'z': 's', 'ʒ': 'ʃ', 'ʤ': 'ʧ'}
_VOICE = {v: k for k, v in _DEVOICE.items()}
_VOICING_TRIGGERS = frozenset('bdɡzʒʤ')
_DEVOICING_TRIGGERS = frozenset('ptkfsʃxʦʧ')
_OBSTRUENTS = _VOICING_TRIGGERS | _DEVOICING_TRIGGERS | frozenset('v')
_VOICING_TARGETS = frozenset('sʃʦʧ')
_WORD_SPLIT = re.compile('(\\s+)')

def _postlexical_word(word: str) -> str:
    core = word.rstrip('.,!?;:"()\'')
    trail = word[len(core):]
    chars = list(core)
    for i in range(len(chars) - 2, -1, -1):
        cur, nxt = (chars[i], chars[i + 1])
        if cur not in _OBSTRUENTS or nxt not in _OBSTRUENTS:
            continue
        if nxt in _VOICING_TRIGGERS and cur in _VOICING_TARGETS:
            chars[i] = _VOICE.get(cur, cur)
    chars = [c for i, c in enumerate(chars) if i == 0 or c != chars[i - 1] or c not in _OBSTRUENTS]
    return ''.join(chars) + trail

def postlexical(ipa: str) -> str:
    return ''.join((tok if not tok or tok.isspace() else _postlexical_word(tok) for tok in _WORD_SPLIT.split(ipa)))

def reduce_unstressed(ipa: str) -> str:
    words = []
    for word in ipa.split(' '):
        if STRESS not in word:
            words.append(word)
            continue
        keep = word.index(STRESS) + 1
        words.append(''.join(('ə' if ch == 'ɛ' and i != keep else ch for i, ch in enumerate(word))))
    return ' '.join(words)

def normalize_ipa_affricates(ipa: str) -> str:
    for pattern, lig in _AFFRICATE_DECOMPOSE:
        ipa = pattern.sub(lig, ipa)
    return ipa

def normalize_ipa_spacing(ipa: str) -> str:
    ipa = re.sub('\\s+', ' ', ipa).strip()
    for punct in [',', '.', '!', '?', ';', ':', "'"]:
        ipa = ipa.replace(f' {punct}', punct)
    return ipa

def g2p_fingerprint() -> str:
    import json as _json
    parts = [_json.dumps(_STRESS_OVERRIDES, sort_keys=True, ensure_ascii=False), _json.dumps(_LOSHN_KOYDESH, sort_keys=True, ensure_ascii=False), _json.dumps(_WORD_LATIN, sort_keys=True, ensure_ascii=False), repr(_UNSTRESSED_PREFIXES), repr(_SEPARABLE_PREFIXES), repr(_TONIC_SUFFIXES), repr(_PRETONIC_SUFFIXES), repr(_NEUTRAL_SUFFIXES), repr(_CIRCUMFIX_ENDINGS), repr(_LATIN_TO_IPA), repr(_COMPOUND_SEAMS), 'schwa_reduction=1', 'postlexical=1', _json.dumps({k: v['ipa_primary'] for k, v in GOLD_LEXICON.items()}, sort_keys=True, ensure_ascii=False), _json.dumps(_ABBREVIATIONS, sort_keys=True, ensure_ascii=False), _json.dumps(_MULTIWORD, sort_keys=True, ensure_ascii=False), repr(sorted(_MULTIWORD_LEGACY)), repr(sorted(_CLITIC_IPA.items()))]
    return hashlib.sha1('|'.join(parts).encode('utf-8')).hexdigest()[:12]

def _rule_path_ipa(text: str, stress: bool=True, lk_penult: bool=False, stem_subs: bool=True) -> str:
    text = _preprocess_hebrew(strip_tags(text))
    latin = hebrew_to_latin(text, stem_subs=stem_subs)
    if stress:
        latin = add_stress(latin, penult=lk_penult)
    ipa = latin_to_ipa(latin)
    ipa = postlexical(ipa)
    if stress:
        ipa = reduce_unstressed(ipa)
    ipa = normalize_ipa_affricates(ipa)
    return normalize_ipa_spacing(ipa)

def _load_gold_lexicon() -> dict:
    path = Path(__file__).resolve().parent / 'data' / 'lexicons' / 'gold_lexicon.py'
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_yiddish_gold_lexicon', path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return dict(getattr(module, 'GOLD_LEXICON', {}) or {})
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(f'{path} exists but could not be loaded ({exc!r}). Returning an empty table here would silently drop every verdict it holds; regenerate it with its builder in scripts/.') from exc
GOLD_LEXICON: dict[str, dict] = _load_gold_lexicon()

def _load_audio_endorsed() -> dict:
    path = Path(__file__).resolve().parent / 'data' / 'lexicons' / 'audio_endorsed_lk.py'
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_yiddish_audio_lk', path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, 'AUDIO_ENDORSED_LK', {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(f'{path} exists but could not be loaded ({exc!r}). Returning an empty table here would silently drop every verdict it holds; regenerate it with its builder in scripts/.') from exc
_AUDIO_ENDORSED: dict[str, dict] = _load_audio_endorsed()

def _load_homograph_lk() -> dict:
    path = Path(__file__).resolve().parent / 'data' / 'lexicons' / 'homograph_lk.py'
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_yiddish_homograph_lk', path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, 'HOMOGRAPH_LK', {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(f'{path} exists but could not be loaded ({exc!r}). Returning an empty table here would silently drop every verdict it holds; regenerate it with its builder in scripts/.') from exc
_HOMOGRAPH_LK: dict[str, dict] = _load_homograph_lk()

def _load_audio_pe() -> dict:
    path = Path(__file__).resolve().parent / 'data' / 'lexicons' / 'audio_pe_lk.py'
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_yiddish_audio_pe_lk', path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, 'AUDIO_PE_LK', {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(f'{path} exists but could not be loaded ({exc!r}). Returning an empty table here would silently drop every verdict it holds; regenerate it with its builder in scripts/.') from exc
_AUDIO_PE: dict[str, dict] = _load_audio_pe()

def _load_audio_vowel() -> dict:
    path = Path(__file__).resolve().parent / 'data' / 'lexicons' / 'audio_vowel_lk.py'
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_yiddish_audio_vowel_lk', path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, 'AUDIO_VOWEL_LK', {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(f'{path} exists but could not be loaded ({exc!r}). Returning an empty table here would silently drop every verdict it holds; regenerate it with its builder in scripts/.') from exc
_AUDIO_VOWEL: dict[str, dict] = _load_audio_vowel()

def _load_sefaria_pointed() -> dict:
    path = Path(__file__).resolve().parent / 'data' / 'lexicons' / 'sefaria_pointed_lk.py'
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_yiddish_sefaria_lk', path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, 'SEFARIA_POINTED_LK', {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(f'{path} exists but could not be loaded ({exc!r}). Returning an empty table here would silently drop every verdict it holds; regenerate it with its builder in scripts/.') from exc
_SEFARIA_POINTED: dict[str, dict] = _load_sefaria_pointed()

def _load_printed_respelling() -> dict:
    path = Path(__file__).resolve().parent / 'data' / 'lexicons' / 'printed_respelling_lk.py'
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_yiddish_respelling_lk', path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, 'PRINTED_RESPELLING_LK', {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(f'{path} exists but could not be loaded ({exc!r}). Returning an empty table here would silently drop every verdict it holds; regenerate it with its builder in scripts/.') from exc
_PRINTED_RESPELLING: dict[str, dict] = _load_printed_respelling()

def _load_model_pointed() -> dict:
    path = Path(__file__).resolve().parent / 'data' / 'lexicons' / 'model_pointed_lk.py'
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_yiddish_model_lk', path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = dict(getattr(module, 'MODEL_POINTED_LK', {}) or {})
        return {lexicon_key(w): v for w, v in raw.items()}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(f'{path} exists but could not be loaded ({exc!r}). Returning an empty table here would silently drop every verdict it holds; regenerate it with its builder in scripts/.') from exc
_MODEL_POINTED: dict[str, dict] = _load_model_pointed()
_ABBREVIATIONS: dict[str, tuple[str, list[str]]] = {"ר'": ('rɛb', []), "ה'": ('haʃˈɛm', []), 'שליט"א': ('ʃlˈitə', []), 'זצ"ל': ('zaʦˈal', []), 'ז"ל': ('zal', []), 'זי"ע': ('zxisˈɔj jˈuɡin ulˈajni', []), 'יו"ט': ('jˈɔntəf', ['jˈɔntif']), 'ב"ה': ('bˈurəx haʃˈɛm', [])}
_ABBREVIATIONS = {lexicon_key(k): v for k, v in _ABBREVIATIONS.items()}
_ACRONYM_WORDS: dict[str, tuple[str, list[str]]] = {'רש"י': ('rˈaʃi', []), 'חז"ל': ('xazˈal', []), 'רמב"ם': ('rˈambam', []), 'רמב"ן': ('rˈamban', []), 'חיד"א': ('xˈidə', []), 'של"ה': ('ʃlu', []), 'אר"י': ('ˈari', []), 'תרי"ג': ('tarjˈaɡ', []), 'ל"ג': ('laɡ', []), 'מהר"ם': ('mahˈaram', []), 'מהר"ל': ('mahˈaral', []), 'מהרש"א': ('mahˈarʃə', []), 'בעש"ט': ('bɛʃt', []), 'ש"ס': ('ʃas', []), 'הש"ס': ('haʃˈas', []), 'תנ"ך': ('tˈanax', []), 'חב"ד': ('xabˈad', [])}
_ACRONYM_WORDS = {lexicon_key(k): v for k, v in _ACRONYM_WORDS.items()}
_LETTER_NAMES: dict[str, str] = {'א': 'ˈaləf', 'ב': 'bajs', 'ג': 'ɡiml', 'ד': 'dˈuləd', 'ה': 'haj', 'ו': 'vuv', 'ז': 'zˈajən', 'ח': 'xɛs', 'ט': 'tɛs', 'י': 'jid', 'כ': 'xuf', 'ך': 'xuf', 'ל': 'lˈaməd', 'מ': 'mɛm', 'ם': 'mɛm', 'נ': 'nin', 'ן': 'nin', 'ס': 'sˈaməx', 'ע': 'ˈajən', 'פ': 'paj', 'ף': 'paj', 'צ': 'ʦˈadik', 'ץ': 'ʦˈadik', 'ק': 'kif', 'ר': 'rajʃ', 'ש': 'ʃin', 'ת': 'tuf'}

def letter_name_ipa(key: str) -> str:
    letters = [c for c in key if c not in '"\'']
    if not letters or any((c not in _LETTER_NAMES for c in letters)):
        return ''
    return ' '.join((_LETTER_NAMES[c] for c in letters))
_LETTER_NAME_WORDS: dict[str, str] = {'אלף': 'ˈaləf', 'בית': 'bajs', 'גימל': 'ɡiml', 'גמל': 'ɡiml', 'דלת': 'dˈuləd', 'הא': 'haj', 'וו': 'vuv', 'ווו': 'vuv', 'זין': 'zˈajən', 'חית': 'xɛs', 'טית': 'tɛs', 'יוד': 'jid', 'כף': 'xuf', 'למד': 'lˈaməd', 'מם': 'mɛm', 'נון': 'nin', 'סמך': 'sˈaməx', 'עין': 'ˈajən', 'צדי': 'ʦˈadik', 'צדיק': 'ʦˈadik', 'קוף': 'kif', 'ריש': 'rajʃ', 'שין': 'ʃin', 'תיו': 'tuf'}
_LETTER_NAME_WORDS = {lexicon_key(k): v for k, v in _LETTER_NAME_WORDS.items()}

def letter_name_word_ipa(key: str) -> str:
    return _LETTER_NAME_WORDS.get(key.replace('"', '').replace("'", ''), '')
_MULTIWORD: dict[str, tuple[str, list[str]]] = {'בית מדרש': ('bis-mˈɛdrəʃ', ['bˈis-mədrəʃ']), 'בית-מדרש': ('bis-mˈɛdrəʃ', ['bˈis-mədrəʃ']), 'א פאר': ('a pˈur', ['a pˈɔr'])}
_MULTIWORD_MINED: dict[str, tuple[str, list[str]]] = {'זכרונו לברכה': ('zixrˈɔjni livrˈuxə', []), 'יום טוב': ('jˈɔntəv', ['jɔjm tɔjv']), 'ארץ ישראל': ('ˈɛrəʦ jisrˈuəl', []), 'ראש השנה': ('rˈuʃəʃunə', ['rɔjʃ haʃˈunə']), 'כלל ישראל': ('klal jisrˈuəl', []), 'יום כיפור': ('jˈunkipər', []), 'שבת קודש': ('ʃˈabəs kˈɔjdəʃ', []), 'ראש חודש': ('rɔjʃ xˈɔjdəʃ', []), 'רבונו של עולם': ('ribˈɔjnə ʃɛl ˈɔjləm', []), 'ריבונו של עולם': ('ribˈɔjnə ʃɛl ˈɔjləm', []), 'ברוך השם': ('bˈurəx haʃˈɛm', []), 'תשעה באב': ('tˈiʃə buv', []), 'לשון הרע': ('lˈuʃn hˈurə', []), 'בעל הבית': ('bˈaləbus', ['baləbˈus']), 'בית דין': ('bˈɛs din', ['bajs din']), 'ערב שבת': ('ˈɛrɛv ʃˈabəs', []), 'ראש ישיבה': ('rɔjʃ jəʃˈivə', []), 'עולם הבא': ('ˈɔjləm habˈu', []), 'אהבת ישראל': ('ˈahavas jisrˈuəl', []), 'ימים טובים': ('jˈumim tˈɔjvim', []), 'קידוש השם': ('kˈidəʃ haʃˈɛm', []), 'חול המועד': ('xɔl hamˈɔjəd', []), 'שבע ברכות': ('ʃˈɛvə brˈuxəs', []), 'עולם הזה': ('ˈɔjləm hˈazɛ', []), 'מחצית השקל': ('maxˈaʦis haʃˈɛkɛl', []), 'מלך מלכי המלכים': ('mˈɛlɛx malxˈaj hamlˈuxim', []), 'בעזרת השם': ('bəˈɛzras haʃˈɛm', []), 'בר מצוה': ('bar mˈiʦvə', []), 'אשת חיל': ('ˈajʃɛs xˈajil', []), 'בעלי בתים': ('baləbˈatim', []), 'מסירת נפש': ('məsˈiras nˈejfiʃ', []), 'אם ירצה השם': ('im jˈirʦə haʃˈɛm', []), 'מלוה מלכה': ('məlˈavə mˈalkə', []), 'פסח שני': ('pˈajsəx ʃˈajni', []), 'שלום בית': ('ʃˈuləm bˈajis', []), 'גמילות חסדים': ('ɡəmˈilus xasˈudim', []), 'זאת אומרת': ('zɔjs ɔjmˈɛrɛs', []), 'יום טובים': ('jˈuntɔjvim', [])}
_MULTIWORD.update(_MULTIWORD_MINED)
_MULTIWORD = {lexicon_key(k): v for k, v in _MULTIWORD.items()}
_MULTIWORD_MINED_KEYS: set[str] = {lexicon_key(k) for k in _MULTIWORD_MINED}
for _mw_key in list(_MULTIWORD):
    _mw_hyphen = _mw_key.replace(' ', '-')
    if _mw_hyphen != _mw_key and _mw_hyphen not in _MULTIWORD:
        _MULTIWORD[_mw_hyphen] = _MULTIWORD[_mw_key]
        if _mw_key in _MULTIWORD_MINED_KEYS:
            _MULTIWORD_MINED_KEYS.add(_mw_hyphen)

def _multiword_confidence(key: str) -> tuple[str, str]:
    return ('LOW', 'mwe-mined') if key in _MULTIWORD_MINED_KEYS else ('HIGH', '')
_MULTIWORD_LEGACY = {lexicon_key(k) for k in _LOSHN_KOYDESH if ' ' in k}
_MAX_MULTIWORD = max([len(k.split()) for k in list(_MULTIWORD) + list(_MULTIWORD_LEGACY)] or [1])
_CLITIC_IPA = {'ס': 's', 'מ': 'm', 'כ': 'x'}
_LK_DETECT = re.compile('[חת]|שׂ|כּ')
_VOWEL_LETTERS = frozenset('אעיוײױ')
A_DEFAULT_LOG: Counter = Counter()
P_DEFAULT_LOG: Counter = Counter()

def reset_default_logs() -> None:
    A_DEFAULT_LOG.clear()
    P_DEFAULT_LOG.clear()

def _lk_detector(bare: str) -> bool:
    if _LK_DETECT.search(bare):
        return True
    letters = _strip_points(bare)
    if letters.endswith('ות'):
        return True
    consonants = sum((1 for c in letters if _HEBREW_CHAR.match(c) and c not in _VOWEL_LETTERS))
    vowels = sum((1 for c in letters if c in _VOWEL_LETTERS))
    return vowels == 0 and consonants >= 2

def _lk_nikud(core: str) -> bool:
    return any((_vowel_point(marks) for _, marks in _split_units(core)))

def _lk_evidence(core: str) -> bool:
    if _LK_PATTERN.search(core) or _has_stem_sub(core):
        return True
    return _lk_nikud(core)

def _has_ambiguous_alef(core: str) -> bool:
    for ch, marks in _split_units(core):
        if ch == 'א' and (not _vowel_point(marks)):
            return True
    return False

def _has_ambiguous_pe(core: str) -> bool:
    for ch, marks in _split_units(core):
        if ch in ('פ', 'ף') and DAGESH not in marks and (RAFE not in marks):
            return True
    return False

def _pe_point_contradicts(core: str, ipa: str) -> bool:
    for ch, marks in _split_units(core):
        if ch in ('פ', 'ף'):
            if DAGESH in marks and 'p' not in ipa:
                return True
            if RAFE in marks and 'f' not in ipa:
                return True
    return False

def _known_word(core: str) -> bool:
    key = lexicon_key(core)
    bare = _strip_points(normalize_surface(core))
    return key in GOLD_LEXICON or bare in _LK_BARE or bare in _WORD_LATIN

def _entry_result(word: str, primary: str, variants: list[str], layer: str, route: str, confidence: str, reason: str='') -> dict:
    return {'word': word, 'ipa_primary': primary, 'variants': [v for v in variants if v != primary], 'layer': layer, 'route': route, 'confidence': confidence, 'reason': reason}
FINAL_DEVOICE_MAP = {'b': 'p', 'd': 't', 'ɡ': 'k', 'v': 'f', 'z': 's', 'ʒ': 'ʃ', 'ʤ': 'ʧ'}

def devoiced_final(ipa: str) -> str:
    parts = re.split('([ -])', ipa)
    out: list[str] = []
    changed = False
    for part in parts:
        if part and part[-1] in FINAL_DEVOICE_MAP:
            out.append(part[:-1] + FINAL_DEVOICE_MAP[part[-1]])
            changed = True
        else:
            out.append(part)
    return ''.join(out) if changed else ''

def _with_auto_variants(rec: dict) -> dict:
    rec['variants'] = list(rec['variants'])
    rec['auto_variants'] = []
    primary = rec['ipa_primary']
    if not primary or rec['route'] == 'fallback':
        return rec
    auto = devoiced_final(primary)
    if auto and auto != primary and (auto not in rec['variants']):
        rec['variants'].append(auto)
        rec['auto_variants'] = [auto]
    return rec

def _multiword_ipa(key: str) -> str:
    if key in _MULTIWORD:
        return _MULTIWORD[key][0]
    return _rule_path_ipa(key, stress=True)

def _guess_layer(core: str) -> str:
    return 'L' if _lk_detector(_strip_points(core)) else 'G'

def _route_token(core: str) -> dict:
    result = _route_token_inner(core)
    primary = result['ipa_primary']
    if primary:
        repaired = _repair_lk_proclitic(core, primary)
        if repaired is not None:
            result, primary = (repaired, repaired['ipa_primary'])
    if result['route'] == 'fallback' or not primary:
        return _audio_endorsed_or(core, result)
    flags = []
    if violates_vowel_ratio(primary):
        flags.append('vowel-ratio')
    if ipa_phone_violations(primary):
        flags.append('bad-phone')
    if not flags:
        return result
    reason = ','.join(filter(None, [result['reason'], *flags]))
    return _audio_endorsed_or(core, _entry_result(core, primary, result['variants'], result['layer'], 'fallback', 'LOW', reason))

def _audio_endorsed_or(core: str, fallback_result: dict) -> dict:
    entry = _AUDIO_ENDORSED.get(lexicon_key(core))
    if entry is None:
        return _homograph_or(core, fallback_result)
    return _entry_result(core, entry['ipa'], [], 'L', 'rule', 'LOW', 'pointed-audio-endorsed')

def _homograph_or(core: str, fallback_result: dict) -> dict:
    entry = _HOMOGRAPH_LK.get(lexicon_key(core))
    if entry is None:
        return _sefaria_pointed_or(core, fallback_result)
    return _entry_result(core, entry['ipa'], list(entry.get('variants') or []), 'L', 'rule', 'LOW', entry['reason'])

def _sefaria_pointed_or(core: str, fallback_result: dict) -> dict:
    entry = _SEFARIA_POINTED.get(lexicon_key(core))
    if entry is not None:
        return _entry_result(core, entry['ipa'], list(entry.get('variants') or []), 'L', 'rule', 'LOW', 'sefaria-pointed')
    entry = _PRINTED_RESPELLING.get(lexicon_key(core))
    if entry is not None:
        return _entry_result(core, entry['ipa'], list(entry.get('variants') or []), 'L', 'rule', 'LOW', 'printed-respelling')
    entry = _MODEL_POINTED.get(lexicon_key(core))
    if entry is not None:
        return _entry_result(core, entry['ipa'], list(entry.get('variants') or []), 'L', 'rule', 'LOW', 'model-pointed-guess')
    return fallback_result
_GERMANIC_SUFFIX_IPA: list[tuple[str, str]] = [('ניקעס', 'nikəs'), ('דיגע', 'diɡə'), ('דיגן', 'diɡn'), ('דיג', 'diɡ'), ('ניק', 'nik'), ('לעך', 'ləx'), ('טע', 'tə'), ('שע', 'ʃə'), ('ס', 's'), ('ן', 'n')]
_MIN_STEM_ROOT = 3
_STEM_NO_SPLIT: frozenset[str] = frozenset({'בנינ', 'כדינ', 'שמחס'})

def _lk_table_reading(core: str) -> tuple[str, str] | None:
    key = lexicon_key(core)
    bare = _strip_points(normalize_surface(core))
    entry = GOLD_LEXICON.get(key)
    if entry is not None and entry['layer'] == 'L':
        return (entry['ipa_primary'], 'gold')
    if bare in _LK_BARE:
        return (_rule_path_ipa(core, stress=True), 'lk-lexicon')
    entry = _AUDIO_ENDORSED.get(key)
    if entry is not None:
        return (entry['ipa'], 'pointed-audio-endorsed')
    entry = _HOMOGRAPH_LK.get(key)
    if entry is not None:
        return (entry['ipa'], entry['reason'])
    entry = _SEFARIA_POINTED.get(key)
    if entry is not None:
        return (entry['ipa'], 'sefaria-pointed')
    entry = _PRINTED_RESPELLING.get(key)
    if entry is not None:
        return (entry['ipa'], 'printed-respelling')
    entry = _MODEL_POINTED.get(key)
    if entry is not None:
        return (entry['ipa'], 'model-pointed-guess')
    return None

def _full_form_resolves(core: str) -> bool:
    key = lexicon_key(core)
    bare = _strip_points(normalize_surface(core))
    if key in GOLD_LEXICON or key in _MULTIWORD or key in _MULTIWORD_LEGACY:
        return True
    if bare in _LK_BARE or bare in _WORD_LATIN:
        return True
    if key in _AUDIO_ENDORSED or key in _HOMOGRAPH_LK or key in _SEFARIA_POINTED or (key in _PRINTED_RESPELLING) or (key in _MODEL_POINTED):
        return True
    return _has_stem_sub(core)

def _model_guess_root_is_lk(root: str) -> bool:
    bare = _strip_points(root)
    if not _lk_nikud(root):
        return _lk_detector(bare)
    return bool(_LK_DETECT.search(normalize_surface(root)) or bare.endswith('ות'))

def _phone_list(ipa: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(ipa):
        if ipa[i] in SEPARATORS or ipa[i] in PHONE_MARKS:
            i += 1
            continue
        for sym in PHONE_SYMBOLS:
            if ipa.startswith(sym, i) and sym not in PHONE_MARKS:
                out.append(sym)
                i += len(sym)
                break
        else:
            out.append(ipa[i])
            i += 1
    return out

def _join_stem(root_ipa: str, suffix_ipa: str) -> str | None:
    root_phones = _phone_list(root_ipa)
    suffix_phones = _phone_list(suffix_ipa)
    if not root_phones or not suffix_phones:
        return None
    link = ''
    if suffix_ipa == 'n' and root_phones[-1] in PHONE_VOWELS and (root_phones[-1] != 'ə'):
        link = 'ə'
    joined = root_ipa + link + suffix_ipa
    phones = _phone_list(joined)
    if any((a == b and a in PHONE_CONSONANTS for a, b in zip(phones, phones[1:]))):
        return None
    return joined

def _stem_suffix_rescue(core: str) -> dict | None:
    if _full_form_resolves(core) or lexicon_key(core) in _STEM_NO_SPLIT:
        return None
    units = _split_units(normalize_surface(core))
    letters = ''.join((base for base, _ in units))
    for suffix, suffix_ipa in _GERMANIC_SUFFIX_IPA:
        if not letters.endswith(suffix):
            continue
        root_units = units[:-len(suffix)]
        root = unicodedata.normalize('NFC', ''.join((base + marks for base, marks in root_units)))
        root_bare = _strip_points(root)
        if sum((1 for c in root_bare if _HEBREW_CHAR.match(c))) < _MIN_STEM_ROOT:
            continue
        reading = _lk_table_reading(root)
        if reading is None:
            continue
        root_ipa, reason = reading
        if not root_ipa:
            continue
        if reason == 'model-pointed-guess' and (not _model_guess_root_is_lk(root)):
            continue
        if reason == 'printed-respelling':
            _entry = _PRINTED_RESPELLING.get(lexicon_key(root))
            if _entry is None or _entry.get('join') != 'exact':
                continue
        joined = _join_stem(root_ipa, suffix_ipa)
        if joined is None:
            continue
        if vowel_consonant_counts(joined)[0] < vowel_consonant_counts(_rule_path_ipa(core, stress=False))[0]:
            continue
        return _entry_result(core, joined, [], 'L', 'rule', 'LOW', reason + '+suffix')
    return None
_PREFIX_RESCUE_TABLE: list[tuple[str, str, str]] = [('אויסגע', 'separable', 'ˈoʊzɡə'), ('אויפגע', 'separable', 'ˈoʊfɡə'), ('איינגע', 'separable', 'ˈaːnɡə'), ('אפגע', 'separable', 'ˈupɡə'), ('אנגע', 'separable', 'ˈunɡə'), ('אונטערגע', 'separable', 'ˈintərɡə'), ('איבערגע', 'separable', 'ˈibərɡə'), ('דורכגע', 'separable', 'dˈirxɡə'), ('מיטגע', 'separable', 'mˈitɡə'), ('צוגע', 'separable', 'ʦˈiɡə'), ('ארויסגע', 'separable', 'arˈoʊzɡə'), ('ארויפגע', 'separable', 'arˈoʊfɡə'), ('אראפגע', 'separable', 'arˈupɡə'), ('אריינגע', 'separable', 'arˈaːnɡə'), ('ארונטערגע', 'separable', 'arˈintərɡə'), ('אריבערגע', 'separable', 'arˈibərɡə'), ('אדורכגע', 'separable', 'adˈirxɡə'), ('אוועקגע', 'separable', 'avˈɛkɡə'), ('אהיימגע', 'separable', 'ahˈajmɡə'), ('צוריקגע', 'separable', 'ʦirˈikɡə'), ('ארויס', 'separable', 'arˈoʊs'), ('ארויפ', 'separable', 'arˈoʊf'), ('אראפ', 'separable', 'arˈup'), ('אריין', 'separable', 'arˈaːn'), ('ארונטער', 'separable', 'arˈintər'), ('אריבער', 'separable', 'arˈibər'), ('אדורך', 'separable', 'adˈirx'), ('אוועק', 'separable', 'avˈɛk'), ('אהיים', 'separable', 'ahˈajm'), ('אנידער', 'separable', 'anˈidər'), ('ארום', 'separable', 'arˈim'), ('אהין', 'separable', 'ahˈin'), ('אהער', 'separable', 'ahˈɛr'), ('צוריק', 'separable', 'ʦirˈik'), ('אונטער', 'separable', 'ˈintər'), ('איבער', 'separable', 'ˈibər'), ('דורכ', 'separable', 'dˈirx'), ('פארביי', 'separable', 'farbˈaj'), ('צוזאמען', 'separable', 'ʦizˈamən'), ('אנטקעגן', 'separable', 'antkˈejɡn'), ('אויס', 'separable', 'ˈoʊs'), ('אויפ', 'separable', 'ˈoʊf'), ('איינ', 'separable', 'ˈaːn'), ('אפ', 'separable', 'ˈup'), ('אנ', 'separable', 'ˈun'), ('מיט', 'separable', 'mˈit'), ('צו', 'separable', 'ʦˈi'), ('גע', 'unstressed', 'ɡə'), ('בא', 'unstressed', 'ba'), ('בע', 'unstressed', 'bə'), ('פאר', 'unstressed', 'far'), ('דער', 'unstressed', 'dər'), ('צע', 'unstressed', 'ʦə'), ('מיס', 'unstressed', 'mis')]
_PREFIX_RESCUE_TABLE.sort(key=lambda item: len(item[0]), reverse=True)
_PREFIX_NO_SPLIT: frozenset[str] = frozenset({'פארעמ', 'אײנטאג', 'איינטאג'})
_PREFIX_BAD_STEMS: frozenset[str] = frozenset({'דער', 'דעם', 'די', 'דאס', 'דו', 'ער', 'זי', 'עס', 'איר', 'מיר', 'זיי', 'אים', 'אונז', 'כי', 'עם', 'אין', 'פון', 'פאר', 'פארן', 'צו', 'ביי', 'נאך', 'און', 'אבער', 'נאר', 'אויך', 'איז', 'האט', 'וואס', 'ווי', 'ווער', 'וועט'})

def _ensure_stress(ipa: str) -> str:
    if STRESS in ipa:
        return ipa
    i = 0
    while i < len(ipa):
        if any((ipa.startswith(v, i) for v in PHONE_VOWELS)):
            return ipa[:i] + STRESS + ipa[i:]
        i += 1
    return ipa

def _get_stem_reading(stem: str) -> tuple[str, str, str] | tuple[None, None, None]:
    key = lexicon_key(stem)
    bare = _strip_points(stem)
    if key in GOLD_LEXICON:
        entry = GOLD_LEXICON[key]
        return (entry['ipa_primary'], entry.get('layer', 'G'), 'gold')
    if bare in _WORD_LATIN:
        ipa = hebrew_to_ipa(bare, stress=True)
        return (ipa, 'G', 'word-latin')
    if bare in _LK_BARE:
        ipa = hebrew_to_ipa(bare, stress=True)
        return (ipa, 'L', 'lk')
    return (None, None, None)

def _prefix_stem_rescue(core: str) -> dict | None:
    if _full_form_resolves(core) or lexicon_key(core) in _PREFIX_NO_SPLIT:
        return None
    bare = _strip_points(normalize_surface(core))
    if len(bare) < 4:
        return None
    for prefix, ptype, prefix_ipa in _PREFIX_RESCUE_TABLE:
        if bare.startswith(prefix) and len(bare) >= len(prefix) + 2:
            stem = bare[len(prefix):]
            if (ptype == 'separable' or prefix == 'גע') and stem in _PREFIX_BAD_STEMS:
                continue
            stem_ipa, layer, src = _get_stem_reading(stem)
            if stem_ipa is None:
                continue
            if ptype == 'unstressed':
                stem_ipa = _ensure_stress(stem_ipa)
                raw = prefix_ipa + stem_ipa
            else:
                stem_nostress = stem_ipa.replace(STRESS, '')
                raw = prefix_ipa + stem_nostress
            joined = reduce_unstressed(postlexical(raw))
            if ipa_phone_violations(joined):
                continue
            return _entry_result(core, joined, [], layer, 'rule', 'MED', f'prefix-rescue:{src}+{prefix}')
    return None
_LK_PROCLITICS: dict[str, str] = {'ה': 'ha', 'ב': 'bə', 'ל': 'lə', 'כ': 'kə', 'ד': 'də', 'ש': 'ʃə', 'מ': 'mi', 'ו': 'və'}

def _lk_stem_reading(stem: str) -> tuple[str, str] | None:
    key = lexicon_key(stem)
    bare = _strip_points(normalize_surface(stem))
    entry = GOLD_LEXICON.get(key)
    if entry is not None and entry['layer'] == 'L':
        return (entry['ipa_primary'], 'gold')
    if bare in _LK_BARE:
        return (_rule_path_ipa(stem, stress=True), 'lk-lexicon')
    entry = _AUDIO_ENDORSED.get(key)
    if entry is not None:
        return (entry['ipa'], 'audio-endorsed')
    entry = _SEFARIA_POINTED.get(key)
    if entry is not None:
        return (entry['ipa'], 'sefaria')
    entry = _PRINTED_RESPELLING.get(key)
    if entry is not None and entry.get('join') == 'exact':
        return (entry['ipa'], 'respelling')
    if any((pat.fullmatch(bare) for pat, _ in _STEM_SUB_RE)):
        return (_rule_path_ipa(stem, stress=True), 'stem-sub')
    return None
_PROCLITIC_ONSET: dict[str, str] = {'ה': 'h', 'ב': 'b', 'ל': 'l', 'כ': 'k', 'ד': 'd', 'ש': 'ʃ', 'מ': 'm', 'ו': 'v'}

def _repair_lk_proclitic(core: str, ipa: str) -> dict | None:
    bare = _strip_points(normalize_surface(core))
    if len(bare) < 4 or bare[0] not in _LK_PROCLITICS:
        return None
    naked = ipa.replace(STRESS, '')
    onset = _PROCLITIC_ONSET[bare[0]]
    if not naked.startswith(onset):
        return None
    rest = naked[len(onset):]
    if not rest or any((rest.startswith(v) for v in PHONE_VOWELS)):
        return None
    if not _lk_detector(bare):
        return None
    stem = bare[1:]
    if len(stem) < 3:
        return None
    got = _lk_stem_reading(stem)
    if got is None:
        return None
    stem_ipa, src = got
    whole = _lk_table_reading(core)
    if whole is not None:
        whole_ipa, whole_src = whole
        naked_whole = whole_ipa.replace(STRESS, '')
        if not naked_whole.startswith(onset) or any((naked_whole[len(onset):].startswith(v) for v in PHONE_VOWELS)):
            if whole_ipa == ipa or ipa_phone_violations(whole_ipa):
                return None
            return _entry_result(core, whole_ipa, [], 'L', 'rule', 'LOW', f'lk-proclitic-pointed:{whole_src}')
        return None
    joined = reduce_unstressed(postlexical(_LK_PROCLITICS[bare[0]] + _ensure_stress(stem_ipa)))
    if ipa_phone_violations(joined) or joined == ipa:
        return None
    return _entry_result(core, joined, [], 'L', 'rule', 'LOW', f'lk-proclitic:{src}+{bare[0]}')

def _route_token_inner(core: str) -> dict:
    key = lexicon_key(core)
    bare = _strip_points(normalize_surface(core))
    if key in _ABBREVIATIONS:
        primary, alts = _ABBREVIATIONS[key]
        return _entry_result(core, primary, [primary, *alts], 'A', 'lexicon', 'HIGH')
    if '"' in key:
        if key in _ACRONYM_WORDS:
            primary, alts = _ACRONYM_WORDS[key]
            return _entry_result(core, primary, [primary, *alts], 'A', 'lexicon', 'LOW', 'acronym-word')
        named = letter_name_word_ipa(key)
        if named:
            return _entry_result(core, named, [], 'A', 'rule', 'LOW', 'letter-name-word')
        spelled = letter_name_ipa(key)
        if spelled:
            return _entry_result(core, spelled, [], 'A', 'rule', 'LOW', 'letter-names')
        approx = _rule_path_ipa(key.replace('"', ''), stress=True)
        return _entry_result(core, approx, [], 'A', 'fallback', 'LOW', 'unknown-abbreviation')
    unquoted = core.strip("'")
    if unquoted and unquoted != core:
        return _route_token(unquoted)
    if key in _MULTIWORD or key in _MULTIWORD_LEGACY:
        primary = _multiword_ipa(key)
        alts = _MULTIWORD.get(key, ('', []))[1]
        conf, why = _multiword_confidence(key)
        return _entry_result(core, primary, [primary, *alts], 'L', 'lexicon', conf, why)
    entry = GOLD_LEXICON.get(key)
    if entry is not None and (not (bare in _WH_WHEN_POINTED and _vowel_point(core))) and (not _pe_point_contradicts(core, entry['ipa_primary'])):
        return _entry_result(core, entry['ipa_primary'], entry['variants'], entry['layer'], 'lexicon', 'HIGH')
    if bare in _LK_BARE or bare in _WORD_LATIN:
        primary = _rule_path_ipa(core, stress=True)
        layer = 'L' if bare in _LK_BARE else _guess_layer(core)
        return _entry_result(core, primary, [], layer, 'lexicon', 'HIGH')
    entry = _AUDIO_PE.get(key)
    if entry is not None and _strip_points(core) == core:
        return _entry_result(core, entry['ipa'], [], _guess_layer(core), 'lexicon', 'MED', 'audio-pe')
    entry = _AUDIO_VOWEL.get(key)
    if entry is not None and _strip_points(core) == core:
        return _entry_result(core, entry['ipa'], [], _guess_layer(core), 'lexicon', 'MED', 'audio-vowel')
    if '-' in core and any((p for p in core.split('-'))):
        parts = [p for p in core.split('-') if p]
        routed = [_route_token(p) for p in parts]
        primary = '-'.join((r['ipa_primary'] for r in routed))
        route = 'lexicon' if all((r['route'] == 'lexicon' for r in routed)) else 'fallback' if any((r['route'] == 'fallback' for r in routed)) else 'rule'
        conf = 'LOW' if any((r['confidence'] == 'LOW' for r in routed)) else 'MED' if any((r['confidence'] == 'MED' for r in routed)) else 'HIGH'
        return _entry_result(core, primary, [], routed[0]['layer'], route, conf, 'hyphen-split')
    clitic = ''
    rest = ''
    if len(core) > 2 and core[0] in _CLITIC_IPA and (core[1] == "'"):
        clitic, rest = (core[0], core[2:])
    elif len(core) > 2 and core[0] in _CLITIC_IPA and (core[1] == 'א') and _known_word(core[1:]):
        clitic, rest = (core[0], core[1:])
    if clitic and rest:
        tail = _route_token(rest)
        primary = _CLITIC_IPA[clitic] + tail['ipa_primary']
        conf = 'MED' if tail['confidence'] == 'HIGH' else tail['confidence']
        return _entry_result(core, primary, [], tail['layer'], 'rule', conf, 'clitic')
    stemmed = _stem_suffix_rescue(core)
    if stemmed is not None:
        return stemmed
    prefixed = _prefix_stem_rescue(core)
    if prefixed is not None:
        return prefixed
    is_lk = _lk_detector(bare)
    primary = _rule_path_ipa(core, stress=True, lk_penult=is_lk and _lk_nikud(core) and (not _has_stem_sub(core)))
    reasons = []
    if _has_ambiguous_alef(core):
        reasons.append('alef-default')
    if _has_ambiguous_pe(core):
        reasons.append('pe-default')
    if is_lk:
        reasons.append('lk-fallback')
        if not _lk_evidence(core):
            return _entry_result(core, primary, [], 'L', 'fallback', 'LOW', ','.join(reasons))
    conf = 'LOW' if reasons else 'MED'
    return _entry_result(core, primary, [], _guess_layer(core), 'rule', conf, ','.join(reasons))
_ROUTE_CACHE: dict[str, dict] = {}
_SENSE_HOMOGRAPHS: dict[str, dict] = {lexicon_key('טויב'): {'alt': 'toʊb', 'sense': 'dove'}, lexicon_key('טויבן'): {'alt': 'toʊbn', 'sense': 'dove'}}
_NOUN_SLOT_ARTICLES = {'די'}

def _apply_sense_context(rec: dict, prev_core: str | None, prev_trail: str, lead: str) -> dict:
    entry = _SENSE_HOMOGRAPHS.get(lexicon_key(rec['word']))
    if entry is None or not prev_core or prev_trail or lead:
        return rec
    if lexicon_key(prev_core) not in _NOUN_SLOT_ARTICLES:
        return rec
    old = rec['ipa_primary']
    rec = dict(rec)
    rec['ipa_primary'] = entry['alt']
    rec['variants'] = [old] + [v for v in rec['variants'] if v != entry['alt']]
    rec['reason'] = (rec['reason'] + ',' if rec['reason'] else '') + 'sense-homograph:' + entry['sense']
    return rec

def g2p_token(word: str, context: str | None=None) -> dict:
    _, core, _ = split_affixes(normalize_surface(word))
    if not core:
        return _entry_result(word, '', [], 'X', 'fallback', 'LOW', 'empty')
    cached = _ROUTE_CACHE.get(core)
    if cached is None:
        cached = _route_token(core)
        _ROUTE_CACHE[core] = cached
    result = _with_auto_variants(dict(cached, word=word))
    if context is not None and lexicon_key(core) in _SENSE_HOMOGRAPHS:
        toks = normalize_surface(strip_tags(context)).split()
        for j, tok in enumerate(toks):
            t_lead, t_core, _t_trail = split_affixes(tok)
            if t_core != core:
                continue
            prev_core, prev_trail = (None, '')
            if j:
                _, prev_core, prev_trail = split_affixes(toks[j - 1])
            result = _apply_sense_context(result, prev_core, prev_trail, t_lead)
            break
    if 'alef-default' in result['reason']:
        A_DEFAULT_LOG[core] += 1
    if 'pe-default' in result['reason']:
        P_DEFAULT_LOG[core] += 1
    return result

def _multiword_match(tokens: list[str], i: int) -> tuple[int, str, str] | None:
    for n in range(min(_MAX_MULTIWORD, len(tokens) - i), 1, -1):
        split = [split_affixes(t) for t in tokens[i:i + n]]
        cores = [core for _, core, _ in split]
        if not all(cores):
            continue
        if any((trail for _, _, trail in split[:-1])) or any((lead for lead, _, _ in split[1:])):
            continue
        key = lexicon_key(' '.join(cores))
        if key in _MULTIWORD or key in _MULTIWORD_LEGACY:
            return (n, key, _multiword_ipa(key))
    return None
_CONTEXT_READER = None

def set_context_reader(reader) -> None:
    global _CONTEXT_READER
    _CONTEXT_READER = reader

def g2p_tokens(text: str) -> list[dict]:
    records = _route_records(text)
    if _CONTEXT_READER is not None:
        records = _CONTEXT_READER(text, records)
    return records

def _route_records(text: str) -> list[dict]:
    tokens = normalize_surface(strip_tags(text)).split()
    records: list[dict] = []
    prev_core: str | None = None
    prev_trail = ''
    i = 0
    while i < len(tokens):
        match = _multiword_match(tokens, i)
        if match is not None:
            count, key, ipa = match
            cores = [split_affixes(t)[1] for t in tokens[i:i + count]]
            conf, why = _multiword_confidence(key)
            rec = _with_auto_variants(_entry_result(' '.join(cores), ipa, [], 'L', 'lexicon', conf, why))
            rec['lead'] = split_affixes(tokens[i])[0]
            rec['trail'] = split_affixes(tokens[i + count - 1])[2]
            records.append(rec)
            prev_core, prev_trail = (cores[-1], rec['trail'])
            i += count
            continue
        lead, core, trail = split_affixes(tokens[i])
        if core:
            rec = _apply_sense_context(g2p_token(core), prev_core, prev_trail, lead)
        else:
            rec = _entry_result(tokens[i], '', [], 'X', 'fallback', 'LOW', 'punctuation')
            lead, trail = (tokens[i], '')
        rec['lead'], rec['trail'] = (lead, trail)
        records.append(rec)
        prev_core, prev_trail = (core or None, trail)
        i += 1
    return records

def hebrew_to_ipa(text: str, stress: bool=True, quarantine: bool=True) -> str:
    if not stress:
        return _rule_path_ipa(text, stress=False)
    out = [r['lead'].replace('"', '') + ('' if quarantine and r['route'] == 'fallback' else r['ipa_primary']) + r['trail'].replace('"', '') for r in g2p_tokens(text)]
    return normalize_ipa_spacing(' '.join(out))
_LK_MARKER = re.compile('[\\u05d7\\u05ea]|\\u05e9\\u05c2')

def find_oov_loshn_koydesh(text: str) -> list[str]:
    after_lexicon = _LK_PATTERN.sub(_lk_replace, strip_tags(text))
    out = []
    for word in after_lexicon.split():
        bare = _strip_points(word.strip('.,!?;:"\'()-'))
        if bare and _LK_MARKER.search(bare) and (bare not in _LK_BARE):
            out.append(bare)
    return out

def validate_ipa_vocab(ipa: str, char_to_id: dict[str, int]) -> tuple[str, list[str]]:
    missing = sorted({ch for ch in ipa if ch not in char_to_id and (not ch.isspace())})
    return (ipa, missing)
_WH_POINT_TO_IPA: dict[str, str] = {HIRIQ: 'i', TSERE: 'aj', 'ֶ': 'ɛ', PATAH: 'a', QAMATS: 'u', HOLAM: 'ɔj', 'ֺ': 'ɔj', QUBUTS: 'u', 'ׇ': 'u', 'ֱ': 'ɛ', 'ֲ': 'a', 'ֳ': 'u'}
_WH_LONG_POINTS = frozenset({QAMATS, TSERE, HOLAM, 'ֺ', QUBUTS})
_WH_CONSONANT: dict[str, str] = {'ב': 'v', 'ג': 'ɡ', 'ד': 'd', 'ה': 'h', 'ו': 'v', 'ז': 'z', 'ח': 'x', 'ט': 't', 'י': 'j', 'כ': 'x', 'ך': 'x', 'ל': 'l', 'מ': 'm', 'ם': 'm', 'נ': 'n', 'ן': 'n', 'ס': 's', 'פ': 'f', 'ף': 'f', 'צ': 'ʦ', 'ץ': 'ʦ', 'ק': 'k', 'ר': 'r', 'ש': 'ʃ', 'ת': 's', 'א': '', 'ע': ''}
_WH_DAGESH_HARD: dict[str, str] = {'ב': 'b', 'כ': 'k', 'ך': 'k', 'פ': 'p', 'ף': 'p', 'ת': 't'}

def _wh_consonant(ch: str, marks: str, initial: bool=False) -> str:
    if ch == 'ש':
        return 's' if SIN_DOT in marks else 'ʃ'
    if ch in _WH_DAGESH_HARD and RAFE not in marks and (DAGESH in marks or initial):
        return _WH_DAGESH_HARD[ch]
    return _WH_CONSONANT.get(ch, '')

def _wh_vowel_point(marks: str) -> str:
    for mark in marks:
        if mark in _WH_POINT_TO_IPA:
            return mark
    return ''

def _wh_sheva_is_na(units: list[tuple[str, str]], i: int, emitted_vowel: bool, prev_point: str, prev_was_nach: bool) -> bool:
    if i == len(units) - 1:
        return False
    if not emitted_vowel:
        return True
    if prev_was_nach:
        return True
    return prev_point in _WH_LONG_POINTS

def _wh_word(word: str) -> list[tuple[str, bool]]:
    units = [(b, m) for b, m in _split_units(word) if b.strip()]
    out: list[tuple[str, bool]] = []
    n = len(units)
    i = 0
    prev_point = ''
    prev_was_nach = False
    emitted_vowel = False

    def nxt(k: int) -> str:
        return units[k][0] if 0 <= k < n else ''
    while i < n:
        ch, marks = units[i]
        point = _wh_vowel_point(marks)
        if ch == 'ו' and DAGESH in marks and (not point):
            out.append(('u', True))
            prev_point, prev_was_nach, emitted_vowel = (QUBUTS, False, True)
            i += 1
            continue
        if ch == 'ו' and point in (HOLAM, 'ֺ') and (not prev_was_nach):
            out.append(('ɔj', True))
            prev_point, prev_was_nach, emitted_vowel = (HOLAM, False, True)
            i += 1
            continue
        if ch == 'ו' and (not marks) and (prev_point in (HOLAM, 'ֺ', QUBUTS)):
            i += 1
            continue
        if ch == 'י' and (not marks) and (prev_point in (HIRIQ, TSERE)):
            i += 1
            continue
        if ch == 'ה' and i == n - 1 and (not point):
            i += 1
            continue
        if i == n - 1 and ch in ('ח', 'ע', 'ה') and (point in (PATAH, 'ֲ')) and emitted_vowel and (prev_point not in (PATAH, QAMATS, 'ׇ', 'ֲ')):
            out.append(('a', True))
            cons = _wh_consonant(ch, marks, initial=False)
            if cons:
                out.append((cons, False))
            prev_point, prev_was_nach, emitted_vowel = (PATAH, False, True)
            i += 1
            continue
        cons = _wh_consonant(ch, marks, initial=i == 0)
        if cons:
            out.append((cons, False))
        if SHEVA in marks:
            na = _wh_sheva_is_na(units, i, emitted_vowel, prev_point, prev_was_nach)
            if na:
                out.append(('ə', True))
                emitted_vowel = True
            prev_point, prev_was_nach = ('', not na)
            i += 1
            continue
        if point:
            out.append((_WH_POINT_TO_IPA[point], True))
            prev_point = point
            prev_was_nach = False
            emitted_vowel = True
            i += 1
            continue
        prev_point, prev_was_nach = ('', False)
        i += 1
    return out

def _wh_stress(segments: list[tuple[str, bool]]) -> str:
    vowels = [k for k, (_, is_v) in enumerate(segments) if is_v]
    if len(vowels) < 2:
        return ''.join((s for s, _ in segments))
    full = [k for k in vowels if segments[k][0] != 'ə']
    if len(full) >= 2:
        at = full[-2]
    elif full:
        at = full[0]
    else:
        at = vowels[-2]
    return ''.join((STRESS + s if k == at else s for k, (s, _) in enumerate(segments)))

def read_pointed_wh(pointed: str) -> str:
    out = []
    for word in re.split('[\\s־]+', strip_tags(pointed)):
        segments = _wh_word(word) if word else []
        if segments:
            out.append(_wh_stress(segments))
    return ' '.join(out)
_BEGADKEFAT = frozenset('בכךפףת')

def _explicit_begadkefat(word: str) -> str:
    units = _split_units(word)
    out: list[str] = []
    first = True
    for ch, marks in units:
        if _HEBREW_CHAR.match(ch) and ch in _BEGADKEFAT and (not (DAGESH in marks or RAFE in marks)):
            marks += DAGESH if first else RAFE
        if _HEBREW_CHAR.match(ch):
            first = False
        out.append(ch + marks)
    return ''.join(out)

def read_pointed_merged(pointed: str) -> str:
    out = []
    for word in re.split('[\\s־]+', strip_tags(pointed)):
        if not word:
            continue
        ipa = _rule_path_ipa(_explicit_begadkefat(word), stress=True, lk_penult=True, stem_subs=False)
        if ipa:
            out.append(ipa)
    return normalize_ipa_spacing(' '.join(out))
