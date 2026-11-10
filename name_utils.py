"""
Understand a person's name from a spoken or typed introduction.

Speech recognition is the weak point for names ("Saja" -> "Sasha", "Sajah", "such a"), so this module:
  * pulls the name out of many phrasings ("my name is ...", "call me ...", "it's ..., nice to meet you"),
  * understands spelled-out names ("S A J A", "s-a-j-a", "Saja, that's S-A-J-A"),
  * works with any alphabet (Arabic, Cyrillic, accented Latin, ...),
  * collects the speech engine's alternative transcriptions as candidate spellings,
  * recognises yes / no answers, so the app can confirm the name before using it.
"""
import re

# One word: a letter followed by letters, apostrophes or hyphens (unicode aware, no digits/underscore).
_L = r"[^\W\d_]"
_WORD = rf"{_L}(?:{_L}|['’\-])*"

# Phrases that introduce a name, most specific first.
_PATTERNS = [
    r"\bmy (?:first |full )?name(?:'s|’s| is| was)?\b", r"\bname(?:'s|’s)\b",
    r"\b(?:you can|you could|you may|just|please|they|people|everyone|everybody|friends|my friends)\s+(?:call|calls|say)(?: me)?\b",
    r"\bcall(?:s)? me\b", r"\bi go by\b", r"\bthis is\b",
    r"\bi am\b", r"\bi'm\b", r"\bi’m\b", r"\bim\b", r"\bit's\b", r"\bit’s\b", r"\bit is\b",
    # Arabic: "my name is", "I am", "call me"
    r"\bاسمي\b", r"\bاسمى\b", r"\bأنا\b", r"\bانا\b", r"\bناديني\b", r"\bنادني\b", r"\bيسموني\b",
]

# Words that are never part of a name. If the "name" starts with one, it was not a name
# ("I'm stressed", "I am feeling tired"); if one appears after the name, we stop there.
_STOP = set("""
a an the and but or so because then also please thanks thank thx nice pleased glad happy great good fine ok okay
well yeah yes no not very really just kind sort feeling feel feels doing going having getting being here there
from in at on today tonight now again still
i i'm im am is are was my me you we it
stressed anxious tired sad worried overwhelmed upset angry nervous scared stressed-out exhausted burnt burned
name called named hello hi hey meet
that that's thats this these those right correct exactly sure yep yup yeah nope nah thing something
""".split())

_FILLERS = re.compile(r"^(?:(?:uh|um|umm|erm|hmm|hi|hello|hey|so|well|okay|ok|yeah)\b[\s,.!-]*)+", re.I)

# A run of single letters: "S A J A", "s-a-j-a", "S. A. J. A."  (three or more)
_SPELLED = re.compile(rf"(?<![\w'’])(?:{_L}(?:[\s.\-,]+|$)){{3,}}(?![\w'’])", re.U)

_YES = re.compile(
    r"^\W*(?:yes|yeah|yep|yup|yea|ya|correct|right|exactly|precisely|sure|mhm+|uh[- ]?huh|perfect|"
    r"that(?:'s|’s| is)? (?:right|correct|it|me)|it(?:'s|’s| is)? (?:right|correct)|you got it|"
    r"نعم|ايوه|أيوه|ايوا|أيوا|صح|صحيح|تمام|اي)\b", re.I | re.U)
_NO = re.compile(
    r"^\W*(?:no|nope|nah|not (?:quite|really|exactly)|wrong|incorrect|that(?:'s|’s| is)? (?:not|wrong)|"
    r"it(?:'s|’s| is)? not|actually|لا|كلا|غلط|مو)\b", re.I | re.U)


def _cap_word(word):
    # keep deliberate inner capitals ("DeShawn", "McKay"); otherwise capitalise each hyphen/apostrophe part
    if any(c.isupper() for c in word[1:]) and not word.isupper():
        return word[0].upper() + word[1:]
    word = word.lower()
    word = re.sub(rf"(^|\-)({_L})", lambda m: m.group(1) + m.group(2).upper(), word)    # Mary-Jane
    # O'Brien, D'Angelo - but not "that's" -> "That'S"
    return re.sub(rf"(['’])({_L})(?={_L}{{2}})", lambda m: m.group(1) + m.group(2).upper(), word)


def format_name(name):
    """'saja' -> 'Saja', "o'brien" -> "O'Brien", 'mary-jane' -> 'Mary-Jane'."""
    name = " ".join((name or "").split()).strip(".,!?;:\"'“”")
    return " ".join(_cap_word(w) for w in name.split()) if name else None


def _take_name(fragment):
    """Up to three leading words of `fragment`, cut off at the first stop word."""
    fragment = _FILLERS.sub("", fragment.strip())
    m = re.match(rf"{_WORD}(?:[ \t]+{_WORD}){{0,2}}", fragment, re.U)
    if not m:
        return None
    words = []
    for w in m.group(0).split():
        if w.lower().strip("'’") in _STOP:
            break
        words.append(w)
    return " ".join(words) or None


def spelled_name(text):
    """The name spelled out letter by letter in `text`, or None."""
    best = None
    for m in _SPELLED.finditer(text or ""):
        letters = re.findall(_L, m.group(0), re.U)
        if len(letters) >= 3 and (best is None or len(letters) > len(best)):
            # "I a m" style noise: every letter is I / A  -> not a name
            if all(c.lower() in "ia" for c in letters):
                continue
            best = letters
    return format_name("".join(best)) if best else None


def extract_name(text):
    """Return the name in `text`, or None if it doesn't contain one."""
    if not text or not text.strip():
        return None
    text = text.strip()

    spelled = spelled_name(text)
    if spelled:
        return spelled

    for pat in _PATTERNS:
        m = re.search(pat + r"[\s,:]*(.*)", text, flags=re.I | re.S | re.U)
        if m:
            name = _take_name(m.group(1))
            if name:
                return format_name(name)

    # The user may have simply said their name ("Saja", "hi, Saja").
    name = _take_name(text)
    # Only trust it if it was the whole answer, not a sentence that happened to start with a name-like word.
    if name and len(text.split()) <= 3 and len(name.split()) == len(_FILLERS.sub("", text).strip(" .,!?").split()):
        return format_name(name)
    return None


def name_candidates(*texts):
    """Distinct names found across several transcriptions (e.g. a speech engine's n-best list), best first."""
    seen, out = set(), []
    for t in texts:
        for item in (t if isinstance(t, (list, tuple)) else [t]):
            n = extract_name(item) if isinstance(item, str) else None
            if n and n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
    return out


def is_yes(text):
    return bool(text) and bool(_YES.match(text.strip()))


def is_no(text):
    return bool(text) and bool(_NO.match(text.strip()))


def clean_typed_name(raw, limit=40):
    """Sanitise a name the user typed in the correction box (keeps any alphabet)."""
    raw = " ".join((raw or "").split())[:limit]
    raw = "".join(c for c in raw if c.isalpha() or c in " '’-.")
    return format_name(raw)


def spell_out(name):
    """'Saja' -> 'S, A, J, A' - read back so the user can check the spelling."""
    return ", ".join(c.upper() for c in (name or "") if c.isalpha())
