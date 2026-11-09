"""Rule-based judge for a person's reply: complying, refusing or unsure.

Rules (from the report):
* Three digits in a row always count as sharing, in numerals of any script or as
  spoken number words, even mixed ("4 7 1 नौ").
* A complying phrase with a negator next to it flips to refusing. Which side the
  negator sits on is a per-language setting (keywords.negation in the language pack).
* The last decision in the reply wins.
* Anything else is unsure, and the caller presses harder.

A reply is matched against its call's language plus English, because replies are
often code-mixed. Phrases live in content/languages/<lang>.json under "keywords".
Nothing here stores the reply.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

COMPLYING, REFUSING, UNSURE = "complying", "refusing", "unsure"

NEGATION_WINDOW = 2  # tokens beside a complying phrase that can flip it
MIN_DIGITS = 3
ENGLISH = "en-IN"
DEFAULT_LANG = "hi-IN"
CUE_ORDER = ("repeat", "wants_time", "asks_what", "confused")

_NUKTAS = "़়਼઼଼"
_ZERO_WIDTH = "​‌‍﻿"
_PACK_DIR = Path(__file__).parent / "content" / "languages"


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFD", text or "")
    for mark in _NUKTAS:
        t = t.replace(mark, "")  # nukta: फ़ -> फ
    t = unicodedata.normalize("NFC", t)
    for z in _ZERO_WIDTH:
        t = t.replace(z, "")
    t = t.replace("ँ", "ं")  # chandrabindu -> anusvara: पाँच == पांच
    t = "".join(str(unicodedata.decimal(ch)) if unicodedata.decimal(ch, None) is not None else ch for ch in t)
    return t.replace("’", "'").lower()


def tokenize(text: str) -> list[str]:
    out, cur = [], []
    for ch in normalize(text):
        if ch.isalnum() or ch == "'" or unicodedata.category(ch).startswith("M"):
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return [t.strip("'") for t in out if t.strip("'")]


@dataclass(frozen=True)
class _Rules:
    comply: tuple
    resist: tuple
    negators: frozenset
    numbers: dict
    directions: dict  # complying phrase -> where its negator sits
    cues: dict  # cue -> tuple of phrases


def _phrases(values) -> set:
    found = {tuple(tokenize(p)) for p in values}
    found.discard(())
    return found


def _load_blocks(path: Path) -> dict:
    blocks = {}
    for file in sorted(path.glob("*.json")):
        try:
            pack = json.loads(file.read_text(encoding="utf-8"))
            blocks[pack["lang"]] = dict(pack["keywords"])
        except (OSError, ValueError, KeyError, TypeError):
            continue  # content.load reports the broken pack on the dashboard
    return blocks


_pack_dir = _PACK_DIR
_blocks = _load_blocks(_pack_dir)
_cache: dict[str, _Rules] = {}


def reload(path: str | Path | None = None) -> None:
    """Re-read the language packs (after editing one, or to test with another folder)."""
    global _pack_dir, _blocks
    _pack_dir = Path(path) if path else _PACK_DIR
    _blocks = _load_blocks(_pack_dir)
    _cache.clear()


def _rules(lang: str | None) -> _Rules:
    lang = lang if lang in _blocks else DEFAULT_LANG
    if lang in _cache:
        return _cache[lang]
    comply, resist, negators, numbers, directions = set(), set(), set(), {}, {}
    cues: dict[str, set] = {c: set() for c in CUE_ORDER}
    # English first, so the call's own language decides any phrase both define.
    for code in [c for c in (ENGLISH, lang) if c in _blocks]:
        block = _blocks[code]
        own = _phrases(block.get("comply", []))
        comply |= own
        for phrase in own:
            directions[phrase] = block.get("negation", "before")
        resist |= _phrases(block.get("resist", []))
        for n in block.get("negators", []):
            negators.update(tokenize(n))
        for digit, words in block.get("number_words", {}).items():
            for w in words:
                for tok in tokenize(w):
                    numbers[tok] = digit
        for cue in CUE_ORDER:
            cues[cue] |= _phrases(block.get("unsure", {}).get(cue, []))
    built = _Rules(tuple(comply), tuple(resist), frozenset(negators), numbers, directions,
                   {c: tuple(v) for c, v in cues.items()})
    _cache[lang] = built
    return built


def _digit_count(tok: str, numbers: dict) -> int:
    if tok.isdigit():
        return len(tok)
    return 1 if tok in numbers else 0


def _find_digit_run(tokens: list[str], numbers: dict) -> tuple[int, int] | None:
    """Return (start, end) of the first run of >= MIN_DIGITS digits, else None."""
    run, start = 0, 0
    for i, tok in enumerate(tokens):
        c = _digit_count(tok, numbers)
        if c:
            if run == 0:
                start = i
            run += c
            if run >= MIN_DIGITS:
                end = i + 1
                while end < len(tokens) and _digit_count(tokens[end], numbers):
                    end += 1
                return start, end
        else:
            run = 0
    return None


def _spans(tokens: list[str], phrases) -> list[tuple[int, int, tuple]]:
    out = []
    for ph in phrases:
        n = len(ph)
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i : i + n]) == ph:
                out.append((i, i + n, ph))
    return out


def classify_detail(text: str, lang: str = DEFAULT_LANG) -> dict:
    """Return {"label", "reason"}."""
    tokens = tokenize(text)
    rules = _rules(lang)

    if _find_digit_run(tokens, rules.numbers):
        return {"label": COMPLYING, "reason": "three digits in a row"}

    # (start, end, polarity, phrase)
    found = [(s, e, "c", ph) for s, e, ph in _spans(tokens, rules.comply)]
    found += [(s, e, "r", ph) for s, e, ph in _spans(tokens, rules.resist)]

    # Longest match wins where matches overlap.
    found.sort(key=lambda m: (-(m[1] - m[0]), m[0]))
    taken, used = [], set()
    for m in found:
        span = set(range(m[0], m[1]))
        if span & used:
            continue
        taken.append(m)
        used |= span

    # "I don't understand" is confusion, not a refusal: its negator does not count.
    in_a_cue = set()
    for phrases in rules.cues.values():
        for s, e, _ in _spans(tokens, phrases):
            in_a_cue |= set(range(s, e))

    events, flipped = [], set()
    for start, end, pol, ph in taken:
        why = " ".join(ph)
        if pol == "c":
            direction = rules.directions.get(ph, "before")
            near = []
            if direction in ("before", "both"):
                near += range(max(0, start - NEGATION_WINDOW), start)
            if direction in ("after", "both"):
                near += range(end, min(len(tokens), end + NEGATION_WINDOW))
            neg = [j for j in near if tokens[j] in rules.negators and j not in used]
            if neg:
                flipped.update(neg)
                events.append((start, REFUSING, f"negated: {why}"))
                continue
            events.append((start, COMPLYING, why))
        else:
            events.append((start, REFUSING, why))

    # A lone "no" that wasn't part of a phrase or a flip is a weak refusal.
    for j, tok in enumerate(tokens):
        if tok in rules.negators and j not in used and j not in flipped and j not in in_a_cue:
            events.append((j, REFUSING, "negator"))

    if not events:
        return {"label": UNSURE, "reason": "no decision"}
    events.sort(key=lambda e: e[0])
    _, label, why = events[-1]
    return {"label": label, "reason": f"last: {why}"}


def classify(text: str, lang: str = DEFAULT_LANG) -> str:
    return classify_detail(text, lang)["label"]


def unsure_reason(text: str, lang: str = DEFAULT_LANG) -> str | None:
    """Why an unsure reply is unsure: repeat, wants_time, asks_what, confused, or None."""
    tokens = tokenize(text)
    rules = _rules(lang)
    for cue in CUE_ORDER:
        if _spans(tokens, rules.cues[cue]):
            return cue
    return None


def mask_digits(text: str) -> str:
    """Replace every digit (in any script) with the • mask."""
    return "".join("•" if ch.isdigit() else ch for ch in text or "")


def mask_for_display(text: str, lang: str = DEFAULT_LANG) -> str:
    """Mask every digit, and spoken number words that sit next to another number
    (so "चार सात एक" is hidden but "एक मिनट" is not)."""
    numbers = _rules(lang).numbers
    pieces = (text or "").split(" ")
    is_num = []
    for raw in pieces:
        toks = tokenize(raw)
        # Prisma mixes numerals and words in one reply ("4 7 1 नौ"), so both count.
        is_num.append(bool(toks) and all(t.isdigit() or t in numbers for t in toks))
    out = []
    for i, raw in enumerate(pieces):
        near = (i > 0 and is_num[i - 1]) or (i + 1 < len(pieces) and is_num[i + 1])
        out.append("•" if is_num[i] and near else mask_digits(raw))
    return " ".join(out)
