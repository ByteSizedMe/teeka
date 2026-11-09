"""Rule-based reply judge: complying / refusing / unsure.

Rules, in order:
1. Three digits in a row (digits or spoken number words) always means sharing.
2. Otherwise find phrase matches (Devanagari and Roman script). The longest
   match wins where phrases overlap.
3. A negator just before a "complying" phrase flips it to refusing.
4. The last thing said wins. No match at all means unsure.

Nothing here stores or logs the text it is given.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

COMPLYING, REFUSING, UNSURE = "complying", "refusing", "unsure"

NEGATION_WINDOW = 2  # tokens before a complying phrase that can flip it
MIN_DIGITS = 3

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZERO_WIDTH = "​‌‍﻿"


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFD", text or "")
    t = t.replace("़", "")  # nukta: फ़ -> फ
    t = unicodedata.normalize("NFC", t)
    for z in _ZERO_WIDTH:
        t = t.replace(z, "")
    t = t.replace("ँ", "ं")  # chandrabindu -> anusvara: पाँच == पांच
    t = t.translate(_DEV_DIGITS).replace("’", "'")
    return t.lower()


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


def _load(path: Path) -> _Rules:
    data = json.loads(path.read_text(encoding="utf-8"))
    comply, resist, negators, numbers = set(), set(), set(), {}
    for lang, block in data.items():
        if lang.startswith("_"):
            continue
        comply.update(tuple(tokenize(p)) for p in block.get("comply", []))
        resist.update(tuple(tokenize(p)) for p in block.get("resist", []))
        for n in block.get("negators", []):
            negators.update(tokenize(n))
        for digit, words in block.get("number_words", {}).items():
            for w in words:
                for tok in tokenize(w):
                    numbers.setdefault(tok, digit)
    comply.discard(())
    resist.discard(())
    return _Rules(tuple(comply), tuple(resist), frozenset(negators), numbers)


_RULES = _load(Path(__file__).with_name("keywords.json"))


def reload(path: str | Path | None = None) -> None:
    """Re-read keywords.json (after editing it, or to test with another file)."""
    global _RULES
    _RULES = _load(Path(path) if path else Path(__file__).with_name("keywords.json"))


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


def classify_detail(text: str, lang: str = "hi-IN") -> dict:
    """Return {"label", "reason"}. `lang` is accepted for the API but all
    languages' keywords are used, because replies are often code-mixed."""
    tokens = tokenize(text)
    rules = _RULES

    if _find_digit_run(tokens, rules.numbers):
        return {"label": COMPLYING, "reason": "three digits in a row"}

    # (start, end, polarity, why)
    found = []
    for polarity, phrases in (("c", rules.comply), ("r", rules.resist)):
        for ph in phrases:
            n = len(ph)
            for i in range(len(tokens) - n + 1):
                if tuple(tokens[i : i + n]) == ph:
                    found.append((i, i + n, polarity, " ".join(ph)))

    # Longest match wins where matches overlap.
    found.sort(key=lambda m: (-(m[1] - m[0]), m[0]))
    taken, used = [], set()
    for m in found:
        span = set(range(m[0], m[1]))
        if span & used:
            continue
        taken.append(m)
        used |= span

    events, flipped = [], set()
    for start, end, pol, why in taken:
        if pol == "c":
            window = range(max(0, start - NEGATION_WINDOW), start)
            neg = [j for j in window if tokens[j] in rules.negators and j not in used]
            if neg:
                flipped.update(neg)
                events.append((start, REFUSING, f"negated: {why}"))
                continue
            events.append((start, COMPLYING, why))
        else:
            events.append((start, REFUSING, why))

    # A lone "no" that wasn't part of a phrase or a flip is a weak refusal.
    for j, tok in enumerate(tokens):
        if tok in rules.negators and j not in used and j not in flipped:
            events.append((j, REFUSING, "negator"))

    if not events:
        return {"label": UNSURE, "reason": "no decision"}
    events.sort(key=lambda e: e[0])
    _, label, why = events[-1]
    return {"label": label, "reason": f"last: {why}"}


def classify(text: str, lang: str = "hi-IN") -> str:
    return classify_detail(text, lang)["label"]


def mask_digits(text: str) -> str:
    """Replace every digit (ASCII or Devanagari) with the • mask."""
    out = []
    for ch in text or "":
        out.append("•" if ch.isdigit() else ch)
    return "".join(out)


def mask_for_display(text: str) -> str:
    """Mask every digit, and spoken number words that sit next to another number
    word (so "चार सात एक" is hidden but "एक मिनट" is not)."""
    pieces = (text or "").split(" ")
    is_num = []
    for raw in pieces:
        toks = tokenize(raw)
        is_num.append(bool(toks) and all(t in _RULES.numbers for t in toks))
    out = []
    for i, raw in enumerate(pieces):
        near = (i > 0 and is_num[i - 1]) or (i + 1 < len(pieces) and is_num[i + 1])
        out.append("•" if is_num[i] and near else mask_digits(raw))
    return " ".join(out)
