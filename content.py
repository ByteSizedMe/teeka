"""Loads and validates Teeka's content packs.

content/languages/<lang>.json   one per language: voices, teacher lines, phone labels,
                                red flags, lessons, classifier keywords
content/scams/<fraud>.json      one per fraud type: structure plus the lines per language

Adding a fraud type or a language needs data files only. A broken pack (or one broken
language inside a scam pack) is disabled with a dashboard notice and never crashes the
server. Safety rules are enforced here, at load time, for every language.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import classifier

CONTENT_ROOT = Path(__file__).parent / "content"

TEMPLATE_KEYS = {"reveal_intro", "reveal_outro", "praise", "wrap", "code_word"}
PHONE_KEYS = {"ringLabel", "declineLabel", "answerLabel", "endLabel", "endTitle", "done", "cancelled"}
CUES = ("repeat", "wants_time", "asks_what", "confused")
TTS_MODELS = ("timbre-v2.5", "timbre-v3.0")
REVIEW_STATES = ("draft", "native_reviewed")
VERDICTS = ("complying", "refusing", "unsure")
DEFAULT_TURNS = 3
MAX_TURNS = 5
MAX_LINE_CHARS = 230  # roughly 12 seconds when spoken, in every script tested

# Matched after classifier.normalize. Latin terms match whole words; the rest match anywhere.
BLOCKED_LATIN = {
    "police", "cbi", "ed officer", "enforcement directorate", "court", "judge", "customs", "rbi", "trai", "sebi",
    "government", "arrest", "arrested", "digital arrest", "warrant",
    "sbi", "hdfc", "icici", "axis", "kotak", "paytm", "phonepe", "gpay", "google pay", "bhim", "lic",
    "bluedart", "blue dart", "dtdc", "delhivery", "amazon", "flipkart", "olx", "meesho", "swiggy", "zomato",
    "airtel", "jio", "vodafone", "bsnl", "indane", "bharat gas", "hp gas", "tata power", "adani",
    "anydesk", "teamviewer", "quicksupport", "microsoft", "aadhaar", "aadhar", "pan card",
    "http", "https", "www",
}
BLOCKED_SCRIPT = [
    # police
    "पुलिस", "पोलीस", "পুলিশ", "પોલીસ", "ಪೊಲೀಸ್", "പോലീസ്", "ਪੁਲਿਸ", "போலீஸ்", "காவல்துறை", "పోలీసు",
    # CBI / ED / RBI / TRAI
    "सीबीआई", "सीबीआय", "আরবিআই", "সিবিআই", "સીબીઆઈ", "ಸಿಬಿಐ", "സിബിഐ", "ਸੀਬੀਆਈ", "சிபிஐ", "సీబీఐ",
    "आरबीआई", "आरबीआय", "આરબીઆઈ", "ಆರ್‌ಬಿಐ", "ആർബിഐ", "ਆਰਬੀਆਈ", "ஆர்பிஐ", "ఆర్బీఐ", "ट्राई",
    # court
    "अदालत", "न्यायालय", "कोर्ट", "আদালত", "কোর্ট", "અદાલત", "કોર્ટ", "ನ್ಯಾಯಾಲಯ", "ಕೋರ್ಟ್", "കോടതി", "ਅਦਾਲਤ", "நீதிமன்ற", "கோர்ட்", "కోర్టు", "న్యాయస్థాన",
    # arrest
    "गिरफ्त", "गिरफ़्त", "অ্যারেস্ট", "গ্রেফতার", "গ্রেপ্তার", "ધરપકડ", "ಬಂಧನ", "ಅರೆಸ್ಟ್", "അറസ്റ്റ്", "ਗ੍ਰਿਫਤਾਰ", "கைது", "అరెస్టు",
    # customs
    "कस्टम", "কাস্টম", "કસ્ટમ", "ಕಸ್ಟಮ್ಸ್", "കസ്റ്റംസ്", "ਕਸਟਮ", "சுங்க", "కస్టమ్స్",
    # government
    "सरकार", "সরকার", "સરકાર", "ಸರ್ಕಾರ", "സർക്കാർ", "ਸਰਕਾਰ", "அரசு", "ప్రభుత్వ",
    # real banks and brands in Indian scripts
    "एसबीआई", "एचडीएफसी", "आईसीआईसीआई", "पेटीएम", "एयरटेल", "एनीडेस्क", "টিমভিউয়ার", "எஸ்பிஐ", "ఎస్బీఐ",
]
BLOCKED_WORDS = {"ईडी"}
_BLOCK_LATIN_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in sorted(BLOCKED_LATIN, key=len, reverse=True)) + r")(?![a-z0-9])")
_BLOCK_SCRIPT = [classifier.normalize(t) for t in BLOCKED_SCRIPT]


class PackError(Exception):
    """A readable reason a pack cannot be used."""


def blocked_term(text: str) -> str | None:
    """The first forbidden organisation, threat or link found in `text`, else None."""
    t = classifier.normalize(text)
    m = _BLOCK_LATIN_RE.search(t)
    if m:
        return m.group(1)
    for term in _BLOCK_SCRIPT:
        if term in t:
            return term
    # "ईडी" is also the end of "आईडी" (ID), so it must be a whole word.
    return next((w for w in classifier.tokenize(text) if w in BLOCKED_WORDS), None)


def _need(cond, why: str) -> None:
    if not cond:
        raise PackError(why)


def _spoken(text, where: str) -> None:
    _need(isinstance(text, str) and text.strip(), f"{where} is empty")
    _need(not any(ch.isdigit() for ch in text), f"{where} has digits; write numbers as words")
    term = blocked_term(text)
    _need(term is None, f"{where} uses a forbidden term ({term})")


def _strings(values, where: str) -> None:
    _need(isinstance(values, list) and values and all(isinstance(v, str) and v.strip() for v in values), f"{where} must be a non-empty list of text")


def _check_language(p: dict) -> None:
    _need(isinstance(p.get("lang"), str) and re.fullmatch(r"[a-z]{2}-[A-Z]{2}", p["lang"]), "lang must look like hi-IN")
    for key in ("name", "native_name", "short"):
        _need(isinstance(p.get(key), str) and p[key].strip(), f"{key} is missing")
    _need(p.get("review") in REVIEW_STATES, "review must be draft or native_reviewed")
    _need(p.get("dir", "ltr") in ("ltr", "rtl"), "dir must be ltr or rtl")
    v = p.get("voices") or {}
    _need(isinstance(v.get("language"), str) and v["language"], "voices.language is missing")
    for model in TTS_MODELS:
        _need(all(isinstance((v.get(model) or {}).get(r), str) for r in ("caller", "teacher")), f"voices for {model} are missing")
    t = p.get("templates") or {}
    _need(TEMPLATE_KEYS <= set(t), "templates are incomplete")
    for key in TEMPLATE_KEYS:
        _spoken(t[key], f"templates.{key}")
    ph = p.get("phone") or {}
    _need(PHONE_KEYS <= set(ph), "phone labels are incomplete")
    for key in PHONE_KEYS:
        _need(isinstance(ph[key], str) and ph[key].strip(), f"phone.{key} is empty")
        _need(blocked_term(ph[key]) is None, f"phone.{key} uses a forbidden term")
    st = p.get("stamps") or {}
    _need(all(isinstance(st.get(k), str) and st[k].strip() for k in ("protected", "missed")), "stamps are missing")
    _need(isinstance(p.get("red_flags"), dict) and p["red_flags"], "red_flags are missing")
    _need(isinstance(p.get("lessons"), dict) and p["lessons"], "lessons are missing")
    for key, text in p["lessons"].items():
        _spoken(text, f"lessons.{key}")
    br = p.get("branches") or {}
    for cue in CUES:
        _strings(br.get(cue), f"branches.{cue}")
        for i, text in enumerate(br[cue]):
            _spoken(text, f"branches.{cue}[{i}]")
    for key in ("openers", "pressers"):
        _strings(p.get(key), key)
        for i, text in enumerate(p[key]):
            _spoken(text, f"{key}[{i}]")
    k = p.get("keywords") or {}
    _strings(k.get("comply"), "keywords.comply")
    _strings(k.get("resist"), "keywords.resist")
    _strings(k.get("negators"), "keywords.negators")
    _need(k.get("negation") in ("before", "after", "both"), "keywords.negation must be before, after or both")
    _need(set(k.get("number_words") or {}) == set("0123456789"), "keywords.number_words must cover 0 to 9")
    for cue in CUES:
        _strings((k.get("unsure") or {}).get(cue), f"keywords.unsure.{cue}")
    for row in p.get("test_replies") or []:
        _need(isinstance(row, list) and len(row) == 2 and row[1] in VERDICTS, "test_replies rows must be [text, verdict]")


def _check_scam_shape(p: dict) -> None:
    _need(isinstance(p.get("id"), str) and re.fullmatch(r"[a-z0-9-]+", p["id"]), "id must be lowercase letters, digits and hyphens")
    for key in ("title", "practise"):
        _need(isinstance(p.get(key), str) and p[key].strip(), f"{key} is missing")
    _strings(p.get("caller_ids"), "caller_ids")
    for cid in p["caller_ids"]:
        term = blocked_term(cid)
        _need(term is None and not any(ch.isdigit() for ch in cid), f"caller id {cid!r} is not allowed")
    turns = p.get("turn_limit", DEFAULT_TURNS)
    _need(isinstance(turns, int) and 1 <= turns <= MAX_TURNS, f"turn_limit must be 1 to {MAX_TURNS}")
    _need(isinstance(p.get("beats"), list) and p["beats"], "beats are missing")
    _need(isinstance(p.get("languages"), dict) and p["languages"], "languages are missing")


def fill(text: str, orgs: list, slots: dict, seed: int) -> str:
    """Put this drill's choices into a sentence. Each pool rotates with the seed."""
    out = text.replace("{org}", orgs[seed % len(orgs)]) if orgs else text
    for name, pool in slots.items():
        out = out.replace("{" + name + "}", pool[seed % len(pool)])
    return out


def _build_scenario(p: dict, lang: str, lp: dict) -> dict:
    sec = p["languages"][lang]
    _need(isinstance(sec.get("title"), str) and sec["title"].strip(), "title is missing")
    orgs = sec.get("orgs") or []
    _need(isinstance(orgs, list), "orgs must be a list")
    slots = sec.get("slots") or {}
    _need(isinstance(slots, dict) and all(isinstance(v, list) and v for v in slots.values()), "slots must be lists")
    beats = sec.get("beats")
    _need(isinstance(beats, list) and len(beats) == len(p["beats"]), "wrong number of beats")
    lines = []
    for i, (shape, beat) in enumerate(zip(p["beats"], beats)):
        where = f"beat {i + 1}"
        _strings(beat.get("variants"), f"{where} variants")
        flags = shape.get("flags") or []
        missing = [f for f in flags if f not in lp["red_flags"]]
        _need(flags and not missing, f"{where} has unknown red flags {missing}")
        lesson = beat.get("lesson") or lp["lessons"].get(shape.get("lesson"))
        _spoken(lesson, f"{where} lesson")
        for text in beat["variants"]:
            for name in re.findall(r"\{(\w+)\}", text):
                _need(name == "org" and orgs or name in slots, f"{where} uses {{{name}}} with no values")
            for n in range(max(1, len(orgs), *[len(v) for v in slots.values()] or [1])):
                filled = fill(text, orgs, slots, n)
                _spoken(filled, f"{where}")
                _need(len(filled) <= MAX_LINE_CHARS, f"{where} is too long to speak in about 12 seconds")
        lines.append({"variants": beat["variants"], "text": fill(beat["variants"][0], orgs, slots, 0), "flags": flags, "lesson": lesson})
    # A language can write fewer organisation names than the pack has caller ids.
    n_ids = len(orgs) if orgs else 1
    caller_ids = p["caller_ids"][:n_ids] or p["caller_ids"][:1]
    return {
        "id": f"{p['id']}-{lp['short']}", "fraud": p["id"], "lang": lang, "title": p["title"], "title_local": sec["title"],
        "practise": p["practise"], "caller_id": caller_ids[0], "caller_ids": caller_ids, "orgs": orgs, "slots": slots,
        "turn_limit": p.get("turn_limit", DEFAULT_TURNS), "lines": lines,
    }


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PackError("the file is not a JSON object")
    return data


def _why(e: Exception) -> str:
    return str(e) if isinstance(e, PackError) else f"{type(e).__name__}: {e}"


def load(root: Path | str | None = None) -> dict:
    root = Path(root) if root else CONTENT_ROOT
    warnings: list[str] = []
    languages: dict[str, dict] = {}
    for path in sorted((root / "languages").glob("*.json")):
        try:
            pack = _read(path)
            _check_language(pack)
            _need(pack["lang"] not in languages, f"{pack['lang']} is defined twice")
            languages[pack["lang"]] = pack
        except Exception as e:
            warnings.append(f"Language pack {path.name} is disabled: {_why(e)}")
    languages = dict(sorted(languages.items(), key=lambda kv: (kv[1].get("order", 99), kv[1]["name"])))

    packs = []
    for path in sorted((root / "scams").glob("*.json")):
        try:
            pack = _read(path)
            _check_scam_shape(pack)
            packs.append(pack)
        except Exception as e:
            warnings.append(f"Scam pack {path.name} is disabled: {_why(e)}")
    packs.sort(key=lambda p: (p.get("order", 99), p["id"]))

    scenarios = []
    for lang, lp in languages.items():
        for p in packs:
            if lang not in p["languages"]:
                continue
            try:
                scenarios.append(_build_scenario(p, lang, lp))
            except Exception as e:
                warnings.append(f"{p['id']} is disabled in {lp['name']}: {_why(e)}")
    return {"languages": languages, "scenarios": scenarios, "by_id": {s["id"]: s for s in scenarios}, "warnings": warnings}


class PackLineSource:
    """Chooses what the caller says. Keep this interface if a model is ever added:
    plan() returns one drill's caller id and first choice of line per beat, and
    opener() returns a sentence that responds to an unsure reply, or "".

    Choices rotate with `seed` (how many times this person has had this scam), so
    repeat calls differ while the set of audio to cache stays small."""

    def __init__(self, languages: dict):
        self.languages = languages

    def plan(self, scenario: dict, seed: int) -> dict:
        lp = self.languages[scenario["lang"]]
        texts = []
        for i, line in enumerate(scenario["lines"]):
            variants = line["variants"]
            text = fill(variants[seed % len(variants)], scenario["orgs"], scenario["slots"], seed)
            if len(variants) == 1 and seed > 0:  # one written sentence: vary it with stand-alone sentences
                extra = lp["openers"] if i == 0 else lp["pressers"]
                pick = extra[(seed + i - 1) % len(extra)]
                joined = f"{pick} {text}" if i == 0 else f"{text} {pick}"
                text = joined if len(joined) <= MAX_LINE_CHARS else text
            texts.append(text)
        return {"caller_id": scenario["caller_ids"][seed % len(scenario["caller_ids"])], "lines": texts}

    def opener(self, lang: str, cue: str | None, seed: int) -> str:
        pool = self.languages[lang]["branches"].get(cue) if cue else None
        return pool[seed % len(pool)] if pool else ""
