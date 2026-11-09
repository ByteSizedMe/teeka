"""Offline tests for classifier.py and the content packs.
Run:  python test_classifier.py     (or: python -m pytest)
"""
import json
import sys
from pathlib import Path

import classifier as c
import content

HERE = Path(__file__).parent

# The 20 reply phrases from the report's accuracy table (#, text, expected).
REPORT_20 = [
    ("अच्छा रुको, चार सात एक नौ", c.COMPLYING),
    ("हाँ, ओटीपी आया है", c.COMPLYING),
    ("लिख लो, बताती हूँ", c.COMPLYING),
    ("ठीक है, लिंक खोल रही हूँ", c.COMPLYING),
    ("कितने पैसे भेजने हैं?", c.COMPLYING),
    ("हाँ, पिन डाल रही हूँ", c.COMPLYING),
    ("ऐप डाउनलोड कर लेती हूँ", c.COMPLYING),
    ("भेज दिया, देखिए", c.COMPLYING),
    ("एक मिनट, नंबर बताती हूँ, दो पाँच आठ", c.COMPLYING),
    ("Okay, OTP is 4 7 1 9", c.COMPLYING),
    ("मैं पहले अपने बेटे से पूछ लेती हूँ", c.REFUSING),
    ("नहीं, मैं नहीं बताऊँगी", c.REFUSING),
    ("मैं फ़ोन रख रही हूँ", c.REFUSING),
    ("मैं खुद बैंक जाकर बात करूँगी", c.REFUSING),
    ("आप कौन बोल रहे हैं?", c.REFUSING),
    ("यह तो धोखा लग रहा है", c.REFUSING),
    ("मुझे नहीं चाहिए", c.REFUSING),
    ("मैं ओटीपी नहीं दूँगी", c.REFUSING),
    ("पुलिस को बताऊँगी", c.REFUSING),
    ("बाद में बात करना", c.REFUSING),
]

# Other behaviours the report describes, plus script / spelling variants.
EXTRA = [
    ("अच्छा रुको, चार, सात, एक", c.COMPLYING),            # report example (3 digits)
    ("मैं नहीं बता रही", c.REFUSING),                        # negation flips
    ("हाँ बताती हूँ… नहीं, बेटे से पूछूँगी", c.REFUSING),   # last thing wins
    ("नहीं, बेटे से पूछूँगी… हाँ बताती हूँ", c.COMPLYING),  # ...in both directions
    ("अरे बाप रे, अब क्या करूँ ?", c.UNSURE),               # no decision
    ("", c.UNSURE),
    ("४ ७ १ ९", c.COMPLYING),                               # Devanagari digits
    ("4719", c.COMPLYING),                                   # digit string
    ("my otp is one two three", c.COMPLYING),                # English number words
    ("ek teen saat", c.COMPLYING),                           # Roman Hindi number words
    ("एक मिनट रुकिए", c.UNSURE),                             # one number word is not sharing
    ("OTP bata rahi hoon", c.COMPLYING),                     # Roman Hindi
    ("main nahi bata rahi", c.REFUSING),
    ("phone rakh rahi hoon", c.REFUSING),
    ("I will not share it", c.REFUSING),
    ("I am calling the police", c.REFUSING),
    ("who are you?", c.REFUSING),
    ("here is the code", c.COMPLYING),
    ("पांच", c.UNSURE),                                      # chandrabindu / anusvara spelling
    ("मैं फोन रख रही हूं", c.REFUSING),                      # no nukta, anusvara spelling
]


def run(cases):
    bad = []
    for text, want in cases:
        got = c.classify(text)
        if got != want:
            bad.append((text, want, got, c.classify_detail(text)["reason"]))
    return bad


def test_report_20():
    assert not run(REPORT_20)


def test_extra():
    assert not run(EXTRA)


def test_masking():
    assert c.mask_digits("OTP 4719 ४७") == "OTP •••• ••"
    assert "4" not in c.mask_for_display("चार 7 सात")
    assert c.mask_for_display("चार, सात एक") == "• • •"
    assert c.mask_for_display("एक मिनट रुकिए") == "एक मिनट रुकिए"
    assert c.mask_for_display("अच्छा रुको, 4 7 1 नौ।") == "अच्छा रुको, • • • •"  # real Prisma output


def test_scenarios_and_keywords():
    data = content.load()
    assert not data["warnings"], data["warnings"]
    scen = data["scenarios"]
    assert {"bank-kyc-hi", "electricity-hi", "parcel-hi", "bank-kyc-en"} <= set(data["by_id"])
    for lang, p in data["languages"].items():
        k = p["keywords"]
        assert set(k["number_words"]) == set("0123456789") and k["comply"] and k["resist"], lang
        assert set(p["templates"]) >= {"reveal_intro", "reveal_outro", "praise", "wrap", "code_word"}
    for s in scen:
        assert len(s["lines"]) == 3, s["id"]
        for line in s["lines"]:
            assert line["lesson"] and set(line["flags"]) <= set(data["languages"][s["lang"]]["red_flags"])
            for text in line["variants"]:
                assert not any(ch.isdigit() for ch in text), "write numbers as words"
                assert len(text.split()) <= 30, (s["id"], len(text.split()))


def _spoken_strings():
    """Everything the system says or shows, from every pack. Keywords are what a person
    might say (\"I will call the police\"), so they are the one thing left out."""
    out = []
    for path in sorted((HERE / "content").rglob("*.json")):
        pack = json.loads(path.read_text(encoding="utf-8"))
        pack.pop("keywords", None)
        pack.pop("test_replies", None)
        out.append((path.name, json.dumps(pack, ensure_ascii=False).lower()))
    assert len(out) >= 25
    return out


def test_no_real_organisations_or_threats():
    for name, text in _spoken_strings():
        for word in ["sbi", "hdfc", "icici", "axis", "paytm", "bluedart", "dtdc", "amazon", "flipkart",
                     "police", "पुलिस", "cbi", "court", "अदालत", "arrest", "गिरफ़्त", "गिरफ्तार", "सीबीआई"]:
            assert word not in text, (name, word)
        hit = content.blocked_term(text)
        assert hit is None, (name, hit)


if __name__ == "__main__":
    ok = True
    for name, cases in (("report 20", REPORT_20), ("extra", EXTRA)):
        bad = run(cases)
        print(f"{name}: {len(cases) - len(bad)}/{len(cases)} passed")
        for t, want, got, why in bad:
            ok = False
            print(f"  FAIL  want={want} got={got} ({why})  <- #{cases.index((t, want)) + 1}")
    for fn in (test_masking, test_scenarios_and_keywords, test_no_real_organisations_or_threats):
        try:
            fn()
            print(f"{fn.__name__}: ok")
        except AssertionError as e:
            ok = False
            print(f"{fn.__name__}: FAIL {e!r}")
    sys.exit(0 if ok else 1)
