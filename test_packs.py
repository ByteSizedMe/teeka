"""Offline tests for the content packs and the less-scripted call engine (no network, no key).
Run:  python test_packs.py

Passing here means the packs are valid and the logic works. It says nothing about how a
language sounds or whether Prisma hears it. Use test_replies.py --lang for that.
"""
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from fastapi.testclient import TestClient

import classifier as c
import content
import server

HERE = Path(__file__).parent
# The docs.gnani.ai list of languages supported by both Prisma and Timbre, read 2026-11-09.
GNANI_LANGUAGES = {"hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN"}
FRAUDS = {"bank-kyc", "electricity", "parcel", "upi-refund", "marketplace-qr", "job-fee", "prize", "loan-app", "insurance-bonus",
          "investment-tip", "card-rewards", "sim-kyc", "gas-cutoff", "tech-support", "relative-trouble"}
DIGIT_SETS = {"bn-IN": "৪৭১৯", "gu-IN": "૪૭૧૯", "hi-IN": "४७१९", "mr-IN": "४७१९", "pa-IN": "੪੭੧੯", "kn-IN": "೪೭೧೯", "ml-IN": "൪൭൧൯", "ta-IN": "௪௭௧௯", "te-IN": "౪౭౧౯", "en-IN": "4719"}
DATA = content.load()


def fresh(tmp):
    server.engine = server.Engine(server.load_settings({}), tmp / "results.json")
    server.engine.ring_timeout = 30
    client = TestClient(server.app)
    client.__enter__()
    return client


def until(ws, kind):
    for _ in range(12):
        m = ws.receive_json()
        if m["type"] == kind:
            return m
    raise AssertionError(f"never got {kind}")


def begin(client, ws, sid, name="Kamla", lang=None):
    r = client.post("/api/drills", json={"scenario_id": sid, "name": name, "lang": lang})
    assert r.status_code == 200, r.text
    ring = until(ws, "ring")
    ws.send_json({"type": "answer"})
    return ring, until(ws, "caller"), r.json()


def finish_with(ws, text):
    ws.send_json({"type": "reply", "text": text})
    teacher = until(ws, "teacher")
    return teacher, until(ws, "end")


# ------------------------------------------------------------------- coverage
def test_exact_languages_and_frauds():
    assert set(DATA["languages"]) == GNANI_LANGUAGES
    assert not DATA["warnings"], DATA["warnings"]
    assert {s["fraud"] for s in DATA["scenarios"]} == FRAUDS
    assert len(DATA["scenarios"]) == len(FRAUDS) * len(GNANI_LANGUAGES)


def test_no_pack_claims_native_review():
    assert all(p["review"] == "draft" for p in DATA["languages"].values())


def test_each_call_matches_its_language_pack_voices():
    for lang, p in DATA["languages"].items():
        assert p["voices"]["language"] == lang
        assert set(p["voices"]["timbre-v3.0"].values()) == {"Prakash", "Shivani"}


def test_every_fraud_in_every_language_end_to_end(tmp):
    c1 = fresh(tmp)
    with c1.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        for s in DATA["scenarios"]:
            comply = DIGIT_SETS[s["lang"]]
            ring, first, _ = begin(c1, ws, s["id"], name="Tester " + s["id"])
            assert ring["caller_id"] in s["caller_ids"] and ring["labels"]["ringLabel"], s["id"]
            assert first["text"] == s["lines"][0]["text"], s["id"]
            t, end = finish_with(ws, comply)
            assert t["kind"] == "reveal" and s["lines"][0]["lesson"] in t["text"] and end["outcome"] == "missed", s["id"]
            d = c1.get("/api/state").json()["drill"]
            assert "•" in [x for x in d["transcript"] if x["who"] == "person"][0]["text"], s["id"]
            assert not any(ch.isdigit() for x in d["transcript"] if x["who"] == "person" for ch in x["text"]), s["id"]
    rows = json.loads((tmp / "results.json").read_text(encoding="utf-8"))
    assert len(rows) == len(DATA["scenarios"])


def test_test_replies_classify_in_their_language():
    for lang, p in DATA["languages"].items():
        assert len(p["test_replies"]) >= 6, lang
        for text, want in p["test_replies"]:
            assert c.classify(text, lang) == want, (lang, text)


# ----------------------------------------------------------------- validation
def _copy(tmp):
    root = tmp / "content"
    shutil.copytree(HERE / "content", root)
    return root


def _edit(path, fn):
    p = json.loads(path.read_text(encoding="utf-8"))
    fn(p)
    path.write_text(json.dumps(p, ensure_ascii=False), encoding="utf-8")


def test_broken_packs_are_disabled_not_fatal(tmp):
    root = _copy(tmp)
    (root / "scams" / "prize.json").write_text("{not json", encoding="utf-8")
    (root / "languages" / "gu-IN.json").write_text('{"lang": "gu-IN"}', encoding="utf-8")
    d = content.load(root)
    assert "gu-IN" not in d["languages"] and not any(s["fraud"] == "prize" for s in d["scenarios"])
    assert any("prize.json" in w for w in d["warnings"]) and any("gu-IN.json" in w for w in d["warnings"])
    assert "bank-kyc-hi" in d["by_id"]


def test_blocked_and_bad_lines_disable_only_that_language(tmp):
    cases = [
        ("a police threat", lambda b: b.__setitem__("variants", ["मैं पुलिस से बोल रहा हूँ।"])),
        ("arrest in Tamil script", lambda b: b.__setitem__("variants", ["உங்களை கைது செய்வோம்."])),
        ("digits", lambda b: b.__setitem__("variants", ["कॉल करें 9876543210"])),
        ("a link", lambda b: b.__setitem__("variants", ["open https now please"])),
        ("a real bank", lambda b: b.__setitem__("variants", ["this is hdfc"])),
        ("too long", lambda b: b.__setitem__("variants", ["ओटीपी बताइए। " * 40])),
    ]
    for why, change in cases:
        root = _copy(tmp / why.replace(" ", "_"))
        _edit(root / "scams" / "bank-kyc.json", lambda p: change(p["languages"]["hi-IN"]["beats"][0]))
        d = content.load(root)
        assert "bank-kyc-hi" not in d["by_id"] and "bank-kyc-en" in d["by_id"], why
        assert any("bank-kyc is disabled in Hindi" in w for w in d["warnings"]), why


def test_pack_rules(tmp):
    root = _copy(tmp)
    _edit(root / "scams" / "gas-cutoff.json", lambda p: p.__setitem__("turn_limit", 9))
    _edit(root / "scams" / "tech-support.json", lambda p: p["beats"][0].__setitem__("flags", ["no_such_flag"]))
    _edit(root / "scams" / "sim-kyc.json", lambda p: p.__setitem__("caller_ids", ["SBI Help"]))
    _edit(root / "languages" / "pa-IN.json", lambda p: p.__setitem__("review", "looks_fine"))
    d = content.load(root)
    assert not any(s["fraud"] in ("gas-cutoff", "tech-support", "sim-kyc") for s in d["scenarios"])
    assert "pa-IN" not in d["languages"] and len(d["warnings"]) >= 4


def test_new_language_and_fraud_need_only_data(tmp):
    root = _copy(tmp)
    p = json.loads((root / "languages" / "en-IN.json").read_text(encoding="utf-8"))
    p.update(lang="xx-IN", name="Testlang", native_name="Testlang", short="xx")
    (root / "languages" / "xx-IN.json").write_text(json.dumps(p), encoding="utf-8")
    scam = json.loads((root / "scams" / "bank-kyc.json").read_text(encoding="utf-8"))
    scam["id"] = "new-fraud"
    scam["languages"]["xx-IN"] = scam["languages"]["en-IN"]
    (root / "scams" / "new-fraud.json").write_text(json.dumps(scam, ensure_ascii=False), encoding="utf-8")
    d = content.load(root)
    assert not d["warnings"], d["warnings"]
    assert "new-fraud-xx" in d["by_id"] and "new-fraud-hi" in d["by_id"] and "xx-IN" in d["languages"]


# ----------------------------------------------------------------- classifier
def test_negation_direction_is_per_language():
    assert c.classify_detail("আমি বলছি না", "bn-IN")["reason"] == "last: negated: বলছি"     # after the verb
    assert c.classify_detail("मी सांगतो नाही", "mr-IN")["reason"] == "last: negated: सांगतो"
    assert c.classify("இல்லை, சொல்கிறேன்", "ta-IN") == c.COMPLYING                          # "no, I will tell": a before-negator does not flip here
    assert c.classify("नहीं, बताती हूँ", "hi-IN") == c.REFUSING                                # Hindi negates before the verb
    assert c.classify_detail("નહીં કહું છું", "gu-IN")["label"] == c.REFUSING                  # Gujarati allows both
    assert c.classify_detail("કહું છું નહીં", "gu-IN")["reason"].startswith("last: negated")


def test_only_call_language_plus_english():
    assert c.classify("ओटीपी आया है", "ta-IN") == c.UNSURE    # Hindi phrase on a Tamil call
    assert c.classify("ஓடிபி வந்தது", "hi-IN") == c.UNSURE    # Tamil phrase on a Hindi call
    assert c.classify("the otp is here", "ta-IN") == c.COMPLYING  # English always counts
    assert c.classify("hang up", "kn-IN") == c.REFUSING


def test_digits_in_every_script_and_mixed():
    for lang, digits in DIGIT_SETS.items():
        assert c.classify(digits, lang) == c.COMPLYING, lang
        assert c.classify(" ".join(digits), lang) == c.COMPLYING, lang
        assert c.mask_digits(digits) == "••••", lang
        words = DATA["languages"][lang]["keywords"]["number_words"]
        spoken = f"4 7 1 {words['9'][0]}"
        assert c.classify(spoken, lang) == c.COMPLYING, lang
        assert c.mask_for_display(spoken, lang) == "• • • •", lang


def test_one_number_word_is_not_masked():
    assert c.mask_for_display("एक मिनट रुकिए", "hi-IN") == "एक मिनट रुकिए"
    assert c.mask_for_display("ஒரு நிமிடம்", "ta-IN") == "ஒரு நிமிடம்"


def test_confusion_is_not_a_refusal():
    for lang, text, cue in [("hi-IN", "समझ नहीं आया", "confused"), ("hi-IN", "अब क्या करूँ", "asks_what"), ("hi-IN", "एक मिनट रुकिए", "wants_time"), ("hi-IN", "फिर से बोलिए", "repeat"),
                            ("en-IN", "I don't understand", "confused"), ("ta-IN", "புரியவில்லை", "confused"), ("bn-IN", "আবার বলুন", "repeat"), ("kn-IN", "ಸ್ವಲ್ಪ ಸಮಯ", "wants_time")]:
        assert c.classify(text, lang) == c.UNSURE, (lang, text)
        assert c.unsure_reason(text, lang) == cue, (lang, text)


# ----------------------------------------------------------------- variety
def test_variants_rotate_and_stay_bounded():
    src = content.PackLineSource(DATA["languages"])
    hi = DATA["by_id"]["bank-kyc-hi"]
    first, second, third = src.plan(hi, 0), src.plan(hi, 1), src.plan(hi, 2)
    assert first["lines"] != second["lines"] and first["caller_id"] == "ABC Bank" and second["caller_id"] == "XYZ Bank"
    assert first == third  # small pools: the audio to cache stays small
    ta = DATA["by_id"]["bank-kyc-ta"]
    assert src.plan(ta, 0)["lines"] != src.plan(ta, 1)["lines"]  # single sentence, varied by stand-alone sentences
    for s in DATA["scenarios"]:
        for seed in range(6):
            plan = src.plan(s, seed)
            assert not any("{" in t for t in plan["lines"]), s["id"]
            assert all(len(t) <= content.MAX_LINE_CHARS for t in plan["lines"]), s["id"]
            assert all(content.blocked_term(t) is None for t in plan["lines"]), s["id"]


def test_second_call_to_the_same_person_differs(tmp):
    c1 = fresh(tmp)
    with c1.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        ring1, first1, _ = begin(c1, ws, "bank-kyc-hi")
        finish_with(ws, "मैं फ़ोन रख रही हूँ")
        ring2, first2, _ = begin(c1, ws, "bank-kyc-hi")
        assert ring1["caller_id"] != ring2["caller_id"] and first1["text"] != first2["text"]


def test_caller_answers_what_was_said(tmp):
    c1 = fresh(tmp)
    with c1.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        begin(c1, ws, "bank-kyc-hi")
        ws.send_json({"type": "reply", "text": "समझ नहीं आया"})
        m = until(ws, "caller")
        assert m["idx"] == 1 and any(o in m["text"] for o in DATA["languages"]["hi-IN"]["branches"]["confused"])
        ws.send_json({"type": "reply", "text": "एक मिनट रुकिए"})
        m = until(ws, "caller")
        assert m["idx"] == 2 and any(o in m["text"] for o in DATA["languages"]["hi-IN"]["branches"]["wants_time"])
        ws.send_json({"type": "reply", "text": "समझ नहीं आया"})
        assert until(ws, "teacher")["kind"] == "wrap"  # three turns used


def test_repeat_says_the_same_beat_and_counts_a_turn(tmp):
    c1 = fresh(tmp)
    with c1.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        begin(c1, ws, "bank-kyc-hi")
        for _ in range(2):
            ws.send_json({"type": "reply", "text": "फिर से बोलिए"})
            assert until(ws, "caller")["idx"] == 0
        ws.send_json({"type": "reply", "text": "फिर से बोलिए"})
        assert until(ws, "teacher")["kind"] == "wrap"


def test_complying_still_stops_after_a_branch(tmp):
    c1 = fresh(tmp)
    with c1.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        begin(c1, ws, "parcel-hi")
        ws.send_json({"type": "reply", "text": "समझ नहीं आया"})
        until(ws, "caller")
        t, end = finish_with(ws, "हाँ, पिन डाल रही हूँ")
        assert t["kind"] == "reveal" and end["outcome"] == "missed"
        assert DATA["by_id"]["parcel-hi"]["lines"][1]["lesson"] in t["text"]


def test_line_source_is_one_small_interface(tmp):
    class Fixed(content.PackLineSource):
        def plan(self, scenario, seed):
            return {"caller_id": "Test Caller", "lines": [f"line {i}" for i in range(len(scenario["lines"]))]}
    c1 = fresh(tmp)
    server.engine.lines = Fixed(DATA["languages"])
    with c1.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        ring, first, _ = begin(c1, ws, "bank-kyc-hi")
        assert ring["caller_id"] == "Test Caller" and first["text"] == "line 0"


# ----------------------------------------------------------------- next dose
def test_surprise_and_next_dose_use_results(tmp):
    c1 = fresh(tmp)
    hi = [s["id"] for s in DATA["scenarios"] if s["lang"] == "hi-IN"]
    with c1.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        _, _, d = begin(c1, ws, "surprise", lang="hi-IN")
        assert d["scenario_id"] == hi[0]  # nothing tried yet: the first
        finish_with(ws, "हाँ, ओटीपी आया है")
        card = c1.get("/api/state", params={"name": "Kamla"}).json()["card"]
        assert card["next_dose"]["kind"] == "booster" and card["next_dose"]["scenario_id"] == hi[0] and card["next_dose"]["days"] == 3
        assert card["next_dose"]["lang"] == "hi-IN"
        _, _, d = begin(c1, ws, "surprise", lang="hi-IN")
        assert d["scenario_id"] == hi[0]  # boost what they fell for
        finish_with(ws, "मैं फ़ोन रख रही हूँ")
        card = c1.get("/api/state", params={"name": "Kamla"}).json()["card"]
        assert card["next_dose"]["kind"] == "next_scam" and card["next_dose"]["scenario_id"] == hi[1] and card["next_dose"]["days"] == 7
        _, _, d = begin(c1, ws, "surprise", lang="hi-IN")
        assert d["scenario_id"] == hi[1]  # then what they have not tried
        finish_with(ws, "मैं फ़ोन रख रही हूँ")
        _, _, other = begin(c1, ws, "surprise", name="Ravi", lang="ta-IN")
        assert other["scenario_id"].endswith("-ta")  # results are per person and language
        finish_with(ws, "நான் போனை வைக்கிறேன்")
        assert c1.post("/api/drills", json={"scenario_id": "surprise", "name": "x", "lang": "zz-IN"}).status_code == 404


def test_public_api_is_built_from_packs(tmp):
    c1 = fresh(tmp)
    langs = c1.get("/api/languages").json()
    assert {l["lang"] for l in langs} == GNANI_LANGUAGES and all(l["draft"] and l["scams"] == len(FRAUDS) for l in langs)
    rows = c1.get("/api/scenarios").json()
    assert len(rows) == len(DATA["scenarios"]) and all(r["practise"] and r["fraud"] for r in rows)
    with c1.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        begin(c1, ws, "parcel-te")
        d = c1.get("/api/state").json()["drill"]
        assert d["draft_language"] and d["lang"] == "te-IN" and d["turn_limit"] == 3
        assert all(f["local"] != f["en"] for f in d["flags"])


def test_pages_hold_no_fixed_language_text():
    """Everything in another language comes from a pack. Only the brand name is written in."""
    for name in ("call.html", "dashboard.html", "app.css"):
        text = (HERE / "static" / name).read_text(encoding="utf-8").replace("टीका", "")
        found = sorted({ch for ch in text if "\u0900" <= ch <= "\u0d7f"})
        assert not found, (name, found)


def test_stamp_is_in_the_call_language(tmp):
    c1 = fresh(tmp)
    with c1.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        begin(c1, ws, "bank-kyc-ml")
        finish_with(ws, "മനസ്സിലായില്ല 4 7 1")
        d = c1.get("/api/state").json()["drill"]
        assert d["stamp"] == "Needs another dose" and d["stamp_local"] == DATA["languages"]["ml-IN"]["stamps"]["missed"]


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    bad = 0
    for fn in TESTS:
        try:
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"ok    {fn.__name__}")
        except Exception:
            bad += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
    sys.exit(1 if bad else 0)
