"""End-to-end tests of the drill engine in mock mode (no network, no key).
Run:  python test_server.py
"""
import json
import sys
import tempfile
import traceback
from pathlib import Path

from fastapi.testclient import TestClient

import server


def fresh(tmp: Path):
    server.engine = server.Engine(server.load_settings({}), tmp / "results.json")
    server.engine.ring_timeout = 0.3
    server.engine.consent_required = False  # consent has its own test in test_packs.py
    client = TestClient(server.app)
    client.__enter__()  # one event loop for the whole test, like uvicorn
    return client


def recv_until(ws, kind):
    for _ in range(10):
        m = ws.receive_json()
        if m["type"] == kind:
            return m
    raise AssertionError(f"never got {kind}")


def start(client, ws, sid="bank-kyc-hi", name="Kamla"):
    r = client.post("/api/drills", json={"scenario_id": sid, "name": name})
    assert r.status_code == 200, r.text
    return recv_until(ws, "ring")


def test_settings():
    s = server.load_settings({})
    assert s.mock and s.stt_mode == "stream" and s.vad_threshold == 0.6 and s.vad_min_silence_ms == 600
    s = server.load_settings({"GNANI_API_KEY": "SECRETKEY123", "STT_MODE": "bogus", "STT_MODEL": "prisma-v9", "VAD_THRESHOLD": "7"})
    assert not s.mock and s.stt_mode == "stream" and s.stt_model == "prisma-v2.5" and len(s.warnings) == 3
    assert "SECRETKEY123" not in json.dumps(s.public()) and "api_key" not in s.public()


def test_complying_path(tmp):
    c = fresh(tmp)
    with c.websocket_connect("/ws/call") as ws:
        assert ws.receive_json()["mock"] is True
        assert c.get("/api/state").json()["config"]["mock"] is True
        assert start(c, ws)["caller_id"] == "ABC Bank"
        ws.send_json({"type": "answer"})
        m = recv_until(ws, "caller")
        assert m["idx"] == 0
        ws.send_json({"type": "reply", "text": "हाँ, चार सात एक नौ बताती हूँ"})
        t = recv_until(ws, "teacher")
        assert t["kind"] == "reveal" and "अभ्यास कॉल" in t["text"]
        assert server.CONTENT["by_id"]["bank-kyc-hi"]["lines"][0]["lesson"] in t["text"]
        assert recv_until(ws, "end")["outcome"] == "missed"
    d = c.get("/api/state").json()["drill"]
    assert d["stamp"] == "Needs another dose" and d["status"] == "done"
    shown = [x for x in d["transcript"] if x["who"] == "person"][0]
    assert "•" in shown["text"] and "चार" not in shown["text"] and shown["verdict"] == "Started to share"
    assert {f["id"] for f in d["flags"]} == {"unsolicited", "urgency"}
    rows = json.loads((tmp / "results.json").read_text(encoding="utf-8"))
    assert len(rows) == 1 and rows[0]["outcome"] == "missed" and rows[0]["next_dose"]["kind"] == "booster"
    assert rows[0]["next_dose"]["days"] == 3 and rows[0]["next_dose"]["scenario_id"] == "bank-kyc-hi"
    raw = (tmp / "results.json").read_text(encoding="utf-8")
    for leaked in ("चार", "बताती", "reply", "text", "transcript"):
        assert leaked not in raw, leaked


def test_refusing_path(tmp):
    c = fresh(tmp)
    with c.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        start(c, ws, "electricity-hi")
        ws.send_json({"type": "answer"}); recv_until(ws, "caller")
        ws.send_json({"type": "reply", "text": "मैं फ़ोन रख रही हूँ"})
        assert recv_until(ws, "teacher")["kind"] == "praise"
        assert recv_until(ws, "end")["outcome"] == "protected"
    row = json.loads((tmp / "results.json").read_text(encoding="utf-8"))[0]
    assert row["next_dose"] == {**row["next_dose"], "kind": "next_scam", "days": 7, "scenario_id": "parcel-hi"}


def test_unsure_escalates_then_wraps(tmp):
    c = fresh(tmp)
    with c.websocket_connect("/ws/call") as ws:
        ws.receive_json()
        start(c, ws)
        ws.send_json({"type": "answer"})
        for i in range(3):
            assert recv_until(ws, "caller")["idx"] == i
            ws.send_json({"type": "reply", "text": "अरे बाप रे, अब क्या करूँ ?"})
        assert recv_until(ws, "teacher")["kind"] == "wrap"
        assert recv_until(ws, "end")["outcome"] == "protected"
    d = c.get("/api/state").json()["drill"]
    assert d["ending"] == "wrap" and {f["id"] for f in d["flags"]} >= {"asks_otp", "threat"}


def test_compliance_on_last_line_uses_that_lines_lesson(tmp):
    c = fresh(tmp)
    with c.websocket_connect("/ws/call") as ws:
        ws.receive_json(); start(c, ws, "parcel-hi")
        ws.send_json({"type": "answer"})
        for _ in range(2):
            recv_until(ws, "caller"); ws.send_json({"type": "silence"})
        recv_until(ws, "caller")
        ws.send_json({"type": "reply", "text": "हाँ, पिन डाल रही हूँ"})
        t = recv_until(ws, "teacher")
        assert server.CONTENT["by_id"]["parcel-hi"]["lines"][2]["lesson"] in t["text"]


def test_hangup_decline_and_unanswered(tmp):
    c = fresh(tmp)
    with c.websocket_connect("/ws/call") as ws:
        ws.receive_json(); start(c, ws)
        ws.send_json({"type": "decline"})
        assert recv_until(ws, "end")["outcome"] == "protected"
        start(c, ws, "parcel-hi")
        ws.send_json({"type": "answer"}); recv_until(ws, "caller")
        ws.send_json({"type": "hangup"})
        assert recv_until(ws, "end")["ending"] == "hangup"
        start(c, ws, "bank-kyc-en")
        assert recv_until(ws, "end")["ending"] == "unanswered"  # ring timeout
    rows = json.loads((tmp / "results.json").read_text(encoding="utf-8"))
    assert [r["ending"] for r in rows] == ["declined", "hangup"]  # unanswered is not a dose
    card = c.get("/api/state", params={"name": "Kamla"}).json()["card"]
    assert card["doses"] == 2 and card["protected"] == 2


def test_busy_and_stop_and_unknown(tmp):
    c = fresh(tmp)
    with c.websocket_connect("/ws/call") as ws:
        ws.receive_json(); start(c, ws)
        assert c.post("/api/drills", json={"scenario_id": "parcel-hi"}).status_code == 409
        assert c.post("/api/drills/stop").status_code == 200
        assert recv_until(ws, "end")["ending"] == "cancelled"
        assert c.post("/api/drills", json={"scenario_id": "nope"}).status_code == 404
    assert not (tmp / "results.json").exists()


def test_pages(tmp):
    c = fresh(tmp)
    for p in ("/dashboard", "/call", "/static/mic-worklet.js", "/static/app.css"):
        assert c.get(p).status_code == 200, p
    assert "Mock mode" in c.get("/dashboard").text or "modePill" in c.get("/dashboard").text
    assert "GNANI" not in c.get("/call").text + c.get("/dashboard").text


class FakeGnani:
    """Stands in for GnaniClient: no network. Records which voice was used."""
    def __init__(self, tmp, clone_fails=False):
        import gnani_client as g
        self.wav = tmp / "a.wav"
        self.wav.write_bytes(g.pcm_to_wav(bytes(48000), 24000))
        self.used, self.clone_fails, self.tts_calls = [], clone_fails, 0
        self.cache_dir = tmp

    async def tts(self, text, *a, **k):
        self.used.append("timbre"); return self.wav

    async def tts_cloned(self, text, emb, **k):
        import gnani_client as g
        if self.clone_fails:
            raise g.GnaniError("Cloned voice failed (HTTP 400: test).")
        self.used.append("cloned"); return self.wav

    async def clone_embedding(self, path):
        return {"embedding": "abc", "shape": [1, 768], "dtype": "torch.bfloat16"}

    async def open_stream(self, *a):
        import gnani_client as g
        raise g.GnaniError("no network in tests")


def test_embedding_shapes():
    import gnani_client as g
    flat = {"embedding": "e1", "shape": [1, 768], "dtype": "torch.bfloat16"}
    wrapped = {"success": True, "message": "ok", "data": {"voice_clone_embedding": flat}}
    assert g.extract_embedding(flat) == g.extract_embedding(wrapped) == flat
    for bad in ({}, {"success": True}, {"data": {}}, {"embedding": ""}):
        try:
            g.extract_embedding(bad); raise AssertionError("accepted a bad body")
        except g.GnaniError:
            pass


def _family_drill(tmp, clone_fails, voice_file):
    s = server.load_settings({"GNANI_API_KEY": "x", "FAMILY_VOICE_FILE": voice_file})
    fake = FakeGnani(tmp, clone_fails)
    server.engine = server.Engine(s, tmp / "results.json", client=fake, cache_dir=tmp)
    server.engine.consent_required = False
    c = TestClient(server.app); c.__enter__()
    with c.websocket_connect("/ws/call") as ws:
        ws.receive_json(); start(c, ws)
        ws.send_json({"type": "answer"}); recv_until(ws, "caller")
        ws.send_json({"type": "reply", "text": "हाँ, ओटीपी आया है"})
        t = recv_until(ws, "teacher")
    return c, fake, t


def test_family_voice_used_with_code_word(tmp):
    ref = tmp / "me.wav"; ref.write_bytes(b"x")
    c, fake, t = _family_drill(tmp, False, str(ref))
    assert t["cloned"] is True and "गुप्त शब्द" in t["text"] and "cloned" in fake.used
    assert c.get("/api/state").json()["drill"]["teacher_label"] == "Teacher, in your cloned voice"


def test_family_voice_failure_falls_back_with_notice(tmp):
    ref = tmp / "me.wav"; ref.write_bytes(b"x")
    c, fake, t = _family_drill(tmp, True, str(ref))
    st = c.get("/api/state").json()
    assert "cloned" not in t and fake.used[-1] == "timbre" and st["drill"]["teacher_label"] == "Teacher"
    assert any("Cloned voice failed" in n for n in st["notices"])


def test_family_voice_missing_file_falls_back(tmp):
    c, fake, t = _family_drill(tmp, False, str(tmp / "nope.wav"))
    st = c.get("/api/state").json()
    assert "cloned" not in t and "गुप्त शब्द" not in t["text"]
    assert any("Family voice file not found" in n for n in st["notices"])


TESTS = [test_settings, test_complying_path, test_refusing_path, test_unsure_escalates_then_wraps,
         test_compliance_on_last_line_uses_that_lines_lesson, test_hangup_decline_and_unanswered,
         test_busy_and_stop_and_unknown, test_pages, test_embedding_shapes,
         test_family_voice_used_with_code_word, test_family_voice_failure_falls_back_with_notice,
         test_family_voice_missing_file_falls_back]

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
