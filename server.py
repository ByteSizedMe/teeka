"""Teeka server: drill engine + pages. The only process that will talk to Gnani,
so the API key never reaches a browser.

Run:  uvicorn server:app --host 0.0.0.0 --port 8000

Privacy: the text of a reply is classified and thrown away. Only a masked copy
(digits and spoken digit runs shown as bullets) is kept in memory for the live
transcript, and results.json holds outcomes only.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, RedirectResponse
import re
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import classifier
import content
from gnani_client import GnaniClient, GnaniError, wav_duration
from listener import CallListener

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

BOOSTER_DAYS = 3  # after a miss
NEXT_SCAM_DAYS = 7  # after a pass
RING_TIMEOUT_S = 45
MAX_REPLY_CHARS = 500

STAMPS = {"protected": "Protected", "missed": "Needs another dose"}
VERDICT_LABELS = {
    classifier.COMPLYING: "Started to share",
    classifier.REFUSING: "Said no",
    classifier.UNSURE: "Unclear",
}
PHONE_PROBLEMS = {
    "mic": "The phone's microphone did not start, so replies cannot be heard. Allow it and use the https link.",
    "sound": "The phone could not play the voice, so it showed the words instead.",
}


# --------------------------------------------------------------------- settings
@dataclass
class Settings:
    api_key: str = ""
    stt_mode: str = "stream"
    stt_model: str = "prisma-v2.5"
    tts_model: str = "timbre-v2.5"
    family_voice_file: str = ""
    listen_max_s: float = 9.0
    vad_threshold: float = 0.6
    vad_min_silence_ms: int = 600
    warnings: list = field(default_factory=list)

    @property
    def mock(self) -> bool:
        return not self.api_key

    def public(self) -> dict:
        """Safe to send to a browser: never includes the key."""
        return {
            "mock": self.mock,
            "stt_mode": self.stt_mode,
            "stt_model": self.stt_model,
            "tts_model": self.tts_model,
            "listen_max_s": self.listen_max_s,
        }


def load_settings(env=None) -> Settings:
    env = os.environ if env is None else env
    s, warn = Settings(), []

    def pick(name, allowed, default):
        v = (env.get(name) or "").strip().lower() or default
        if v not in allowed:
            warn.append(f"{name} must be one of {', '.join(allowed)}; using {default}.")
            return default
        return v

    def num(name, cast, default, lo, hi):
        raw = (env.get(name) or "").strip()
        if not raw:
            return default
        try:
            v = cast(raw)
        except ValueError:
            v = None
        if v is None or not lo <= v <= hi:
            warn.append(f"{name} must be a number from {lo} to {hi}; using {default}.")
            return default
        return v

    s.api_key = (env.get("GNANI_API_KEY") or "").strip()
    s.stt_mode = pick("STT_MODE", ("stream", "rest"), "stream")
    # Gnani silently ignores unknown model names, so check here.
    s.stt_model = pick("STT_MODEL", ("prisma-v2.5", "prisma-v3.0"), "prisma-v2.5")
    s.tts_model = pick("TTS_MODEL", ("timbre-v2.5", "timbre-v3.0"), "timbre-v2.5")
    s.family_voice_file = (env.get("FAMILY_VOICE_FILE") or "").strip()
    s.listen_max_s = num("LISTEN_MAX_S", float, 9.0, 2.0, 60.0)
    s.vad_threshold = num("VAD_THRESHOLD", float, 0.6, 0.0, 1.0)
    s.vad_min_silence_ms = num("VAD_MIN_SILENCE_MS", int, 600, 100, 2000)
    s.warnings = warn
    return s


# ----------------------------------------------------------------------- content
CONTENT = content.load()
SURPRISE = "surprise"


def lang_pack(lang: str) -> dict:
    return CONTENT["languages"][lang]


def scenarios_in(lang: str) -> list:
    return [s["id"] for s in CONTENT["scenarios"] if s["lang"] == lang]


def voice_for(lang: str, role: str, tts_model: str) -> tuple[str, str]:
    """(voice, language) for a caller/teacher line. Voices belong to one model."""
    v = lang_pack(lang)["voices"]
    return v[tts_model][role], v["language"]


def teacher_text(kind: str, scenario: dict, line_idx: int, with_code_word: bool) -> str:
    t = lang_pack(scenario["lang"])["templates"]
    if kind == "reveal":
        parts = [t["reveal_intro"], scenario["lines"][line_idx]["lesson"], t["reveal_outro"]]
    elif kind == "praise":
        parts = [t["praise"]]
    else:
        parts = [t["wrap"]]
    if with_code_word:
        parts.append(t["code_word"])
    return " ".join(parts)


def flag_public(flag: str, lang: str) -> dict:
    english = CONTENT["languages"].get("en-IN", {}).get("red_flags", {}).get(flag, flag)
    return {"id": flag, "en": english, "local": lang_pack(lang)["red_flags"].get(flag, english)}


# ------------------------------------------------------------------------ engine
class Engine:
    def __init__(self, settings: Settings, results_path: Path, client=None, cache_dir: Path | None = None):
        self.s = settings
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT / "cache" / "tts"
        self.client = client if client is not None else (
            None if settings.mock else GnaniClient(settings.api_key, self.cache_dir))
        self.listener: CallListener | None = None
        self.lines = content.PackLineSource(CONTENT["languages"])
        self.runtime_notices: list[str] = []
        self._listen_task: asyncio.Task | None = None
        self._prewarm_task: asyncio.Task | None = None
        self._listener_task: asyncio.Task | None = None
        self._family: dict | None = None
        self._family_tried = False
        self.results_path = Path(results_path)
        self.phone: WebSocket | None = None
        self.drill: dict | None = None
        self.latencies: list[int] = []  # Prisma latency (ms), numbers only
        self.ring_timeout = RING_TIMEOUT_S
        self._ring_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    # ---- results (outcomes only) ----
    def read_results(self) -> list:
        try:
            data = json.loads(self.results_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _store(self, rec: dict) -> None:
        rows = self.read_results() + [rec]
        tmp = self.results_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.results_path)

    def card(self, name: str | None) -> dict:
        rows = self.read_results()
        if not name and rows:
            name = rows[-1]["name"]
        mine = [r for r in rows if r["name"] == name]
        return {
            "name": name or "",
            "doses": len(mine),
            "protected": sum(r["outcome"] == "protected" for r in mine),
            "history": mine[-8:],
            "next_dose": mine[-1]["next_dose"] if mine else None,
        }

    # ---- phone link ----
    async def _send(self, msg: dict) -> None:
        if self.phone is None:
            return
        try:
            await self.phone.send_json(msg)
        except Exception:
            self.phone = None

    async def attach_phone(self, ws: WebSocket) -> None:
        old, self.phone = self.phone, ws
        if old is not None and old is not ws:
            try:
                await old.close(code=4001)  # tells that page not to reconnect and take the line back
            except Exception:
                pass
        await self._send({"type": "config", **self.s.public()})
        d = self.drill
        if d and d["status"] == "ringing":
            await self._send_ring()

    def detach_phone(self, ws: WebSocket) -> None:
        if self.phone is ws:
            self.phone = None

    async def _send_ring(self) -> None:
        d = self.drill
        sc = CONTENT["by_id"][d["scenario_id"]]
        await self._send({"type": "ring", "caller_id": d["caller_id"], "lang": sc["lang"], "labels": lang_pack(sc["lang"])["phone"]})

    # ---- drill lifecycle ----
    def public_drill(self) -> dict | None:
        d = self.drill
        if not d:
            return None
        sc = CONTENT["by_id"][d["scenario_id"]]
        return {
            "id": d["id"], "name": d["name"], "scenario_id": d["scenario_id"],
            "title": sc["title"], "caller_id": d["caller_id"], "lang": sc["lang"],
            "status": d["status"], "ending": d["ending"], "outcome": d["outcome"],
            "stamp": STAMPS.get(d["outcome"]), "stamp_local": lang_pack(sc["lang"])["stamps"].get(d["outcome"]), "line": max(1, d["turn"]), "turn_limit": sc["turn_limit"],
            "draft_language": lang_pack(sc["lang"])["review"] == "draft",
            "teacher_label": d["teacher_label"],
            "transcript": d["transcript"],
            "flags": [flag_public(f, sc["lang"]) for f in d["flags"]],
        }

    def _rows_for(self, name: str, lang: str) -> list:
        return [r for r in self.read_results() if r["name"] == name and r.get("lang") == lang and r.get("scenario_id") in CONTENT["by_id"]]

    def choose_dose(self, name: str, lang: str, after: str | None = None) -> str:
        """Which scam next, from this person's results: first anything they last fell for,
        then types not yet tried (in pack order, after `after`), then the least recent."""
        everything = scenarios_in(lang)
        pool = everything
        if after in everything:
            cut = everything.index(after) + 1
            pool = (everything[cut:] + everything[:cut - 1]) or [after]
        last: dict[str, str] = {}
        for r in self._rows_for(name, lang):
            last.pop(r["scenario_id"], None)  # re-insert, so the dict runs oldest first
            last[r["scenario_id"]] = r["outcome"]
        for sid, outcome in last.items():
            if outcome == "missed" and sid in pool:
                return sid
        for sid in pool:
            if sid not in last:
                return sid
        return next((sid for sid in last if sid in pool), pool[0])

    async def start(self, scenario_id: str, name: str, lang: str | None = None) -> dict:
        name = (name or "").strip()[:40] or "Family"
        if scenario_id == SURPRISE:
            if not scenarios_in(lang or ""):
                raise HTTPException(404, "Choose a language first")
            scenario_id = self.choose_dose(name, lang)
        if scenario_id not in CONTENT["by_id"]:
            raise HTTPException(404, "Unknown scenario")
        scenario = CONTENT["by_id"][scenario_id]
        seed = sum(r.get("scenario_id") == scenario_id for r in self.read_results() if r["name"] == name)
        plan = self.lines.plan(scenario, seed)
        async with self._lock:
            if self.drill and self.drill["status"] in ("ringing", "active"):
                raise HTTPException(409, "A practice call is already in progress")
            self.drill = {
                "id": uuid.uuid4().hex[:8], "scenario_id": scenario_id,
                "name": name, "seed": seed, "caller_id": plan["caller_id"], "plan": plan["lines"], "turn": 0,
                "status": "ringing", "line_idx": 0, "awaiting": False,
                "transcript": [], "flags": [], "outcome": None, "ending": None,
                "teacher_label": "Teacher",
            }
            if self._ring_task:
                self._ring_task.cancel()
            self._ring_task = asyncio.create_task(self._ring_timeout(self.drill["id"]))
            self.runtime_notices = []
            if self._family is None:
                self._family_tried = False  # retry the family voice each drill until it works
            if self.client is not None:  # live: have the caller's voice ready before they answer
                self._prewarm_task = asyncio.create_task(self._prewarm(scenario["lang"], plan["lines"]))
        await self._send_ring()
        return self.public_drill()

    async def _prewarm(self, lang: str, lines: list) -> None:
        voice, language = voice_for(lang, "caller", self.s.tts_model)
        await asyncio.gather(
            *(self.client.tts(text, voice, language, self.s.tts_model) for text in lines),
            return_exceptions=True,
        )

    def _notice(self, text: str) -> None:
        if text not in self.runtime_notices:
            self.runtime_notices.append(text)

    # ---- family voice (optional) ----
    async def _family_voice(self) -> dict | None:
        """Embedding of the family member's own recording, or None (Timbre teacher is used)."""
        if self._family_tried or not self.s.family_voice_file or self.client is None:
            return self._family
        self._family_tried = True
        path = Path(self.s.family_voice_file)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            self._notice("Family voice file not found. The Timbre teacher will speak instead.")
            return None
        try:
            self._family = await self.client.clone_embedding(path)
        except Exception as e:
            self._notice(f"Family voice unavailable, using the Timbre teacher. {e}")
        return self._family

    # ---- live audio ----
    async def _speak(self, role: str, text: str, lang: str, cloned: dict | None = None) -> dict:
        """Timbre audio for a line (cached). Returns extra message fields, or {} if
        it fails, in which case the phone shows the words instead."""
        if self.client is None:
            return {}
        if cloned is not None:
            try:
                path = await self.client.tts_cloned(text, cloned)
                return {"audio_url": "/api/audio/" + path.name, "duration": wav_duration(path.read_bytes()),
                        "cloned": True}
            except GnaniError as e:
                self._notice(f"Cloned voice failed, using the Timbre teacher. {e}")
        voice, language = voice_for(lang, role, self.s.tts_model)
        try:
            path = await self.client.tts(text, voice, language, self.s.tts_model,
                                         speed=1.0 if role == "caller" else 0.95)
        except GnaniError as e:
            self._notice(f"Voice unavailable, showing text instead. {e}")
            return {}
        return {"audio_url": "/api/audio/" + path.name, "duration": wav_duration(path.read_bytes())}

    async def _start_listener(self, lang: str) -> None:
        if self.client is None or self.listener is not None:
            return
        s, c = self.s, self.client

        async def open_stream():
            return await c.open_stream(lang, s.stt_model, s.vad_threshold, s.vad_min_silence_ms)

        async def transcribe(pcm):
            return await c.transcribe(pcm, lang, s.stt_model)

        self.listener = CallListener(
            mode=s.stt_mode, lang=lang, listen_max_s=s.listen_max_s, min_silence_ms=s.vad_min_silence_ms,
            open_stream=open_stream, transcribe=transcribe, on_notice=self._notice)
        # Prisma connects while the caller's first line plays; _listen waits for it.
        self._listener_task = asyncio.create_task(self.listener.start())

    async def _stop_listening(self) -> None:
        t, self._listen_task = self._listen_task, None
        if t and t is not asyncio.current_task():
            t.cancel()
        lst, self.listener = self.listener, None
        if lst:
            await lst.stop()

    async def _listen(self, listener: CallListener) -> None:
        try:
            if self._listener_task is not None:
                await self._listener_task
                self._listener_task = None
            text, ms = await listener.listen()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._notice(f"Listening failed ({type(e).__name__}).")
            text, ms = "", None
        await self.on_reply(text, ms)

    async def _ring_timeout(self, drill_id: str) -> None:
        await asyncio.sleep(self.ring_timeout)
        d = self.drill
        if d and d["id"] == drill_id and d["status"] == "ringing":
            d["status"], d["ending"] = "done", "unanswered"  # no dose recorded
            await self._send({"type": "end", "ending": "unanswered"})

    async def stop(self) -> None:
        """Family cancels from the dashboard. Nothing is recorded."""
        d = self.drill
        if d and d["status"] in ("ringing", "active"):
            d["status"], d["ending"], d["awaiting"] = "done", "cancelled", False
            if self._ring_task:
                self._ring_task.cancel()
            await self._stop_listening()
            await self._send({"type": "end", "ending": "cancelled"})

    async def handle(self, msg: dict) -> None:
        kind = msg.get("type")
        d = self.drill
        if not d:
            return
        if kind == "answer" and d["status"] == "ringing":
            if self._ring_task:
                self._ring_task.cancel()
            d["status"] = "active"
            await self._start_listener(CONTENT["by_id"][d["scenario_id"]]["lang"])  # connects in the background
            await self._say_caller(0)
        elif kind == "decline" and d["status"] == "ringing":
            await self._finish("declined", outcome="protected", teacher=None)
        elif kind == "hangup" and d["status"] == "active":
            await self._finish("hangup", outcome="protected", teacher=None)
        elif kind == "reply" and d["status"] == "active":
            text = str(msg.get("text", ""))[:MAX_REPLY_CHARS]
            await self.on_reply(text, latency_ms=None)
        elif kind == "silence" and d["status"] == "active":
            await self.on_reply("", latency_ms=None)
        elif kind == "phone_problem" and msg.get("what") in PHONE_PROBLEMS:
            self._notice(PHONE_PROBLEMS[msg["what"]])
        elif kind == "playback_start" and self.listener:
            self.listener.playback_start()
        elif kind == "playback_end" and self.listener:
            self.listener.playback_end()

    async def on_audio(self, frame: bytes) -> None:
        """Mic frame from the phone. Goes through the mic gate to Prisma (live mode only)."""
        if self.listener is not None:
            self.listener.push_frame(frame)

    async def _say_caller(self, idx: int, opener: str = "") -> None:
        d = self.drill
        sc = CONTENT["by_id"][d["scenario_id"]]
        line = sc["lines"][idx]
        text = f"{opener} {d['plan'][idx]}".strip()
        extra = await self._speak("caller", text, sc["lang"])
        if d["status"] != "active":  # hung up while the voice was being made
            return
        if self.listener is not None:  # close the mic gate BEFORE the phone is told to play
            self.listener.gate_close(extra.get("duration", 0.0))
        d["line_idx"], d["awaiting"] = idx, True
        d["turn"] += 1
        d["transcript"].append({"who": "caller", "text": text})
        for f in line["flags"]:
            if f not in d["flags"]:
                d["flags"].append(f)
        await self._send({"type": "caller", "idx": idx, "text": text, **extra})
        if self.listener is not None:
            self._listen_task = asyncio.create_task(self._listen(self.listener))

    async def on_reply(self, text: str, latency_ms: int | None) -> None:
        """Judge one reply. `text` is classified here and then dropped."""
        async with self._lock:
            d = self.drill
            if not d or d["status"] != "active" or not d["awaiting"]:
                return
            d["awaiting"] = False
            sc = CONTENT["by_id"][d["scenario_id"]]
            verdict = classifier.classify(text, sc["lang"])
            cue = classifier.unsure_reason(text, sc["lang"]) if verdict == classifier.UNSURE else None
            shown = classifier.mask_for_display(text, sc["lang"]) if text.strip() else "(no reply)"
            del text  # the raw words are never kept
            if latency_ms is not None:
                self.latencies.append(int(latency_ms))
            d["transcript"].append({"who": "person", "text": shown, "verdict": VERDICT_LABELS[verdict]})
            idx = d["line_idx"]
        if verdict == classifier.COMPLYING:
            await self._finish("missed", outcome="missed", teacher="reveal")
        elif verdict == classifier.REFUSING:
            await self._finish("refused", outcome="protected", teacher="praise")
        elif d["turn"] < sc["turn_limit"] and (idx if cue == "repeat" else idx + 1) < len(sc["lines"]):
            # Unsure: answer what they said, then press on (or say the same thing again).
            opener = self.lines.opener(sc["lang"], cue, d["seed"] + d["turn"])
            await self._say_caller(idx if cue == "repeat" else idx + 1, opener)
        else:
            await self._finish("wrap", outcome="protected", teacher="wrap")

    async def _finish(self, ending: str, outcome: str, teacher: str | None) -> None:
        d = self.drill
        sc = CONTENT["by_id"][d["scenario_id"]]
        d["status"], d["ending"], d["outcome"], d["awaiting"] = "done", ending, outcome, False
        if self._ring_task:
            self._ring_task.cancel()
        lst = self.listener
        if teacher:
            family = await self._family_voice()
            text = teacher_text(teacher, sc, d["line_idx"], with_code_word=family is not None)
            extra = await self._speak("teacher", text, sc["lang"], cloned=family)
            if extra.get("cloned"):
                d["teacher_label"] = "Teacher, in your cloned voice"
            if lst is not None:  # nothing is listened for now, but keep the gate shut during the lesson
                lst.gate_close(extra.get("duration", 0.0))
            d["transcript"].append({"who": "teacher", "text": text})
            await self._send({"type": "teacher", "kind": teacher, "text": text, **extra})
        await self._stop_listening()
        if outcome == "missed":
            nxt = {"kind": "booster", "scenario_id": sc["id"], "days": BOOSTER_DAYS}
        else:
            nxt = {"kind": "next_scam", "scenario_id": self.choose_dose(d["name"], sc["lang"], after=sc["id"]), "days": NEXT_SCAM_DAYS}
        nxt["due"] = (datetime.now().date() + timedelta(days=nxt["days"])).isoformat()
        nxt["title"] = CONTENT["by_id"][nxt["scenario_id"]]["title"]
        nxt["lang"] = sc["lang"]
        self._store({
            "id": d["id"], "ts": datetime.now().isoformat(timespec="seconds"),
            "name": d["name"], "scenario_id": sc["id"], "lang": sc["lang"],
            "ending": ending, "outcome": outcome, "line_reached": d["line_idx"] + 1,
            "flags": list(d["flags"]), "next_dose": nxt,
        })
        await self._send({"type": "end", "ending": ending, "outcome": outcome})

    def median_latency(self) -> int | None:
        return int(statistics.median(self.latencies)) if self.latencies else None

    def notices(self) -> list:
        out = list(self.s.warnings) + list(CONTENT["warnings"]) + list(self.runtime_notices)
        return out


engine = Engine(load_settings(), ROOT / "results.json")

# --------------------------------------------------------------------------- app
app = FastAPI(title="Teeka")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class StartBody(BaseModel):
    scenario_id: str
    name: str = ""
    lang: str | None = None


@app.get("/")
async def index():
    return RedirectResponse("/dashboard")


@app.get("/dashboard")
async def dashboard():
    return FileResponse(ROOT / "static" / "dashboard.html")


@app.get("/call")
async def call_page():
    return FileResponse(ROOT / "static" / "call.html")


_AUDIO_NAME = re.compile(r"^[0-9a-f]{64}\.wav$")


@app.get("/api/audio/{name}")
async def audio(name: str):
    path = engine.cache_dir / name
    if not _AUDIO_NAME.match(name) or not path.is_file():
        raise HTTPException(404, "No such audio")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/scenarios")
async def scenarios():
    return [
        {"id": s["id"], "fraud": s["fraud"], "title": s["title"], "title_local": s["title_local"], "lang": s["lang"],
         "caller_id": s["caller_id"], "practise": s["practise"]}
        for s in CONTENT["scenarios"]
    ]


@app.get("/api/languages")
async def languages():
    return [
        {"lang": code, "name": p["name"], "native_name": p["native_name"], "dir": p.get("dir", "ltr"),
         "draft": p["review"] == "draft", "scams": len(scenarios_in(code))}
        for code, p in CONTENT["languages"].items() if scenarios_in(code)
    ]


@app.get("/api/state")
async def state(name: str | None = None):
    return {
        "config": engine.s.public(),
        "notices": engine.notices(),
        "phone_connected": engine.phone is not None,
        "drill": engine.public_drill(),
        "card": engine.card(name),
        "median_latency_ms": engine.median_latency(),
    }


@app.post("/api/drills")
async def start_drill(body: StartBody):
    return await engine.start(body.scenario_id, body.name, body.lang)


@app.post("/api/drills/stop")
async def stop_drill():
    await engine.stop()
    return {"ok": True}


@app.websocket("/ws/call")
async def ws_call(ws: WebSocket):
    await ws.accept()
    await engine.attach_phone(ws)
    try:
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                await engine.on_audio(m["bytes"])
            elif m.get("text"):
                try:
                    msg = json.loads(m["text"])
                except ValueError:
                    continue
                if isinstance(msg, dict):
                    await engine.handle(msg)
    except Exception:
        pass
    finally:
        engine.detach_phone(ws)


if engine.s.mock:
    print("Teeka: mock mode (no GNANI_API_KEY). Typed replies, no audio.")
else:
    print(f"Teeka: live mode, key set (hidden), STT {engine.s.stt_mode}/{engine.s.stt_model}")
