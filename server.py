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
import csv
import io
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, RedirectResponse, Response
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
LLM_TIMEOUT_S = 2.0
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
    grace_s: float = 0.8
    gate_tail_s: float = 0.5
    evon_url: str = ""
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
    s.grace_s = num("GRACE_S", float, 0.8, 0.1, 3.0)
    s.gate_tail_s = num("GATE_TAIL_S", float, 0.5, 0.1, 2.0)
    s.evon_url = (env.get("EVON_URL") or "").strip()
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
    elif kind == "comfort":
        parts = [t["comfort"]]
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
        self.lines = (content.EvonLineSource(CONTENT["languages"], settings.evon_url) if settings.evon_url
                      else content.PackLineSource(CONTENT["languages"]))
        self.turn_ms: list[int] = []  # end of the person's speech to the caller's first sound
        self._timing_turn = False
        self.runtime_notices: list[str] = []
        self._listen_task: asyncio.Task | None = None
        self._prewarm_task: asyncio.Task | None = None
        self._listener_task: asyncio.Task | None = None
        self._family: dict | None = None
        self._family_tried = False
        self.results_path = Path(results_path)
        self.people_path = self.results_path.with_name("people.json")
        self.consent_required = True
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

    # ---- family profiles: name, language, consent, and whether a code word exists ----
    def read_people(self) -> dict:
        try:
            data = json.loads(self.people_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_person(self, name: str, lang: str, consent: bool, code_word_set: bool) -> dict:
        people = self.read_people()
        people[name] = {"name": name, "lang": lang, "consent": consent, "code_word_set": code_word_set}
        tmp = self.people_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(people, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.people_path)
        return people[name]

    def report(self, name: str | None = None) -> dict:
        """Everything here is counted from stored outcomes. Nothing is estimated."""
        rows = [r for r in self.read_results() if not name or r["name"] == name]
        scored = [r for r in rows if r["outcome"] in STAMPS]
        recent = scored[-10:]
        protected = sum(r["outcome"] == "protected" for r in scored)
        seen, missed, doses, per_person = {}, {}, {}, {}
        for r in scored:
            for f in r.get("flags", []):
                seen[f] = seen.get(f, 0) + 1
                missed[f] = missed.get(f, 0) + (r["outcome"] == "missed")
            n = per_person[r["name"]] = per_person.get(r["name"], 0) + 1
            total, passed = doses.get(n, (0, 0))
            doses[n] = (total + 1, passed + (r["outcome"] == "protected"))
        english = CONTENT["languages"].get("en-IN", {}).get("red_flags", {})
        pct = lambda a, b: round(100 * a / b)  # noqa: E731
        refusals = [r["decide_s"] for r in scored if r.get("ending") == "refused" and r.get("decide_s") is not None]
        cohort = {}
        for r in scored:
            c = cohort.setdefault(r["lang"], {"people": set(), "doses": 0, "protected": 0})
            c["people"].add(r["name"]); c["doses"] += 1; c["protected"] += r["outcome"] == "protected"
        return {
            "doses": len(scored), "protected": protected,
            "immunity": pct(sum(r["outcome"] == "protected" for r in recent), len(recent)) if recent else None,
            "flags": [{"id": f, "en": english.get(f, f), "seen": n, "missed": missed[f], "pct": pct(missed[f], n)} for f, n in seen.items()],
            "by_dose": [{"dose": d, "total": t, "passed": p, "pct": pct(p, t)} for d, (t, p) in sorted(doses.items())],
            "median_refuse_s": round(statistics.median(refusals), 1) if refusals else None,
            "cohort": [{"lang": k, "name": CONTENT["languages"].get(k, {}).get("name", k), "people": len(v["people"]), "doses": v["doses"], "pct": pct(v["protected"], v["doses"])} for k, v in cohort.items()],
        }

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
        if self.consent_required and not self.read_people().get(name, {}).get("consent"):
            raise HTTPException(403, "Please confirm this person agreed in advance to practice calls.")
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
                "name": name, "seed": seed, "caller_id": plan["caller_id"], "plan": plan["lines"], "turn": 0, "said": set(),
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
                self._prewarm_task = asyncio.create_task(self._prewarm(scenario["lang"], self._lines_to_warm(scenario, plan, seed)))
        await self._send_ring()
        return self.public_drill()

    def _lines_to_warm(self, scenario: dict, plan: dict, seed: int) -> list:
        """The caller's beats, the fillers, and the first answer to the questions people ask most."""
        lang_fillers = [self.lines.filler(scenario["lang"], n) for n in range(2)]
        common = [self.lines.answer(scenario, i, seed, set()) for i in ("where_message", "which_app", "how_much", "how_to_pay")]
        return list(dict.fromkeys(plan["lines"] + lang_fillers + common))

    async def _prewarm(self, lang: str, lines: list) -> None:
        voice, language = voice_for(lang, "caller", self.s.tts_model)
        paths = await asyncio.gather(
            *(self.client.tts(text, voice, language, self.s.tts_model) for text in lines),
            return_exceptions=True,
        )
        urls = ["/api/audio/" + p.name for p in paths if isinstance(p, Path)]
        if urls:
            await self._send({"type": "prefetch", "urls": urls})  # so the phone has them before they are needed

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

    async def _play_filler_if_slow(self, d: dict, sc: dict, parts: list) -> None:
        """If this line's audio still has to be made, cover the wait with a short cached line."""
        c = self.client
        if c is None or not hasattr(c, "is_cached"):
            return
        voice, language = voice_for(sc["lang"], "caller", self.s.tts_model)
        if all(c.is_cached(p, voice, language, self.s.tts_model) for p in parts):
            return
        extra = await self._speak("caller", self.lines.filler(sc["lang"], d["turn"]), sc["lang"])
        if extra and d["status"] == "active":
            if self.listener is not None:
                self.listener.gate_close(extra.get("duration", 0.0))
            await self._send({"type": "filler", **extra})

    async def _improvise(self, sc: dict, masked: str, d: dict) -> str | None:
        """Only for replies the rules could not place. Any failure means the pack line is used."""
        try:
            line = await asyncio.wait_for(self.lines.improvise(sc, masked, d["said"]), LLM_TIMEOUT_S)
        except Exception:
            return None
        return line if content.line_is_safe(line, sc["lang"], CONTENT["languages"]) else None

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
        self.listener.grace_s, self.listener.gate_tail_s = s.grace_s, s.gate_tail_s
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
            d["answered_at"] = time.monotonic()
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
            heard = self.listener.last_voice
            if self._timing_turn and heard is not None:
                self.turn_ms.append(int((self.listener.clock() - heard) * 1000))
            self._timing_turn = False
        elif kind == "playback_end" and self.listener:
            self.listener.playback_end()

    async def on_audio(self, frame: bytes) -> None:
        """Mic frame from the phone. Goes through the mic gate to Prisma (live mode only)."""
        if self.listener is not None:
            self.listener.push_frame(frame)

    async def _say_caller(self, idx: int, opener: str = "") -> None:
        d = self.drill
        sc = CONTENT["by_id"][d["scenario_id"]]
        if idx is not None:
            for f in sc["lines"][idx]["flags"]:
                if f not in d["flags"]:
                    d["flags"].append(f)
        body = d["plan"][idx] if idx is not None else ""
        text = f"{opener} {body}".strip()
        parts = [p for p in (opener, body) if p]  # spoken separately, so answers and beats each cache once
        await self._play_filler_if_slow(d, sc, parts)
        spoken = [await self._speak("caller", p, sc["lang"]) for p in parts]
        extra = dict(spoken[0])
        if extra and len(spoken) > 1:
            extra["more"] = [x["audio_url"] for x in spoken[1:] if x]
            extra["duration"] = sum(x.get("duration", 0.0) for x in spoken)
        if d["status"] != "active":  # hung up while the voice was being made
            return
        if self.listener is not None:  # close the mic gate BEFORE the phone is told to play
            self.listener.gate_close(extra.get("duration", 0.0))
        d["line_idx"] = d["line_idx"] if idx is None else idx
        d["awaiting"] = True
        d["said"].add(text)
        if opener:
            d["said"].add(opener)
        d["turn"] += 1
        d["transcript"].append({"who": "caller", "text": text})
        await self._send({"type": "caller", "idx": d["line_idx"], "text": text, **extra})
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
            distressed = classifier.is_distress(text, sc["lang"])
            verdict = classifier.classify(text, sc["lang"])
            cue = classifier.unsure_reason(text, sc["lang"]) if verdict == classifier.UNSURE else None
            shown = classifier.mask_for_display(text, sc["lang"]) if text.strip() else "(no reply)"
            not_blank = bool(text.strip())
            self._timing_turn = not_blank
            del text  # the raw words are never kept
            if latency_ms is not None:
                self.latencies.append(int(latency_ms))
            d["transcript"].append({"who": "person", "text": shown, "verdict": VERDICT_LABELS[verdict]})
            idx = d["line_idx"]
        if distressed:
            await self._finish("distress", outcome="paused", teacher="comfort")
        elif verdict == classifier.COMPLYING:
            await self._finish("missed", outcome="missed", teacher="reveal")
        elif verdict == classifier.REFUSING:
            await self._finish("refused", outcome="protected", teacher="praise")
        elif cue in classifier.INTENTS and d["turn"] < sc["turn_limit"]:
            # A question: answer it from this scam's facts, then keep pressing.
            answer = self.lines.answer(sc, cue, d["seed"], d["said"])
            d["said"].add(answer)
            if idx + 1 < len(sc["lines"]):
                await self._say_caller(idx + 1, answer)
            else:
                press = self.lines.press(sc["lang"], d["seed"] + d["turn"], d["said"])
                await self._say_caller(None, f"{answer} {press}")
        elif d["turn"] < sc["turn_limit"] and (idx if cue == "repeat" else idx + 1) < len(sc["lines"]):
            # Unsure: answer what they said, then press on (or say the same thing again).
            opener = self.lines.opener(sc["lang"], cue, d["seed"] + d["turn"])
            if not cue and not_blank:
                opener = await self._improvise(sc, shown, d) or opener
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
        if outcome == "paused":  # never schedule more for someone a call upset
            self._store({
                "id": d["id"], "ts": datetime.now().isoformat(timespec="seconds"), "name": d["name"], "scenario_id": sc["id"],
                "lang": sc["lang"], "ending": ending, "outcome": outcome, "line_reached": d["line_idx"] + 1,
                "flags": list(d["flags"]), "next_dose": None,
            })
            await self._send({"type": "end", "ending": ending, "outcome": outcome})
            return
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
            "decide_s": round(time.monotonic() - d["answered_at"], 1) if d.get("answered_at") else None,
        })
        await self._send({"type": "end", "ending": ending, "outcome": outcome})

    def median_latency(self) -> int | None:
        return int(statistics.median(self.latencies)) if self.latencies else None

    def median_turn(self) -> int | None:
        return int(statistics.median(self.turn_ms)) if self.turn_ms else None

    def notices(self) -> list:
        out = list(self.s.warnings) + list(CONTENT["warnings"]) + list(self.runtime_notices)
        return out


engine = Engine(load_settings(), ROOT / "results.json")

# --------------------------------------------------------------------------- app
app = FastAPI(title="Teeka")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class PersonBody(BaseModel):
    name: str
    lang: str = "hi-IN"
    consent: bool = False
    code_word_set: bool = False  # only whether one exists: the word itself is never sent or kept


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


@app.get("/people")
async def people_page():
    return FileResponse(ROOT / "static" / "people.html")


@app.get("/library")
async def library_page():
    return FileResponse(ROOT / "static" / "library.html")


@app.get("/reports")
async def reports_page():
    return FileResponse(ROOT / "static" / "reports.html")


@app.get("/call")
async def call_page():
    return FileResponse(ROOT / "static" / "call.html")


_AUDIO_NAME = re.compile(r"^[0-9a-f]{64}\.wav$")


@app.get("/api/audio/{name}")
async def audio(name: str):
    path = engine.cache_dir / name
    if not _AUDIO_NAME.match(name) or not path.is_file():
        raise HTTPException(404, "No such audio")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "public, max-age=31536000, immutable"})


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
        "median_turn_ms": engine.median_turn(),
        "consent_required": engine.consent_required,
        "person": engine.read_people().get((name or "").strip()[:40]),
    }


@app.get("/api/people")
async def people():
    rows = engine.read_results()
    return [{**p, "doses": sum(r["name"] == p["name"] for r in rows)} for p in engine.read_people().values()]


@app.post("/api/people")
async def save_person(body: PersonBody):
    name = body.name.strip()[:40]
    if not name or body.lang not in CONTENT["languages"]:
        raise HTTPException(400, "A name and a supported language are needed.")
    return engine.save_person(name, body.lang, body.consent, body.code_word_set)


@app.get("/api/reports")
async def reports(name: str | None = None):
    return engine.report(name)


@app.get("/api/reports.csv")
async def reports_csv():
    """Outcomes only. People appear as numbers, so the file can be shared."""
    out, ids = io.StringIO(), {}
    w = csv.writer(out)
    w.writerow(["date", "person", "language", "scam", "result", "caller_line_reached", "seconds_to_decide"])
    for r in engine.read_results():
        pid = ids.setdefault(r["name"], len(ids) + 1)
        title = CONTENT["by_id"].get(r["scenario_id"], {}).get("title", r["scenario_id"])
        w.writerow([r["ts"][:10], f"person-{pid}", r.get("lang", ""), title, r["outcome"], r.get("line_reached", ""), r.get("decide_s") if r.get("decide_s") is not None else ""])
    return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="teeka-results.csv"'})


@app.get("/api/library")
async def library(lang: str = "hi-IN"):
    if not scenarios_in(lang):
        raise HTTPException(404, "Unknown language")
    items = []
    for s in CONTENT["scenarios"]:
        if s["lang"] != lang:
            continue
        flags = list(dict.fromkeys(f for line in s["lines"] for f in line["flags"]))
        lessons = list(dict.fromkeys(line["lesson"] for line in s["lines"]))
        items.append({"fraud": s["fraud"], "title": s["title"], "title_local": s["title_local"], "lang": lang,
                      "practise": s["practise"], "flags": [flag_public(f, lang) for f in flags], "lessons": lessons})
    return items


@app.get("/api/next_dose.ics")
async def next_dose_ics(name: str):
    nd = engine.card(name).get("next_dose")
    if not nd:
        raise HTTPException(404, "No dose is scheduled")
    day = nd["due"].replace("-", "")
    body = "\r\n".join(["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Teeka//Next dose//EN", "BEGIN:VEVENT",
                         f"UID:teeka-{day}-{abs(hash(name))}@teeka.local", f"DTSTAMP:{day}T000000Z", f"DTSTART;VALUE=DATE:{day}",
                         "SUMMARY:Teeka: next practice call", "END:VEVENT", "END:VCALENDAR", ""])
    return Response(body, media_type="text/calendar", headers={"Content-Disposition": 'attachment; filename="teeka-next-dose.ics"'})


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
