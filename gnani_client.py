"""Gnani API calls: Timbre TTS (cached), Prisma realtime stream, Prisma REST.
Imported only by the server. Endpoints, headers and fields are taken from
docs.gnani.ai (Quick Start, STT Realtime, STT REST, TTS inference).

The API key is only ever put in request headers. It is never logged, and error
messages built here never include it.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import struct
import time
import wave
from pathlib import Path

import httpx
import websockets

TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
STT_REST_URL = "https://api.vachana.ai/stt/v3"
EMBED_URL = "https://api.vachana.ai/api/v1/tts/voice-clone/embeddings"
STT_WS_URL = "wss://api.vachana.ai/stt/v3/stream"


class GnaniError(Exception):
    """A readable problem with a Gnani call. Safe to show on the dashboard."""


def _body_message(resp: httpx.Response) -> str:
    try:
        j = resp.json()
    except ValueError:
        return resp.text[:160]
    if isinstance(j, dict):
        err = j.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("type") or "")[:160]
        for k in ("message", "detail"):
            if j.get(k):
                return str(j[k])[:160]
    return str(j)[:160]


def pcm_to_wav(pcm: bytes, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def fix_wav(data: bytes) -> bytes:
    """Timbre sends WAVs whose RIFF and data sizes are 0xFFFFFFFF (streaming style).
    Rewrite both from the real byte count so durations and players are right."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    i = data.find(b"data", 12)
    if i < 0:
        return data
    body = len(data) - (i + 8)
    out = bytearray(data)
    out[4:8] = struct.pack("<I", len(data) - 8)
    out[i + 4 : i + 8] = struct.pack("<I", body)
    return bytes(out)


def wav_to_pcm(data: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(fix_wav(data)), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def wav_duration(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(fix_wav(data)), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError):
        return 0.0


def extract_embedding(body: dict) -> dict:
    """The docs show two shapes: flat {embedding, shape, dtype} (Quick Start) and
    {success, message, data: {voice_clone_embedding: {...}}} (embeddings page). Accept both."""
    if isinstance(body, dict):
        node = body
        if isinstance(body.get("data"), dict):
            node = body["data"].get("voice_clone_embedding", body["data"])
        if isinstance(node.get("voice_clone_embedding"), dict):
            node = node["voice_clone_embedding"]
        if isinstance(node.get("embedding"), str) and node["embedding"]:
            return {"embedding": node["embedding"], "shape": node.get("shape", [1, 768]),
                    "dtype": node.get("dtype", "torch.bfloat16")}
    raise GnaniError("Voice cloning returned no embedding.")


class PrismaStream:
    """One realtime Prisma connection. Send 1,024-byte frames at 32 ms cadence;
    read `events` (dicts with type: processing | speech_start | transcript | error | closed)."""

    def __init__(self, api_key, lang, model, vad_threshold, min_silence_ms):
        self._key, self.lang, self.model = api_key, lang, model
        self.vad_threshold, self.min_silence_ms = vad_threshold, min_silence_ms
        self.events: asyncio.Queue = asyncio.Queue()
        self.ws = None
        self._reader: asyncio.Task | None = None

    async def open(self) -> None:
        headers = {
            "x-api-key-id": self._key,
            "language_code": self.lang,
            "model": self.model,
            "x-sample-rate": "16000",
            "x-vad-threshold": str(self.vad_threshold),
            "x-min-silence-ms": str(self.min_silence_ms),
            "x-idle-timeout-sec": "300",
        }
        try:
            self.ws = await websockets.connect(
                STT_WS_URL, additional_headers=headers, open_timeout=10, max_size=2**20, ping_interval=20
            )
            first = json.loads(await asyncio.wait_for(self.ws.recv(), 10))
        except websockets.exceptions.InvalidStatus as e:
            raise GnaniError(f"Prisma stream refused the connection (HTTP {e.response.status_code}).") from None
        except Exception as e:
            raise GnaniError(f"Could not open the Prisma stream ({type(e).__name__}).") from None
        if first.get("type") == "error":
            raise GnaniError("Prisma stream error: " + str(first.get("message", ""))[:120])
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                self.events.put_nowait(msg)
        except Exception:
            pass
        self.events.put_nowait({"type": "closed"})

    async def send(self, frame: bytes) -> None:
        await self.ws.send(frame)

    async def commit(self) -> None:
        await self.ws.send(json.dumps({"type": "commit"}))

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass


class GnaniClient:
    def __init__(self, api_key: str, cache_dir: Path):
        self._key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(40.0, connect=10.0))
        self._locks: dict[str, asyncio.Lock] = {}
        self.tts_calls = 0  # real network calls (cache misses), for the credit count

    async def close(self) -> None:
        await self._http.aclose()

    # ---------------------------------------------------------------- Timbre
    def cache_path(self, text, voice, language, model, sample_rate, speed) -> Path:
        key = json.dumps([model, voice, language, sample_rate, speed, text], ensure_ascii=False)
        return self.cache_dir / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".wav")

    def is_cached(self, text, voice, language, model, sample_rate=24000, speed=1.0) -> bool:
        return self.cache_path(text, voice, language, model, sample_rate, speed).exists()

    async def tts(self, text, voice, language, model, sample_rate=24000, speed=1.0) -> Path:
        """Speak `text`; returns the cached WAV path. Only a cache miss costs credits."""
        path = self.cache_path(text, voice, language, model, sample_rate, speed)
        if path.exists():
            return path
        lock = self._locks.setdefault(path.name, asyncio.Lock())
        async with lock:
            if path.exists():
                return path
            payload = {
                "text": text, "voice": voice, "model": model, "language": language, "speed": speed,
                "audio_config": {"sample_rate": sample_rate, "num_channels": 1, "sample_width": 2,
                                 "encoding": "linear_pcm", "container": "wav"},
            }
            last = "no response"
            for attempt in (1, 2):
                try:
                    r = await self._http.post(
                        TTS_URL, json=payload, headers={"X-API-Key-ID": self._key, "Content-Type": "application/json"}
                    )
                except httpx.HTTPError as e:
                    last = f"network error ({type(e).__name__})"
                    continue
                if r.status_code == 200 and r.content[:4] == b"RIFF":
                    self.tts_calls += 1
                    tmp = path.with_suffix(".part")
                    tmp.write_bytes(fix_wav(r.content))
                    tmp.replace(path)
                    return path
                last = f"HTTP {r.status_code}: {_body_message(r)}"
                if r.status_code < 500 and r.status_code != 429:
                    break
                await asyncio.sleep(1.0)
            raise GnaniError(f"Timbre failed ({last}).")

    # ------------------------------------------------------- Prisma, one reply
    async def transcribe(self, pcm16k: bytes, lang: str, model: str) -> tuple[str, int]:
        """POST /stt/v3 with one recorded reply. Returns (text, wall-clock ms)."""
        data = {"language_code": lang}
        if model == "prisma-v3.0":  # v2.5 is the default when `model` is left out
            data["model"] = model
        t0 = time.monotonic()
        try:
            r = await self._http.post(
                STT_REST_URL, headers={"X-API-Key-ID": self._key}, data=data,
                files={"audio_file": ("reply.wav", pcm_to_wav(pcm16k), "audio/wav")},
            )
        except httpx.HTTPError as e:
            raise GnaniError(f"Prisma REST network error ({type(e).__name__}).") from None
        ms = int((time.monotonic() - t0) * 1000)
        if r.status_code != 200:
            raise GnaniError(f"Prisma REST failed (HTTP {r.status_code}: {_body_message(r)}).")
        try:
            body = r.json()
        except ValueError:
            raise GnaniError("Prisma REST returned something that is not JSON.") from None
        if body.get("success") is False:
            raise GnaniError("Prisma REST error: " + _body_message(r))
        return str(body.get("transcript") or ""), ms

    # --------------------------------------------------------- voice cloning
    async def clone_embedding(self, audio_path: Path) -> dict:
        """Upload a 5-30 s reference recording; returns {"embedding","shape","dtype"}.
        Cached by file content, so each recording is uploaded once."""
        data = Path(audio_path).read_bytes()
        cache = self.cache_dir.parent / "voice" / (hashlib.sha256(data).hexdigest() + ".json")
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        try:
            r = await self._http.post(
                EMBED_URL, headers={"X-API-Key-ID": self._key},
                files={"audio_file": (Path(audio_path).name, data)},
            )
        except httpx.HTTPError as e:
            raise GnaniError(f"Voice cloning network error ({type(e).__name__}).") from None
        if r.status_code != 200:
            raise GnaniError(f"Voice cloning refused the recording (HTTP {r.status_code}: {_body_message(r)}).")
        emb = extract_embedding(r.json())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(emb), encoding="utf-8")
        return emb

    async def tts_cloned(self, text: str, embedding: dict, sample_rate: int = 24000) -> Path:
        """Speak `text` in the cloned voice (model vachana-vc-v1), cached."""
        eh = hashlib.sha256(embedding["embedding"].encode()).hexdigest()[:16]
        key = json.dumps(["vachana-vc-v1", eh, sample_rate, text], ensure_ascii=False)
        path = self.cache_dir / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".wav")
        if path.exists():
            return path
        payload = {
            "text": text, "model": "vachana-vc-v1", "speaker_embedding": embedding,
            "audio_config": {"sample_rate": sample_rate, "num_channels": 1, "sample_width": 2,
                             "encoding": "linear_pcm", "container": "wav"},
        }
        try:
            r = await self._http.post(
                TTS_URL, json=payload, headers={"X-API-Key-ID": self._key, "Content-Type": "application/json"})
        except httpx.HTTPError as e:
            raise GnaniError(f"Cloned voice network error ({type(e).__name__}).") from None
        if r.status_code != 200 or r.content[:4] != b"RIFF":
            raise GnaniError(f"Cloned voice failed (HTTP {r.status_code}: {_body_message(r)}).")
        self.tts_calls += 1
        tmp = path.with_suffix(".part")
        tmp.write_bytes(fix_wav(r.content))
        tmp.replace(path)
        return path

    # --------------------------------------------------------- Prisma, stream
    async def open_stream(self, lang, model, vad_threshold, min_silence_ms) -> PrismaStream:
        s = PrismaStream(self._key, lang, model, vad_threshold, min_silence_ms)
        await s.open()
        return s


if __name__ == "__main__":
    # python gnani_client.py warm hi-IN         -> says how many Timbre calls it would make
    # python gnani_client.py warm hi-IN --yes   -> makes them, into cache/tts
    import sys

    import server  # loads .env and the content packs

    async def warm(lang, spend):
        st = server.engine.s
        client = server.engine.client
        if client is None:
            sys.exit("No GNANI_API_KEY: nothing to warm.")
        voice, language = server.voice_for(lang, "caller", st.tts_model)
        lines = [line["text"] for sc in server.CONTENT["scenarios"] if sc["lang"] == lang for line in sc["lines"]]
        missing = [t for t in lines if not client.cache_path(t, voice, language, st.tts_model, 24000, 1.0).exists()]
        print(f"{lang}: {len(lines)} caller lines, {len(missing)} not cached yet = {len(missing)} Timbre calls")
        if spend:
            for text in missing:
                await client.tts(text, voice, language, st.tts_model)
            print("Timbre calls made this run:", client.tts_calls)
        else:
            print("Nothing spent. Add --yes to make these calls.")
        await client.close()

    if len(sys.argv) >= 3 and sys.argv[1] == "warm":
        asyncio.run(warm(sys.argv[2], "--yes" in sys.argv))
