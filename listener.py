"""Call-audio pipeline between the phone's mic and Prisma.

* The phone sends 1,024-byte frames (16 kHz, 16-bit, mono, 32 ms each).
* MIC GATE: while the caller's (or teacher's) voice is playing on the phone, the
  frames that go to Prisma are silence, and the real mic frames are thrown away.
  The gate opens a short tail after the phone reports that playback ended.
* Stream mode: one frame is sent every 32 ms (real-time cadence) whatever the
  phone does; silence fills any gap.
* REST mode: ungated frames are cut into utterances by a simple energy VAD and
  each utterance is sent to Prisma as one WAV.

Nothing here keeps the text of a reply beyond returning it to the caller.
"""
from __future__ import annotations

import asyncio
import math
import time
from array import array
from collections import deque

FRAME_BYTES = 1024
FRAME_S = 0.032
SILENCE = bytes(FRAME_BYTES)
GATE_TAIL_S = 0.5  # keep the gate shut a little after playback ends (speaker echo, buffering)
GATE_MARGIN_S = 8.0  # fail-safe: never stay gated longer than duration + this (allows a slow download over the tunnel)
GRACE_S = 0.8  # wait this long after a transcript for the rest of the sentence
PROCESSING_WAIT_S = 5.0  # extra time to wait for a transcript once Prisma says it is processing
MAX_BACKLOG = 15  # frames; beyond this we drop the oldest to stay real-time


class CallListener:
    def __init__(self, *, mode, lang, listen_max_s, min_silence_ms, open_stream=None, transcribe=None,
                 on_notice=None, clock=time.monotonic):
        self.mode = mode  # "stream" or "rest"
        self.lang, self.listen_max_s, self.min_silence_ms = lang, listen_max_s, min_silence_ms
        self._open_stream, self._transcribe = open_stream, transcribe
        self._notice = on_notice or (lambda t: None)
        self.clock = clock
        self.stream = None
        self._pump_task: asyncio.Task | None = None
        self._frames: deque = deque()
        # gate
        self._playing = False
        self._failsafe_until = 0.0
        self._gated_until = 0.0
        self._dur = 0.0
        # rest-mode VAD
        self._listening = False
        self._preroll: deque = deque(maxlen=6)
        self._utt = bytearray()
        self._speech = False
        self._last_speech = 0.0
        self._floor = 120.0
        self._utt_done = asyncio.Event()
        self.stats = {"real": 0, "silence": 0, "gated_dropped": 0, "bad_frames": 0}

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self.mode == "stream":
            try:
                self.stream = await self._open_stream()
            except Exception as e:  # GnaniError or anything else: fall back
                self.mode = "rest"
                self._notice(f"Streaming failed, using per-reply mode instead. {e}")
            else:
                self._pump_task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        if self._pump_task:
            self._pump_task.cancel()
            self._pump_task = None
        if self.stream:
            await self.stream.close()
            self.stream = None

    # ----------------------------------------------------------------- gate
    def gate_closed(self) -> bool:
        now = self.clock()
        if self._playing and now < self._failsafe_until:
            return True
        return now < self._gated_until

    def gate_close(self, duration_s: float) -> None:
        """Call just before telling the phone to play audio of this length."""
        now = self.clock()
        self._playing, self._dur = True, duration_s
        self._failsafe_until = now + duration_s + GATE_MARGIN_S
        self._reset_utterance()

    def playback_start(self) -> None:
        self._failsafe_until = self.clock() + self._dur + 1.0
        self._playing = True

    def playback_end(self) -> None:
        self._playing = False
        self._gated_until = self.clock() + GATE_TAIL_S
        self._reset_utterance()

    # ------------------------------------------------------- frames in
    def push_frame(self, data: bytes) -> None:
        if len(data) != FRAME_BYTES:
            self.stats["bad_frames"] += 1
            return
        if self.mode == "stream":
            self._frames.append(data)
            if len(self._frames) > MAX_BACKLOG:
                while len(self._frames) > MAX_BACKLOG // 2:
                    self._frames.popleft()
        else:
            self._rest_frame(data)

    async def _pump(self) -> None:
        """Send exactly one frame to Prisma every 32 ms: a real one, or silence."""
        nxt = self.clock()
        while True:
            gated = self.gate_closed()
            frame = self._frames.popleft() if self._frames else None
            if gated:
                self._frames.clear()
                if frame is not None:
                    self.stats["gated_dropped"] += 1
            if gated or frame is None:
                out = SILENCE
                self.stats["silence"] += 1
            else:
                out = frame
                self.stats["real"] += 1
            try:
                await self.stream.send(out)
            except Exception:
                self.stream.events.put_nowait({"type": "closed"})
                return
            nxt += FRAME_S
            delay = nxt - self.clock()
            if delay < -0.5:
                nxt = self.clock()
            await asyncio.sleep(max(0.0, delay))

    # -------------------------------------------------------- rest-mode VAD
    def _reset_utterance(self) -> None:
        self._utt = bytearray()
        self._speech = False
        self._preroll.clear()
        self._utt_done.clear()

    @staticmethod
    def _rms(frame: bytes) -> float:
        a = array("h")
        a.frombytes(frame)
        return math.sqrt(sum(x * x for x in a) / len(a))

    def _rest_frame(self, frame: bytes) -> None:
        if self.gate_closed():
            self.stats["gated_dropped"] += 1
            return
        if not self._listening or self._utt_done.is_set():
            return
        now = self.clock()
        level = self._rms(frame)
        loud = level > max(350.0, self._floor * 3)
        if loud:
            if not self._speech:
                self._speech = True
                for f in self._preroll:
                    self._utt += f
            self._last_speech = now
            self._utt += frame
        elif self._speech:
            self._utt += frame
            if now - self._last_speech >= self.min_silence_ms / 1000.0:
                self._utt_done.set()
        else:
            self._floor = 0.95 * self._floor + 0.05 * level
            self._preroll.append(frame)
        if len(self._utt) > 16000 * 2 * 25:  # 25 s cap (REST limit is 30 s)
            self._utt_done.set()

    # --------------------------------------------------------------- listen
    async def _wait_gate_open(self) -> None:
        while self.gate_closed():
            await asyncio.sleep(0.02)

    async def listen(self) -> tuple[str, int | None]:
        """Wait for one reply. Returns (text, prisma_latency_ms). ("", None) = silence."""
        await self._wait_gate_open()
        if self.mode == "stream":
            return await self._listen_stream()
        return await self._listen_rest()

    async def _listen_stream(self) -> tuple[str, int | None]:
        ev = self.stream.events
        while not ev.empty():  # drop anything left over from before the gate opened
            ev.get_nowait()
        deadline = self.clock() + self.listen_max_s
        processing_since = None
        parts, latencies = [], []
        grace_until = None
        while True:
            now = self.clock()
            if grace_until is not None:
                wait = grace_until - now
            elif processing_since is not None:
                wait = processing_since + PROCESSING_WAIT_S - now
            else:
                wait = deadline - now
            if wait <= 0:
                break
            try:
                msg = await asyncio.wait_for(ev.get(), wait)
            except asyncio.TimeoutError:
                break
            kind = msg.get("type")
            if kind == "processing":
                processing_since = self.clock()
            elif kind == "speech_start":
                deadline = max(deadline, self.clock() + 3.0)  # someone is talking: don't cut them off
                grace_until = None
            elif kind == "transcript":
                if self.gate_closed():
                    continue  # never accept words heard while the caller's voice was playing
                text = str(msg.get("text") or "").strip()
                processing_since = None
                if text:
                    parts.append(text)
                    if isinstance(msg.get("latency"), (int, float)):
                        latencies.append(int(msg["latency"]))
                    grace_until = self.clock() + GRACE_S
            elif kind in ("error", "closed"):
                self._notice("The Prisma stream dropped; switching to per-reply mode.")
                self.mode = "rest"
                break
        return " ".join(parts), (max(latencies) if latencies else None)

    async def _listen_rest(self) -> tuple[str, int | None]:
        self._reset_utterance()
        self._listening = True
        try:
            try:
                await asyncio.wait_for(self._utt_done.wait(), self.listen_max_s)
            except asyncio.TimeoutError:
                if not self._speech:
                    return "", None
                try:  # they are mid-sentence: give them a little longer
                    await asyncio.wait_for(self._utt_done.wait(), PROCESSING_WAIT_S)
                except asyncio.TimeoutError:
                    pass
            pcm = bytes(self._utt)
        finally:
            self._listening = False
        if len(pcm) < FRAME_BYTES * 3:
            return "", None
        try:
            return await self._transcribe(pcm)
        except Exception as e:
            self._notice(f"Per-reply transcription failed: {e}")
            return "", None
