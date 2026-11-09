"""Mic gate checks: the caller's voice must never reach Prisma as mic audio.

Offline (always):  fake Prisma that records every frame it is sent.
Live (needs key):  the caller's real cached Timbre audio is fed in as "mic" audio
                   against the real Prisma stream, gate on vs gate off.

Run:  python test_mic_gate.py          (add --no-live to skip the live part)
"""
import array
import asyncio
import sys
import time
import traceback

import listener as L
from listener import CallListener, FRAME_BYTES, FRAME_S, SILENCE

LEAK = (b"\x11\x11" * 512)   # recognisable "caller voice" frame
REPLY = (b"\x22\x22" * 512)  # recognisable "person's reply" frame


class FakeStream:
    def __init__(self):
        self.sent = []  # (time, frame)
        self.events = asyncio.Queue()
        self.closed = False

    async def send(self, frame):
        self.sent.append((time.monotonic(), frame))

    async def close(self):
        self.closed = True


async def make(mode="stream", **kw):
    fs = FakeStream()

    async def open_stream():
        return fs

    notices = []
    lst = CallListener(mode=mode, lang="hi-IN", listen_max_s=kw.pop("listen_max_s", 2.0), min_silence_ms=300,
                       open_stream=open_stream, on_notice=notices.append, **kw)
    await lst.start()
    return lst, fs, notices


async def feed(lst, frame, seconds):
    """Pretend to be the phone: one mic frame every 32 ms."""
    nxt = time.monotonic()
    for _ in range(int(seconds / FRAME_S)):
        lst.push_frame(frame)
        nxt += FRAME_S
        await asyncio.sleep(max(0, nxt - time.monotonic()))


# ------------------------------------------------------------------ offline
async def t_no_leak_while_gated():
    lst, fs, _ = await make()
    lst.gate_close(0.8)
    lst.playback_start()
    t_start = time.monotonic()
    await feed(lst, LEAK, 0.8)           # caller's voice in the mic during playback
    lst.playback_end()
    t_end = time.monotonic()
    await feed(lst, LEAK, 0.3)           # room echo just after playback ends (inside the tail)
    await feed(lst, REPLY, 0.6)          # the person answers
    await lst.stop()
    leaked = [f for _, f in fs.sent if f == LEAK]
    assert not leaked, f"{len(leaked)} caller frames reached Prisma"
    assert all(f in (SILENCE, REPLY) for _, f in fs.sent)
    first_reply = min(t for t, f in fs.sent if f == REPLY)
    assert first_reply >= t_end + L.GATE_TAIL_S - 0.1, "reply audio passed before the tail ended"
    assert any(f == REPLY for _, f in fs.sent), "gate never reopened"
    return f"{len(fs.sent)} frames sent, 0 caller frames, reply passes {first_reply - t_end:.2f}s after playback end"


async def t_control_gate_open_leaks():
    """If the gate is disabled the same test MUST see the leak, or the test proves nothing."""
    lst, fs, _ = await make()
    lst.gate_closed = lambda: False
    await feed(lst, LEAK, 0.5)
    await lst.stop()
    n = sum(f == LEAK for _, f in fs.sent)
    assert n > 5, "control did not detect a leak, so the gate test is not trustworthy"
    return f"control: with the gate off {n} caller frames leaked (detector works)"


async def t_cadence():
    lst, fs, _ = await make()
    await feed(lst, REPLY, 1.5)
    await lst.stop()
    ts = [t for t, _ in fs.sent]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    mean = sum(gaps) / len(gaps)
    assert abs(mean - FRAME_S) < 0.004, mean
    assert max(gaps) < 0.12, max(gaps)
    assert all(len(f) == FRAME_BYTES for _, f in fs.sent)
    return f"{len(ts)} frames, mean gap {mean * 1000:.1f} ms, worst {max(gaps) * 1000:.0f} ms, all 1,024 bytes"


async def t_silence_fills_gaps_and_bad_frames_dropped():
    lst, fs, _ = await make()
    await asyncio.sleep(0.4)                       # phone sends nothing
    lst.push_frame(b"short")
    lst.push_frame(REPLY + b"x")
    await asyncio.sleep(0.2)
    await lst.stop()
    assert all(f == SILENCE for _, f in fs.sent) and len(fs.sent) >= 15
    assert lst.stats["bad_frames"] == 2
    return f"{len(fs.sent)} silence frames kept the stream alive; 2 malformed frames rejected"


async def t_failsafe_when_phone_never_reports_end():
    old = L.GATE_MARGIN_S
    L.GATE_MARGIN_S = 0.3
    try:
        lst, fs, _ = await make()
        lst.gate_close(0.3)                         # phone never sends playback_end
        assert lst.gate_closed()
        await asyncio.sleep(0.8)
        assert not lst.gate_closed(), "gate stuck shut forever"
        await lst.stop()
    finally:
        L.GATE_MARGIN_S = old
    return "gate reopens by itself if the phone never reports playback end"


async def t_stale_transcripts_never_become_replies():
    lst, fs, _ = await make(listen_max_s=0.6)
    lst.gate_close(0.3)
    lst.playback_start()
    fs.events.put_nowait({"type": "transcript", "text": "नमस्कार मैं एबीसी बैंक से बोल रहा हूँ", "latency": 100})
    await asyncio.sleep(0.2)
    lst.playback_end()
    text, ms = await lst.listen()                  # waits for the gate, drops the stale transcript
    await lst.stop()
    assert text == "" and ms is None, text
    return "a transcript produced while the caller spoke is discarded, not judged"


async def t_reply_after_gate_is_returned():
    lst, fs, _ = await make(listen_max_s=2.0)
    lst.gate_close(0.1)
    lst.playback_end()

    async def person():
        await asyncio.sleep(0.9)
        fs.events.put_nowait({"type": "speech_start"})
        fs.events.put_nowait({"type": "transcript", "text": "हाँ,", "latency": 90})
        await asyncio.sleep(0.3)
        fs.events.put_nowait({"type": "transcript", "text": "बताती हूँ", "latency": 110})

    asyncio.create_task(person())
    text, ms = await lst.listen()
    await lst.stop()
    assert text == "हाँ, बताती हूँ" and ms == 110, (text, ms)
    return "two segments of one sentence are joined; latency reported"


async def t_rest_mode_drops_gated_frames():
    sent = []

    async def transcribe(pcm):
        sent.append(pcm)
        return "ठीक है", 900

    lst = CallListener(mode="rest", lang="hi-IN", listen_max_s=3.0, min_silence_ms=200, transcribe=transcribe)
    await lst.start()
    loud = array.array("h", [4000, -4000] * 256).tobytes()
    lst.gate_close(0.5)
    lst.playback_start()
    task = asyncio.create_task(lst.listen())
    await feed(lst, loud, 0.5)                     # caller leak while gated
    lst.playback_end()
    await feed(lst, SILENCE, 0.7)                  # gate tail + quiet
    assert not task.done(), "leaked audio started an utterance"
    await feed(lst, loud, 0.5)                     # the person speaks
    await feed(lst, SILENCE, 0.5)
    text, ms = await asyncio.wait_for(task, 3)
    assert text == "ठीक है" and len(sent) == 1
    assert len(sent[0]) < 16000 * 2 * 1.5, "utterance includes the gated caller audio"
    return f"REST mode: gated frames dropped ({lst.stats['gated_dropped']}), one utterance of {len(sent[0]) / 32000:.1f}s sent"


async def t_stream_failure_falls_back_to_rest():
    notices = []

    async def bad():
        raise RuntimeError("refused")

    lst = CallListener(mode="stream", lang="hi-IN", listen_max_s=1, min_silence_ms=300, open_stream=bad,
                       on_notice=notices.append)
    await lst.start()
    assert lst.mode == "rest" and notices and "Streaming failed" in notices[0]
    return "stream open failure -> per-reply mode + dashboard notice"


async def t_stream_drop_mid_call_falls_back():
    lst, fs, notices = await make(listen_max_s=2.0)
    lst.gate_close(0.05)
    lst.playback_end()

    async def drop():
        await asyncio.sleep(0.8)
        fs.events.put_nowait({"type": "closed"})

    asyncio.create_task(drop())
    text, _ = await lst.listen()
    await lst.stop()
    assert text == "" and lst.mode == "rest" and notices
    return "stream dropping mid-listen -> per-reply mode + notice (that reply counts as unclear)"


# --------------------------------------------------------------------- live
async def live_part():
    import server
    from gnani_client import wav_to_pcm

    st, client = server.engine.s, server.engine.client
    if client is None:
        return ["SKIP live part: no GNANI_API_KEY"]
    sc = server.CONTENT["by_id"]["bank-kyc-hi"]
    voice, language = server.voice_for("hi-IN", "caller", st.tts_model)
    path = client.cache_path(sc["lines"][0]["text"], voice, language, st.tts_model, 24000, 1.0)
    if not path.exists():
        return ["SKIP live part: caller audio not cached (run: python gnani_client.py warm)"]
    pcm24, rate = wav_to_pcm(path.read_bytes())
    a = array.array("h")
    a.frombytes(pcm24)
    n = int(len(a) * 16000 / rate)  # linear resample 24k -> 16k
    out = array.array("h", [0] * n)
    for i in range(n):
        x = i * rate / 16000
        j = int(x)
        fr = x - j
        out[i] = int(a[j] * (1 - fr) + a[min(j + 1, len(a) - 1)] * fr)
    raw = out.tobytes()
    frames = [raw[i:i + FRAME_BYTES].ljust(FRAME_BYTES, b"\0") for i in range(0, len(raw), FRAME_BYTES)]
    dur = len(raw) / 32000

    async def run(gate_on):
        async def open_stream():
            return await client.open_stream("hi-IN", st.stt_model, st.vad_threshold, st.vad_min_silence_ms)

        lst = CallListener(mode="stream", lang="hi-IN", listen_max_s=5, min_silence_ms=st.vad_min_silence_ms,
                           open_stream=open_stream)
        if not gate_on:
            lst.gate_closed = lambda: False
        await lst.start()
        assert lst.stream is not None, "could not open the real Prisma stream"
        lst.gate_close(dur)
        lst.playback_start()
        nxt = time.monotonic()
        for f in frames:                              # the caller's voice arrives via the "mic"
            lst.push_frame(f)
            nxt += FRAME_S
            await asyncio.sleep(max(0, nxt - time.monotonic()))
        lst.playback_end()
        await asyncio.sleep(3.5)                      # let Prisma finish any segment (raw events, not listen())
        events = []
        while not lst.stream.events.empty():
            events.append(lst.stream.events.get_nowait())
        await lst.stop()
        words = " ".join(e.get("text", "") for e in events if e.get("type") == "transcript")
        return words.strip(), sum(e.get("type") == "speech_start" for e in events), lst.stats

    lines = []
    heard, starts, stats = await run(gate_on=False)
    assert heard, "CONTROL FAILED: Prisma did not transcribe the caller audio even with the gate off"
    lines.append(f"live control (gate OFF): Prisma heard the caller: '{heard[:50]}…'")
    heard2, starts2, stats2 = await run(gate_on=True)
    assert heard2 == "" and starts2 == 0, f"LEAK: Prisma transcribed '{heard2}'"
    lines.append(f"live (gate ON): 0 transcripts, 0 speech starts; {stats2['silence']} silence frames sent, "
                 f"{stats2['gated_dropped']} real mic frames discarded")
    return lines


OFFLINE = [t_no_leak_while_gated, t_control_gate_open_leaks, t_cadence,
           t_silence_fills_gaps_and_bad_frames_dropped, t_failsafe_when_phone_never_reports_end,
           t_stale_transcripts_never_become_replies, t_reply_after_gate_is_returned,
           t_rest_mode_drops_gated_frames, t_stream_failure_falls_back_to_rest, t_stream_drop_mid_call_falls_back]


async def main():
    bad = 0
    for fn in OFFLINE:
        try:
            print(f"ok    {fn.__name__[2:]}: {await fn()}")
        except Exception:
            bad += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    if "--no-live" not in sys.argv:
        try:
            for line in await live_part():
                print(("ok    " if not line.startswith("SKIP") else "") + line)
        except Exception:
            bad += 1
            print("FAIL  live part")
            traceback.print_exc()
    print("RESULT:", "all passed" if not bad else f"{bad} failed")
    return bad


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
