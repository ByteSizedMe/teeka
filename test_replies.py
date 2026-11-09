"""Run the report's 20 reply phrases through real Prisma and the classifier.

Each phrase is spoken by a Timbre voice (cached after the first run), sent to Prisma
at real-time pace, and the transcript is judged by classifier.py. Prints every result,
the score and the median latency. Nothing is written to disk.

IMPORTANT: this is synthetic speech, not a person with a TV on. A Timbre voice is
cleaner and more regular than a real older speaker, so treat the score as an upper
bound for the Prisma + classifier path. --snr adds synthetic white noise as a rough
stand-in for a noisy room; it is not a recording of one.

    python test_replies.py                       # settings from .env
    python test_replies.py --model prisma-v3.0   # compare models
    python test_replies.py --mode rest
    python test_replies.py --snr 10              # add noise at 10 dB signal-to-noise
    python test_replies.py --voice Bhavna        # try another speaker
    python test_replies.py --lang ta-IN          # another language's test replies
    python test_replies.py --lang ta-IN --count  # say how many API calls it would make, spend nothing
"""
import argparse
import array
import asyncio
import math
import random
import statistics
import sys
import time

import classifier
from listener import FRAME_BYTES, FRAME_S, SILENCE

# (phrase, expected) from the report's accuracy table. #10 is the report's English/Hinglish one.
PHRASES = [
    ("अच्छा रुको, चार सात एक नौ", "complying"),
    ("हाँ, ओटीपी आया है", "complying"),
    ("लिख लो, बताती हूँ", "complying"),
    ("ठीक है, लिंक खोल रही हूँ", "complying"),
    ("कितने पैसे भेजने हैं?", "complying"),
    ("हाँ, पिन डाल रही हूँ", "complying"),
    ("ऐप डाउनलोड कर लेती हूँ", "complying"),
    ("भेज दिया, देखिए", "complying"),
    ("एक मिनट, नंबर बताती हूँ, दो पाँच आठ", "complying"),
    ("Okay, OTP is 4 7 1 9", "complying"),
    ("मैं पहले अपने बेटे से पूछ लेती हूँ", "refusing"),
    ("नहीं, मैं नहीं बताऊँगी", "refusing"),
    ("मैं फ़ोन रख रही हूँ", "refusing"),
    ("मैं खुद बैंक जाकर बात करूँगी", "refusing"),
    ("आप कौन बोल रहे हैं?", "refusing"),
    ("यह तो धोखा लग रहा है", "refusing"),
    ("मुझे नहीं चाहिए", "refusing"),
    ("मैं ओटीपी नहीं दूँगी", "refusing"),
    ("पुलिस को बताऊँगी", "refusing"),
    ("बाद में बात करना", "refusing"),
]
TAIL_FRAMES = 40  # 1.3 s of quiet after the phrase so Prisma's VAD closes the segment


def add_noise(pcm: bytes, snr_db: float, rng: random.Random) -> bytes:
    a = array.array("h")
    a.frombytes(pcm)
    rms = math.sqrt(sum(x * x for x in a) / max(1, len(a))) or 1.0
    sigma = rms / (10 ** (snr_db / 20))
    out = array.array("h", (max(-32768, min(32767, int(x + rng.gauss(0, sigma)))) for x in a))
    return out.tobytes()


def to_frames(pcm: bytes) -> list[bytes]:
    f = [pcm[i:i + FRAME_BYTES].ljust(FRAME_BYTES, b"\0") for i in range(0, len(pcm), FRAME_BYTES)]
    return f + [SILENCE] * TAIL_FRAMES


async def via_stream(client, frames, lang, model, vad, sil):
    s = await client.open_stream(lang, model, vad, sil)
    parts, lats, last = [], [], None
    t_end = None

    async def reader():
        nonlocal last
        while True:
            m = await s.events.get()
            if m["type"] == "transcript" and (m.get("text") or "").strip():
                parts.append(m["text"].strip())
                if isinstance(m.get("latency"), (int, float)):
                    lats.append(int(m["latency"]))
                last = time.monotonic()
            elif m["type"] == "closed":
                return

    r = asyncio.create_task(reader())
    nxt = time.monotonic()
    for f in frames:
        await s.send(f)
        nxt += FRAME_S
        await asyncio.sleep(max(0, nxt - time.monotonic()))
    t_end = time.monotonic()
    waited = 0.0
    while waited < 4.0 and not (last and time.monotonic() - last > 1.0):
        await asyncio.sleep(0.1)
        waited += 0.1
    await s.close()
    r.cancel()
    return " ".join(parts), (max(lats) if lats else None)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stream", "rest"])
    ap.add_argument("--model", choices=["prisma-v2.5", "prisma-v3.0"])
    ap.add_argument("--snr", type=float, help="add white noise at this SNR in dB")
    ap.add_argument("--voice")
    ap.add_argument("--lang", default="hi-IN")
    ap.add_argument("--count", action="store_true", help="print the API calls this would make, and stop")
    args = ap.parse_args()

    import server
    from gnani_client import wav_to_pcm

    if args.lang not in server.CONTENT["languages"]:
        sys.exit(f"Unknown language {args.lang}.")
    pack = server.CONTENT["languages"][args.lang]
    phrases = PHRASES if args.lang == "hi-IN" else [tuple(r) for r in pack.get("test_replies", [])]
    voice = args.voice or ("Nalini" if args.lang == "hi-IN" else pack["voices"]["timbre-v2.5"]["teacher"])
    if args.count:
        print(f"{args.lang}: {len(phrases)} replies = up to {len(phrases)} Timbre calls (cached ones are free) and {len(phrases)} Prisma calls.")
        return 0
    st, client = server.engine.s, server.engine.client
    if client is None:
        sys.exit("No GNANI_API_KEY in .env: this test needs the real API.")
    mode, model = args.mode or st.stt_mode, args.model or st.stt_model
    rng = random.Random(7)
    print(f"SYNTHETIC SPEECH, not a real person. language={args.lang} mode={mode} model={model} voice={voice} noise={'%g dB SNR' % args.snr if args.snr else 'none'} "
          f"VAD={st.vad_threshold}/{st.vad_min_silence_ms}ms  (synthetic speech, see file header)\n")

    rows, lats, mism = [], [], []
    for n, (phrase, want) in enumerate(phrases, 1):
        english = args.lang == "hi-IN" and n == 10
        speaker, tts_lang = ("Poorvi", "hi-en") if english else (voice, args.lang)
        path = await client.tts(phrase, speaker, tts_lang, "timbre-v2.5", sample_rate=16000)
        pcm, rate = wav_to_pcm(path.read_bytes())
        assert rate == 16000
        if args.snr:
            pcm = add_noise(pcm, args.snr, rng)
        try:
            if mode == "stream":
                heard, ms = await via_stream(client, to_frames(pcm), args.lang, model, st.vad_threshold, st.vad_min_silence_ms)
            else:
                heard, ms = await client.transcribe(pcm, args.lang, model)
        except Exception as e:
            heard, ms = f"<error: {e}>", None
        got = classifier.classify(heard, args.lang)
        ok = got == want
        if ms is not None:
            lats.append(ms)
        rows.append(ok)
        if not ok:
            mism.append((n, phrase, heard, want, got))
        print(f"{n:2} {'PASS' if ok else 'FAIL'}  want={want:9} got={got:9} {('%4d ms' % ms) if ms is not None else '   - ms'}"
              f"  said: {phrase}\n{'':32}heard: {heard}")

    score = sum(rows)
    print(f"\nSCORE ({args.lang}, synthetic speech): {score}/{len(phrases)}   (report target: 90% or better)")
    if lats:
        print(f"MEDIAN LATENCY: {int(statistics.median(lats))} ms "
              f"({'Prisma-reported, per segment' if mode == 'stream' else 'wall clock, REST'}), "
              f"range {min(lats)}-{max(lats)} ms")
    print(f"Timbre calls this run (cache misses): {client.tts_calls}")
    for n, phrase, heard, want, got in mism:
        print(f"MISMATCH #{n}: wanted {want}, got {got}. Prisma printed: {heard!r}")
    await client.close()
    return 0 if score >= 0.9 * len(phrases) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
