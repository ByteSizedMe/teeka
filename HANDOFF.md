# Teeka handoff

## Conversation and features (latest)
- The caller answers questions from each scam's facts, up to 8 lines. Distress ends the call kindly. Consent is required before a call.
- New pages: `/people`, `/library`, `/reports`. Dashboard shows immunity, red flags, help card and side rails on wide screens.
- `median_turn_ms` on the dashboard is real: end of the person's speech to the caller's first sound.
- Not yet tuned: `GRACE_S`, `GATE_TAIL_S` and the v2.5 versus v3.0 choice. These need live runs with `test_replies.py`.
- Regional languages have one written answer per fact, with rephrasing on repeat. Hindi and English have two.

## Content packs
- 15 fraud types in the 10 languages Gnani supports for both Prisma and Timbre, in `content/`. `content.py` validates them at startup.
- All packs have `"review": "draft"`. A native speaker must check each one, then change it to `native_reviewed`.
- Hindi and English have two written sentences per beat. The other eight have one, varied by interchangeable opening and closing sentences.
- Tamil and Telugu callers are female, because Timbre v2.5 lists no male voice preferring those languages.
- Run `python test_packs.py` as well as the three offline tests. `results.json` from before this change still works.

## Done (stages 1-8)
- Classifier, scenarios, keywords: 20/20 report phrases as text.
- Server, dashboard, phone page: mock mode and live mode. 12 server tests pass.
- Live Gnani: cached Timbre, Prisma stream with REST fallback, mic gate, voice cloning with Timbre fallback.
- `test_mic_gate.py` (10 offline + live A/B) and `test_replies.py` pass. README is written.

## Not done
- Nothing is half-built. The commit plan is in the chat. I ran no git commands.

## Run
    .venv\Scripts\activate ; set PYTHONUTF8=1
    python test_classifier.py ; python test_server.py ; python test_mic_gate.py --no-live
    uvicorn server:app --host 0.0.0.0 --port 8000
- Dashboard `/dashboard`; phone `/call`, then "Set up this phone". Empty key = mock mode.
- Live tests: `python test_mic_gate.py`, `python test_replies.py [--model prisma-v3.0] [--snr 10] [--voice Deepak]`.

## Known issues
- Reply scores (20/20) use synthetic Timbre speech. Quote only your own TV test.
- Keywords were tuned on Nalini's output; a second voice (Deepak) also scored 20/20.
- Roman "do", "no", "sat", "oh" are not number words. The digit rule always wins, even in a refusal.
- Wrap-up, hangup and decline are all stamped Protected. That is my reading of the report.
- Teacher reveal audio is about 21 s (23 s cloned). Consider trimming `reveal_outro`.
- Prisma v2.5 never sent the documented `processing` event. Timbre WAV headers needed repair (`fix_wav`).
- One phone and one drill at a time. No login. The 0.5 s gate tail is a guess.
- `STT_MODEL` in `.env` is your choice; v3.0 was faster and equally accurate on my runs.

## Test by hand
- Open both pages in a real browser, then a real phone over the HTTPS tunnel with the TV on.
- Check the caller's voice never shows up as a reply on a real speaker.
- Listen to every Timbre line. Have a Hindi speaker rewrite any that sound off.
- Record 15-25 s of YOUR OWN voice, set `FAMILY_VOICE_FILE`, run a drill, listen to it.
- Run the report's TV test with a real person; quote only that score.
- Before every commit run `git status` and confirm `.env`, `results.json`, `cache/`, audio and `family_voice*` are not listed.
