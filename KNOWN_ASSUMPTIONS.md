# KNOWN_ASSUMPTIONS — Jarvis Voice Assistant

This file records what could **not** be verified in the current build sandbox (Windows, no /dev/snd, no GPU, no Ollama daemon) so a human reviewer can double-check before production.

## Verified (via `pip index versions` on 2026-09-25)
- `pydantic==2.11.9`, `pydantic-settings==2.10.1` — installed and import-tested
- `faster-whisper==1.2.1`, `openwakeword==0.6.0`, `onnxruntime==1.30.0`, `sounddevice==0.5.6`
- `langchain==1.4.2`, `langchain-community==0.4.2`, `openai==1.107.3` (1.x line, compatible with langchain-openai 0.2.x), `ollama==0.6.2`
- `piper-tts==1.8.0`, `edge-tts==7.2.8`, `pytest==8.4.2`
- `config.py` validation — `pytest tests/test_config.py` passed (10 tests) in Windows sandbox.

## Not verified / assumed
- **Audio passthrough (`/dev/snd`, `arecord`, `aplay`)**: Windows host has no ALSA devices. Linux+Docker `/dev/snd` mapping could not be tested. Assumed to work on Linux with `alsa-utils` installed.
- **Silero VAD ONNX model**: `onnxruntime` can load ONNX, but actual Silero VAD model file (`silero_vad.onnx`) not present in sandbox; download via `scripts/download_models.sh` not executed end-to-end (no network guarantee in sandbox).
- **Faster-Whisper model cache**: `small` model not downloaded; `WHISPER_MODEL_CACHE=/models/whisper` volume not tested. Assumes `int8` on CPU works; `float16` on GPU path assumed based on docs, not GPU-tested.
- **Piper TTS binding vs CLI**: `piper-tts==1.8.0` assumed to provide Python binding; fallback to `piper` CLI not tested in container. Turkish model `tr_TR-model.onnx` not present — assumed URL in README is valid.
- **openWakeWord model**: No pretrained "Jarvis" model ships; assumed user will supply `.onnx` at `WAKEWORD_MODEL_PATH`. Threshold/cooldown logic not tested against real model.
- **Ollama connectivity**: No `ollama` daemon in sandbox; `OLLAMA_BASE_URL=http://ollama:11434` not reachable. Timeout/retry logic mocked in tests, not integration-tested.
- **GPU profile**: `deploy.resources.reservations.devices: [gpu]` not tested (no GPU in sandbox). Assumes `WHISPER_DEVICE=cuda` + `float16` works with CUDA image.
- **Transitive deps**: `numpy`, `ctranslate2`, `langchain-core`, `langchain-openai`, `soundfile`, `scipy` versions not pinned — assumed compatible. A lock file (e.g. `uv.lock`) should be generated on Linux before production.
- **SQLite persistence**: `memory.py` SQLite fallback not tested under full concurrent load; fallback to in-memory assumed.
- **Docker build**: `docker compose build` not run in Windows sandbox (Docker not available or not Linux). Dockerfile `apt-get` deps (`ffmpeg`, `alsa-utils`, `libasound2`, `libsndfile1`, `build-essential`) assumed correct from docs.

## Actions required before production
1. On a Linux host: `getent group audio` → set `AUDIO_GID` in `.env` / `docker-compose.yml` `group_add`.
2. Run `scripts/download_models.sh` and verify models exist at `/models/*`.
3. `docker compose build` + `docker compose up -d` + `arecord -l` / `aplay -l` inside container.
4. Generate lock file: `pip-compile requirements.txt` or `uv pip compile`.
5. Manual voice loop test (wake word → VAD → STT → LLM → TTS) with real mic/speaker.

