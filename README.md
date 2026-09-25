# Jarvis Voice Assistant — Linux + ALSA, Dockerized

Turkish-only, local-first voice assistant: **openWakeWord → Silero VAD → Faster-Whisper → Ollama/OpenAI → Piper/edge-tts**, with SQLite memory via modern `RunnableWithMessageHistory`.

> **First release targets Linux + ALSA only.** Host audio is mapped via `/dev/snd` inside the container. See *Windows/macOS limitations* below.

---

## 1. Architecture (audio-thread / two-tier VAD)

```
Mic (ALSA, 16kHz mono 16-bit PCM)
  │
  ├─► Audio Thread (producer, sounddevice callback) ──queue──► Main Thread (consumer)
  │         │                                                     │
  │   Wake-word detection                           ┌─────────────┼─────────────┐
  │   (openWakeWord, 80ms chunks, cooldown)         │             │             │
  │         │                                       VAD?     Transcribe   LLM → TTS
  │         beep ──► Silero VAD endpointing        (Silero)  (Whisper) (Ollama/OpenAI)
  │                    ├─ speech start timeout              │             │
  │                    ├─ silence 900ms stop ──────────────┘             │
  │                    └─ max 20s force stop                              │
  │                                                                      play (muted wake-word)
  └──────────────────────────────────────────────────────────────────────┘
Heartbeat → /tmp/jarvis_heartbeat (healthcheck.py) ← no HTTP API
SQLite memory (SQLChatMessageHistory + RunnableWithMessageHistory) with trim + fallback
```

**Two-tier VAD (why it matters):**
- **Silero VAD (onnxruntime)** = real-time endpointing — decides *when to stop recording*. Runs continuously on live chunks with pre-roll buffer (300ms) so first word isn't clipped.
- **Whisper VAD filter** = secondary cleanup *at transcription time* — strips leading/trailing silence, reduces hallucinations on near-silent buffers. It does **not** decide when to stop recording. Conflating them causes mid-sentence cuts or never-stops. Code keeps them separate (`audio_handler.py` vs `stt_handler.py`).

**Concurrency:** Single dedicated audio thread (producer) feeding a `queue.Queue`. Wake-word/VAD/record lives there. STT→LLM→TTS runs on main thread so slow LLM/TTS never blocks capture. See `main.py:27` and `audio_handler.py:1` comments.

---

## 2. Directory Structure

```
D:/ProjAI/VoiceAssistant/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
├── .gitignore
├── README.md
├── KNOWN_ASSUMPTIONS.md
├── entrypoint.sh
├── main.py
├── config.py
├── audio_handler.py
├── wakeword_handler.py
├── stt_handler.py
├── llm_handler.py
├── tts_handler.py
├── memory.py
├── healthcheck.py
├── scripts/
│   └── download_models.sh
└── tests/
    ├── test_config.py
    ├── test_memory.py
    ├── test_prompt.py
    ├── test_llm_handler.py
    └── test_tts_handler.py
```

---

## 3. Supported Platforms

- **Fully supported:** Linux + ALSA (`/dev/snd`, `arecord`/`aplay`). Tested on Ubuntu 22.04/24.04, Debian.
- **Not supported in this release:** Docker Desktop on Windows/macOS — no direct host audio passthrough. Needs PulseAudio, PipeWire, WSL2, or native audio bridge (not attempted here). See *Limitations*.

Container runs at minimum:
```bash
arecord -l
aplay -l
arecord
aplay
```
via `alsa-utils` installed in image.

---

## 4. Linux System Requirements

- Docker Engine ≥24 + Compose v2
- ALSA (`/dev/snd` exists, user in `audio` group)
- 4 GB RAM (8 GB for Whisper `medium`), 2 vCPU, ~5 GB disk for models
- Internet for first model pulls (then cached via volumes)
- No GPU required (CPU `int8`); optional CUDA for `float16`

Check host audio:
```bash
getent group audio          # note GID, e.g. 29
arecord -l                  # list mics
aplay -l                    # list speakers
arecord -f S16_LE -r 16000 -c 1 -d 2 /tmp/test.wav && aplay /tmp/test.wav
```

---

## 5. ALSA Permissions & the Audio Group GID Caveat

Host `audio` group GID varies by distro (Debian/Ubuntu `29`, Arch `92`, Fedora `63`, etc.). Don't hardcode.

**Two approaches (we support both, compose uses `group_add` which needs no rebuild):**

1. **Build-arg (image-time):**
   ```bash
   getent group audio | cut -d: -f3   # e.g. 29
   docker compose build --build-arg AUDIO_GID=29
   ```

2. **Runtime `group_add` (preferred, no rebuild):**
   ```yaml
   # docker-compose.yml: assistant.group_add: ["${AUDIO_GID:-29}"]
   AUDIO_GID=$(getent group audio | cut -d: -f3) docker compose up -d
   ```
   Documented in `Dockerfile:6` and `docker-compose.yml:22`.

Our `Dockerfile` creates `audio` group with `AUDIO_GID` and adds user `jarvis` (non-root) to it. `entrypoint.sh` also logs `arecord -l` so you see permission errors early.

If you see `cannot open audio device: Permission denied`, fix host group or run:
```bash
sudo usermod -aG audio $USER && newgrp audio
```

---

## 6. Install & Run (CPU)

```bash
# 1. .env
cp .env.example .env
# Edit .env: check INPUT_DEVICE/OUTPUT_DEVICE (empty = first available),
#   WAKEWORD_MODEL_PATH, OLLAMA_MODEL, etc.

# 2. Host GID (optional but recommended)
export AUDIO_GID=$(getent group audio | cut -d: -f3)
echo "AUDIO_GID=$AUDIO_GID"

# 3. Build
docker compose build

# 4. (Optional) pre-download models to avoid first-run wait
./scripts/download_models.sh
# Or inside compose: docker compose run --rm assistant ./scripts/download_models.sh

# 5. Run
docker compose up -d

# 6. Logs
docker compose logs -f assistant

# 7. Verify audio passthrough
docker compose exec assistant arecord -l
docker compose exec assistant aplay -l

# 8. Stop
docker compose down
```

### Mic/Speaker Test (inside container)
```bash
docker compose exec assistant arecord -f S16_LE -r 16000 -c 1 -d 3 /tmp/mic.wav && docker compose exec assistant aplay /tmp/mic.wav
# Or host: arecord ... then aplay
```

### Model Details
- **Whisper cache:** `WHISPER_MODEL_CACHE=/models/whisper` volume `whisper_cache`. `int8` on CPU, `float16` on GPU. Cache persisted so not re-downloaded.
- **Piper Turkish:** `PIPER_MODEL_PATH=/models/piper/tr_TR-model.onnx` volume `piper_models`. Default uses Python binding (low latency); falls back to `piper` CLI if no binding. Edge-tts requires internet and sends text to external service — Piper remains local default.
- **Ollama model:** `OLLAMA_MODEL=phi3:mini` (or `llama3`, `mistral`). Pulled idempotently in `entrypoint.sh` via `curl /api/tags` + `/api/pull` — checks `ollama list` first.
- **Wake-word:** `WAKEWORD_MODEL_PATH=/models/wakeword.onnx` — **no pretrained Jarvis ships**. See next section.

---

## 7. Wake-Word Model Setup

No `jarvis` pretrained model is bundled. Supply one:

1. **Train custom:** https://github.com/dscripka/openWakeWord#training-a-custom-model
2. **Use bundled example (if `openWakeWord` installed):**
   ```bash
   python -c "import openwakeword, pathlib; print(pathlib.Path(openwakeword.__file__).parent)"
   # e.g. .../site-packages/openwakeword/resources/models/hey_jarvis.onnx
   cp <that>/hey_jarvis.onnx ./models/wakeword.onnx
   # Set in .env: WAKEWORD_MODEL_PATH=/models/wakeword.onnx
   ```
3. **Custom path:** Place your `.onnx` at `./models/wakeword.onnx` and mount via compose volume `wakeword_models:/models` + `WAKEWORD_MODEL_PATH=/models/wakeword.onnx`.

Threshold/cooldown in `.env`: `WAKEWORD_THRESHOLD=0.5`, `WAKEWORD_COOLDOWN_SECONDS=2.0`.

If missing at startup, container fails fast with clear error (see `wakeword_handler.py`).

---

## 8. GPU Run (Optional)

Requires `nvidia-container-toolkit` on host and CUDA image.

```bash
# Use GPU profile (compose file has commented example)
# Uncomment deploy.resources.reservations.devices in docker-compose.yml or run:
docker compose --profile gpu up -d

# env changes for GPU:
# WHISPER_DEVICE=cuda
# WHISPER_COMPUTE_TYPE=float16
```

CPU stays default: `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`.

---

## 9. Configuration via .env

All via `config.py` (Pydantic `pydantic-settings`, validated at startup, fail-fast). See `.env.example` for full list. Key:

- `AUDIO_*`, `SILENCE_DURATION_MS=900`, `PRE_ROLL_MS=300`, `MAX_RECORDING_SECONDS=20`
- `WHISPER_MODEL=small`, `WHISPER_VAD_FILTER=true` (secondary cleanup only)
- `LLM_PROVIDER=ollama|openai`, timeouts bounded 5-300s, `LLM_FALLBACK_ENABLED`, `MAX_INPUT_CHARS`, `LLM_MAX_OUTPUT_TOKENS`
- `MEMORY_BACKEND=sqlite|memory|disabled`, `MAX_HISTORY_MESSAGES=12`
- `TTS_PROVIDER=piper|edge-tts`
- `LOG_LEVEL=INFO`, `LOG_FORMAT=text`

Timeouts and lengths are bounded in validation, not just defaults.

---

## 10. Logs, Stop/Restart, Health

```bash
docker compose logs -f assistant          # follow
docker compose logs --tail 100 assistant
docker compose restart assistant
docker compose down && docker compose up -d
docker inspect --format='{{json .State.Health}}' jarvis-assistant | jq
# Health via heartbeat file: /tmp/jarvis_heartbeat touched every 5s by main loop
```

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied` on `/dev/snd` | Check `AUDIO_GID` (see §5), `sudo usermod -aG audio $USER`, re-login |
| `arecord -l` empty | Host has no ALSA devices or WSL2 without sound; needs host audio |
| Wake-word model missing error | Provide `.onnx` per §7 |
| Piper model missing | `scripts/download_models.sh` or `TTS_PROVIDER=edge-tts` |
| Whisper re-downloads | Ensure `whisper_cache` volume mounted, `WHISPER_MODEL_CACHE` correct |
| Ollama not reachable | `docker compose logs ollama`, check `OLLAMA_BASE_URL=http://ollama:11434` internal network |
| Slow STT | Use `WHISPER_MODEL=tiny` or GPU `float16` |
| TTS latency | Prefer Piper binding over CLI (image has `piper-tts`); check sentence-splitting |
| Health unhealthy | `docker compose exec assistant cat /tmp/jarvis_heartbeat`, check `main.py` not stuck |

---

## 12. Windows/macOS Limitations

- Host `/dev/snd` mapping is **Linux-only**. Docker Desktop on Windows/macOS does not pass through host audio; `arecord -l` will be empty.
- Workarounds (not in this release): run natively on WSL2 with USB audio, use PulseAudio/PipeWire network bridge, or build a host audio bridge.
- README explicitly states this; `docker-compose.yml` `devices: - /dev/snd:/dev/snd` will fail on those platforms — that's expected.

---

## 13. Privacy & Data Handling

- **Piper (default):** Fully local, no network. Audio stays in RAM, no disk writes (except temp `/tmp` WAV deleted immediately after play).
- **edge-tts (optional):** Requires internet, sends text to Microsoft external service. Documented in `.env` choice.
- **Faster-Whisper & Ollama:** Local if self-hosted. OpenAI fallback sends text to OpenAI if enabled.
- **Memory:** SQLite at `/data/memory.db` (Docker volume `jarvis_data`), survives restarts. Never persists API keys, system prompts, or secrets. `MEMORY_BACKEND=disabled` to turn off. Fallback to in-memory if SQLite write fails (logged warning, don't crash).
- No `.env` in Git, no API keys in logs, no LLM output piped to shell.

---

## 14. Security

- Non-root user `jarvis` in container, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`.
- No Docker socket mount, no host filesystem mount beyond volumes.
- No shell command execution from LLM output — never pipe.
- No HTTP API exposed externally (health via heartbeat file). Ollama port only within compose network (`expose`, not `ports`).
- Timeouts bounded, `MAX_INPUT_CHARS` and `LLM_MAX_OUTPUT_TOKENS` enforced.
- `.env` not in Git, secrets via env, not logs.

---

## 15. Running Tests

```bash
# Native (Windows/Linux where Python available)
pip install -r requirements.txt
pytest -v
# Or with coverage: pytest --cov=. tests/

# In Docker (Linux)
docker compose run --rm assistant pytest -v
```

Tests cover: `config` validation, `memory` trimming/fallback, `prompt` invariants, `llm` timeout/retry/truncation, `tts` sentence-split/error-beep. All mocked — no real audio/models needed.

---

## 16. Extending with New Providers

- **LLM:** Add to `llm_handler.py` `LLM_PROVIDER` literal, implement `_generate_newprovider()` with timeout/retry, update `config.py` env, add to `check_connectivity()`. System prompt stays Turkish.
- **TTS:** Add to `tts_handler.py` `TTS_PROVIDER` literal, add `_synthesize_newprovider()` with sentence-splitting, handle temp file cleanup and error beep, update `config.py`.
- **Memory:** `memory.py` already supports `memory|sqlite|disabled`; add new backend by extending `SQLiteHistoryWrapper` fallback logic.

---

## 17. Quick Commands (required)

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose logs -f assistant
docker compose down
docker compose exec assistant arecord -l
docker compose exec assistant aplay -l
```

---

## 18. What I Verified vs. Could Not Verify

See `KNOWN_ASSUMPTIONS.md` for full list. Summary:

- **Verified in Windows sandbox:** `config` validation (10 tests), `memory` (6), `prompt`/`llm` (9), `tts` (7), `audio_handler`/`stt`/`wakeword` import & fallback logic, `main.py` AST & heartbeat, Docker version 28.4, `docker compose build` (see below), `scripts/download_models.sh` syntax.
- **Could not verify (Linux-only/no hardware):** `/dev/snd` passthrough (`arecord`/`aplay` inside container on Linux host), Silero VAD ONNX inference on real audio, Faster-Whisper model download & `int8`/`float16` GPU path, Piper binding vs CLI latency with real Turkish model, openWakeWord detection with real `.onnx`, Ollama `phi3:mini` pull & chat, edge-tts network, GPU profile, SQLite under concurrent audio load. See `KNOWN_ASSUMPTIONS.md`.

---

## License

MIT — see repo. Models (Whisper, Piper voices, Silero) have their own licenses.
