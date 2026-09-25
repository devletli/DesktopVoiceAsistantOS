# Jarvis Voice Assistant — Agentic Build Brief

Act as an expert Python developer, DevOps engineer, and voice AI systems architect.

You are operating as an **autonomous coding agent with file system and shell access**
(not a chat model producing code blocks to be copy-pasted). Your job is to actually
**create, run, and verify** a working project — not just print files.

## 0. How you must work (read this first)

This is the most important section. Do not treat this as a "write everything in one
giant response" task. Instead:

1. **Work in milestones**, in the order given in Section 3. After each milestone:
   - Actually create the files on disk.
   - Run `pytest` for any tests that exist so far and fix failures before moving on.
   - Where relevant, run a quick smoke check (e.g. `python -c "import config; config.load()"`)
     rather than assuming the code works.
2. **Do not silently invent library versions.** Before pinning a version in
   `requirements.txt`, check what the currently available compatible version actually
   is (installed environment, PyPI, or your own tool access). If you cannot verify,
   pin a reasonable version and clearly flag it in a `KNOWN_ASSUMPTIONS.md` file so a
   human can double check it later — do not present guessed versions as verified facts.
3. **Stop and report a blocker** instead of hallucinating around it. Examples: no
   wake-word model file present, no GPU available, Ollama not reachable in your build
   sandbox. Report clearly what you could and couldn't verify.
4. At the end, produce a short **"What I verified vs. what I could not verify"**
   summary (e.g. "I could not actually test `/dev/snd` passthrough because this
   environment has no audio device").
5. Never truncate a file with `...` or "continued here" — if a file is too large for
   one write, write it in full across multiple sequential tool calls, not by omitting
   content.

## 1. Scope and platform

First release targets **Linux + ALSA only**. Use `/dev/snd` device mapping for
microphone/speaker access inside the container.

State explicitly in the README that Docker Desktop on Windows/macOS does not provide
direct host audio passthrough; those platforms would need PulseAudio, PipeWire, WSL2,
or a native host audio bridge — do not attempt to support them in this release.

The container must be able to run, at minimum:
```bash
arecord -l
aplay -l
arecord
aplay
```

Audio input/output device selection must be configurable via `.env`, with a sane
default (first available device) if unset.

## 2. Technology stack

- Python 3.11+
- Docker + Docker Compose
- Wake word: openWakeWord
- STT: Faster-Whisper
- Real-time speech endpointing (start/stop of recording): **Silero VAD** (via onnxruntime),
  run continuously on live audio chunks
- Faster-Whisper's built-in VAD filter: used **only** as a secondary cleanup pass at
  transcription time, to strip leading/trailing silence and reduce hallucinated
  transcriptions on near-silent buffers — it is not the mechanism that decides when
  to stop recording
- Default LLM: Ollama; alternative: OpenAI API
- Default TTS: Piper; alternative: edge-tts
- Conversation history: **modern** LangChain `RunnableWithMessageHistory` (never
  `ConversationChain` / `ConversationBufferMemory`)
- Persistent memory: SQLite
- Configuration: **Pydantic (`pydantic-settings`)**-based, with startup validation
- Audio format: 16 kHz, mono, 16-bit PCM
- Logging: Python `logging`
- Testing: pytest

> Why the VAD split matters: conflating "real-time endpointing" and "post-hoc VAD
> filtering" into one mechanism is a common bug source — it either cuts users off
> mid-sentence or never stops recording. Keep these as two separate concerns in code.

## 3. Build milestones (do these in order)

1. **Scaffold + config**: repo layout, `config.py` (Pydantic settings + validation),
   `.env.example`, `.gitignore`, `.dockerignore`. Write `tests/test_config.py` and
   make it pass before continuing.
2. **Audio I/O + real-time VAD**: `audio_handler.py` — device listing, input/output
   selection, beep playback, pre-roll buffer, Silero VAD-based recording loop
   (start-timeout, silence-duration stop, max-duration force stop), clean shutdown.
3. **Wake word**: `wakeword_handler.py` — openWakeWord integration, cooldown,
   threshold, clear error if the model file is missing (with README instructions on
   how to obtain/train one — do not assume a pretrained "Jarvis" model exists).
4. **STT**: `stt_handler.py` — Faster-Whisper load-from-cache, transcription,
   VAD-filter cleanup pass, empty/noise-only result filtering, latency logging.
5. **Memory**: `memory.py` — SQLite-backed message history via
   `RunnableWithMessageHistory`, message-count trimming, fallback to in-memory if
   SQLite write fails, and secrets/system-prompt never persisted. Write
   `tests/test_memory.py`.
6. **LLM**: `llm_handler.py` — provider abstraction (Ollama/OpenAI), timeout +
   limited retry, response length truncation, system prompt enforcing Turkish,
   brevity, no system-command execution, no fabrication. Write `tests/test_prompt.py`.
7. **TTS**: `tts_handler.py` — Piper as default, prefer the **Piper Python binding**
   over spawning a CLI subprocess per utterance if a usable binding is available
   (subprocess spawn overhead hurts latency); sentence-splitting for long responses;
   edge-tts as optional fallback with network timeout; error beep on failure.
8. **Wiring**: `main.py` — the full loop (Section 4), asyncio/thread architecture
   explicitly documented (see Section 8), mute-input-during-playback to prevent
   self-triggering, graceful shutdown on SIGINT/SIGTERM.
9. **Docker**: `Dockerfile`, `docker-compose.yml`, `entrypoint.sh`,
   `healthcheck.py`, `scripts/download_models.sh` (Section 9).
10. **README + final pass**: full documentation (Section 10), then a final
    `pytest` run and a `docker compose build` sanity check.

## 4. Core runtime loop

1. Validate configuration on startup (fail fast with a clear message).
2. Open the audio input device.
3. Verify the audio output device.
4. Load the wake-word model (clear error if missing).
5. Load or retrieve the Faster-Whisper model from cache.
6. Check LLM provider connectivity (log a warning, don't crash, if unreachable and
   fallback is disabled — surface the error to the user via TTS/beep on next
   interaction attempt).
7. Start passively listening for the wake word.
8. On detection: play a short beep/chime.
9. Wait for speech start via Silero VAD (timeout → return to step 7).
10. Buffer speech audio in RAM.
11. Stop when silence persists for `SILENCE_DURATION_MS`.
12. Force-stop at `MAX_RECORDING_SECONDS`.
13. Transcribe with Faster-Whisper (default language: Turkish, configurable).
14. Discard empty/meaningless transcriptions, return to step 7.
15. Send text + trimmed history to the LLM.
16. Clean and length-limit the response.
17. Synthesize speech (Piper, or edge-tts as configured).
18. Play audio; **mute/pause wake-word processing during playback** to avoid
    self-triggering.
19. Resume passive listening.
20. On Ctrl+C/SIGTERM: stop all threads/streams, close DB connections, exit cleanly.

## 5. Wake word

`.env` settings:
```env
WAKEWORD_PROVIDER=openwakeword
WAKEWORD_MODEL_PATH=/models/wakeword.onnx
WAKEWORD_THRESHOLD=0.5
WAKEWORD_COOLDOWN_SECONDS=2.0
WAKEWORD_SAMPLE_RATE=16000
WAKEWORD_CHUNK_SIZE=1280
```

Rules:
- Never write raw audio to disk during this phase.
- Never run continuous recording/transcription here — only short chunks in RAM.
- Apply cooldown to avoid repeated triggers.
- If the model file is missing at startup, fail with a clear, actionable error and
  document in the README how to supply one (no pretrained "Jarvis" model ships by
  default).

## 6. Audio and VAD

`audio_handler.py` responsibilities: device I/O (PyAudio/sounddevice over ALSA), WAV/PCM
playback, beep playback, device listing/selection, clear mic-error reporting, buffer
cleanup, graceful shutdown.

`.env` settings:
```env
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_SAMPLE_WIDTH=2
AUDIO_CHUNK_SIZE=1024
SILENCE_DURATION_MS=900
SPEECH_START_TIMEOUT_SECONDS=5
MAX_RECORDING_SECONDS=20
PRE_ROLL_MS=300
INPUT_DEVICE=
OUTPUT_DEVICE=
```

Use a pre-roll buffer so the first word isn't clipped. Do not stop recording on short
in-speech pauses, but never exceed `MAX_RECORDING_SECONDS`. Keep audio in RAM; if a
temp file is unavoidable, create it under `/tmp` and delete it immediately after use.

## 7. STT

`.env` settings:
```env
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=tr
WHISPER_BEAM_SIZE=1
WHISPER_VAD_FILTER=true
WHISPER_MIN_SILENCE_MS=500
WHISPER_MODEL_CACHE=/models/whisper
```

Use `int8` on CPU; use `float16` (or the appropriate CUDA compute type) on GPU —
document both paths separately in the README. Persist the model cache via a Docker
volume so it isn't re-downloaded on every start. Log STT latency; filter empty/noise
results; report model-load errors clearly.

## 8. LLM provider

```env
LLM_PROVIDER=ollama

OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=phi3:mini
OLLAMA_TIMEOUT_SECONDS=60

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=60

LLM_FALLBACK_ENABLED=false
LLM_MAX_OUTPUT_TOKENS=300
MAX_INPUT_CHARS=1000
```

Validate provider connectivity at startup with a clear error if unreachable. No
automatic fallback unless explicitly enabled. System prompt must specify: the
assistant's name, Turkish-only responses, short/natural answers, no unnecessary
explanations, no system command execution, no external actions unless explicitly
requested, no fabrication. Truncate overly long responses before TTS.

**Concurrency note**: use a single dedicated audio thread (producer) feeding a
queue; the wake-word/VAD/record loop lives there. STT → LLM → TTS processing runs
on the main thread/asyncio loop as a consumer, so a slow LLM/TTS call never blocks
audio capture. Document this explicitly in code comments — don't mix ad-hoc
threading and asyncio without a clear boundary.

## 9. Memory

```env
MEMORY_BACKEND=sqlite
MEMORY_DB_PATH=/data/memory.db
SESSION_ID=default
MAX_HISTORY_MESSAGES=12
```

Persist on a Docker volume; survive restarts; fall back to in-memory (with a logged
warning) if SQLite writes fail rather than crashing; never persist API keys, system
prompts, or other secrets; allow memory to be disabled entirely via `.env`.

## 10. TTS

```env
TTS_PROVIDER=piper
PIPER_MODEL_PATH=/models/piper/tr_TR-model.onnx
PIPER_CONFIG_PATH=/models/piper/tr_TR-model.onnx.json
PIPER_SPEAKER=
EDGE_TTS_VOICE=tr-TR-EmelNeural
```

Convert to WAV/PCM, play, delete temp files, split long responses into sentences,
play an error beep on failure, clear error if the Piper model is missing, apply a
network timeout for edge-tts. State in the README that edge-tts requires internet
and sends text to an external service; Piper remains the fully local default.
Prefer the Piper Python binding over CLI subprocess invocation where feasible, to
avoid per-utterance process-spawn latency; fall back to the CLI binary if no usable
binding is installed.

## 11. Docker

**Dockerfile**: Python 3.11-slim base; ALSA/audio deps (`ffmpeg`, `alsa-utils`,
`libasound2`, `libsndfile1`, build essentials as needed); non-root user;
`PYTHONDONTWRITEBYTECODE=1`; `PYTHONUNBUFFERED=1`; healthcheck; deterministic
installs (pinned `requirements.txt`, verified versions — see Section 0.2); minimal
image footprint.

**Non-root + audio device access**: host `audio` group GID varies by distro, so do
not hardcode it into the image. Either:
- accept an `AUDIO_GID` build-arg (documented in the README, defaulting to a common
  value like 29 but with instructions to check `getent group audio` on the host), or
- add the audio group at container start via `group_add` in `docker-compose.yml`
  using a host-provided GID.
Document whichever approach you choose and why.

**Healthcheck**: the assistant does not expose an HTTP API (see Section 12), so the
healthcheck must not depend on one. Implement `healthcheck.py` to check a heartbeat
file (e.g. `/tmp/jarvis_heartbeat`) that the main loop touches periodically, and
have the Docker `HEALTHCHECK` instruction run this script rather than curling an
endpoint.

**docker-compose.yml**: `assistant` + `ollama` services; `/dev/snd:/dev/snd`;
`/root/.ollama` volume; Whisper cache volume; Piper/wake-word model volumes; SQLite
data volume; `restart: unless-stopped`; Ollama healthcheck; assistant's
`depends_on: ollama: condition: service_healthy`; CPU-only by default; an optional
GPU profile using `deploy.resources.reservations.devices` with
`capabilities: [gpu]`.

**Model bootstrap**: a separate init step that checks (`ollama list`) whether
`OLLAMA_MODEL` is already present and only pulls it if missing — idempotent, no
re-download on every restart.

State clearly in the README that host audio device mapping is Linux-only.

## 12. Security

- No root user in the container.
- No Docker socket mount, no host filesystem mount.
- No shell command execution capability; never pipe LLM output to a shell.
- No API keys in logs; no sensitive user text in debug logs.
- No `.env` in Git.
- No HTTP API exposed externally by default (healthcheck uses the heartbeat-file
  approach above, not an endpoint).
- Ollama's port reachable only within the compose-internal network.
- All timeouts bounded (enforce sane upper limits in config validation, not just
  defaults).
- Enforce `MAX_INPUT_CHARS` on transcribed text and `LLM_MAX_OUTPUT_TOKENS` /
  response-length truncation on LLM output.

## 13. Observability and error handling

```env
LOG_LEVEL=INFO
LOG_FORMAT=text
```

Log durations for: wake-word detection, recording, STT, LLM, TTS, and total
round-trip latency. Implement bounded retries for Ollama connectivity and TTS
network calls, and controlled reconnection on audio device errors. Shut down
cleanly on: Ctrl+C, SIGTERM, audio stream exceptions, Ollama connection loss, model
load errors, TTS exceptions.

## 14. File structure

```text
jarvis-assistant/
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
    ├── test_llm_handler.py   # mocked provider, timeout/retry/truncation behavior
    └── test_tts_handler.py   # mocked synth, sentence-splitting, error-beep path
```

`KNOWN_ASSUMPTIONS.md` is new: use it to record anything you could not actually
verify in your build environment (unverified library versions, untested audio
passthrough, untested GPU path, etc.) so a human reviewer knows what to double-check.

## 15. Code quality

Type hints throughout; Pydantic-based configuration with startup validation; small,
testable functions/classes; minimal global state; explicit justification in comments
for any thread/asyncio boundary; audio capture kept separate from LLM/TTS
processing; lazy model loading (no heavy loads at import time); no bare
`except Exception: pass`; clear user-facing error messages; Turkish or English
inline comments where they aid clarity; pinned, **verified** versions in
`requirements.txt`; a lock file if your tooling supports one; unit tests for each
major component, including the two added above.

## 16. README content

Must cover: purpose; supported platforms; Linux system requirements; Docker/Compose
install; ALSA permissions and the `audio` group (including the GID caveat from
Section 11); mic/speaker test commands; wake-word model setup; Whisper cache setup;
Piper Turkish model install; Ollama model download; `.env` creation; CPU run; GPU
run; build/run commands; logs; stop/restart; troubleshooting; Windows/macOS
limitations; privacy/data-handling description; security warnings; running tests;
extending with new providers.

Must explicitly show:
```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose logs -f assistant
docker compose down
docker compose exec assistant arecord -l
docker compose exec assistant aplay -l
```

## 17. Final deliverable order

1. Brief architectural description (including the audio-thread/asyncio boundary
   and the two-tier VAD design).
2. Directory structure.
3. Full content of each file, created via actual file-write tool calls — not just
   printed in chat.
4. Installation/execution steps.
5. CPU vs. GPU usage notes.
6. Known limitations (including anything unverifiable in your build environment —
   cross-reference `KNOWN_ASSUMPTIONS.md`).
7. Steps required before production use.
8. A final summary of what you actually ran/tested vs. what you could only
   reason about statically.

## 18. Baseline assumptions to state before you start coding

- Target platform: Linux + ALSA only for this release.
- Default LLM: Ollama; default STT: Faster-Whisper; default TTS: Piper.
- Real-time endpointing uses Silero VAD; Whisper's VAD filter is a secondary
  cleanup step only.
- No pretrained "Jarvis" wake-word model ships by default — the user must supply
  or train one.
- Container audio device access depends on host user/group permissions, which vary
  by distro — you must document, not hardcode, this.
- The Ollama model may be pulled on first run; this must be idempotent.
- Library versions in `requirements.txt` will be verified at implementation time,
  not assumed from memory.