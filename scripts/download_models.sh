#!/bin/bash
set -euo pipefail

# scripts/download_models.sh — idempotent model bootstrap
# - Checks if models exist, downloads only if missing
# - Persists via Docker volumes so not re-downloaded on every start

MODELS_DIR="/models"
WHISPER_CACHE="${WHISPER_MODEL_CACHE:-/models/whisper}"
PIPER_DIR="/models/piper"
WAKEWORD_PATH="${WAKEWORD_MODEL_PATH:-/models/wakeword.onnx}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"
PIPER_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/seda/medium/tr_TR-seda-medium.onnx"
PIPER_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/seda/medium/tr_TR-seda-medium.onnx.json"
SILERO_URL="https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"

echo "[download] Models dir: $MODELS_DIR"
mkdir -p "$WHISPER_CACHE" "$PIPER_DIR" "$(dirname "$WAKEWORD_PATH")"

# ── Silero VAD ─────────────────────────────────────────────────
SILERO_PATH="/models/silero_vad.onnx"
if [ -f "$SILERO_PATH" ]; then
  echo "[download] Silero VAD already at $SILERO_PATH — skip"
else
  echo "[download] Fetching Silero VAD to $SILERO_PATH ..."
  if command -v curl >/dev/null; then
    curl -L -o "$SILERO_PATH" "$SILERO_URL" || echo "[download] Silero download failed (optional, fallback to energy VAD)"
  elif command -v wget >/dev/null; then
    wget -O "$SILERO_PATH" "$SILERO_URL" || echo "[download] Silero download failed"
  else
    echo "[download] No curl/wget — skip Silero (energy fallback will be used)"
  fi
fi

# ── Piper Turkish ──────────────────────────────────────────────
PIPER_MODEL="${PIPER_MODEL_PATH:-/models/piper/tr_TR-model.onnx}"
PIPER_CONFIG="${PIPER_CONFIG_PATH:-/models/piper/tr_TR-model.onnx.json}"

if [ -f "$PIPER_MODEL" ]; then
  echo "[download] Piper model already at $PIPER_MODEL — skip"
else
  echo "[download] Fetching Piper Turkish model to $PIPER_MODEL ..."
  mkdir -p "$(dirname "$PIPER_MODEL")"
  if command -v curl >/dev/null; then
    curl -L -o "$PIPER_MODEL" "$PIPER_VOICE_URL" || echo "[download] Piper model download failed — check URL"
    curl -L -o "$PIPER_CONFIG" "$PIPER_CONFIG_URL" || echo "[download] Piper config download failed"
  elif command -v wget >/dev/null; then
    wget -O "$PIPER_MODEL" "$PIPER_VOICE_URL" || echo "[download] Piper model download failed"
    wget -O "$PIPER_CONFIG" "$PIPER_CONFIG_URL" || echo "[download] Piper config download failed"
  fi
fi

# ── Wake-word ──────────────────────────────────────────────────
if [ -f "$WAKEWORD_PATH" ]; then
  echo "[download] Wake-word model already at $WAKEWORD_PATH — skip"
else
  echo "[download] WARNING: No wake-word model at $WAKEWORD_PATH"
  echo "  No pretrained 'Jarvis' ships by default."
  echo "  Options:"
  echo "    1. Train: https://github.com/dscripka/openWakeWord#training-a-custom-model"
  echo "    2. Copy bundled 'hey_jarvis' if available:"
  echo "       python -c \"import openwakeword; print(openwakeword.__file__)\" && cp .../hey_jarvis.onnx $WAKEWORD_PATH"
  echo "    3. Place your .onnx at $WAKEWORD_PATH and mount via compose volume"
fi

# ── Whisper ────────────────────────────────────────────────────
echo "[download] Whisper model '$WHISPER_MODEL' will be auto-downloaded by Faster-Whisper on first run to $WHISPER_CACHE"
echo "  To pre-download manually:"
echo "    python -c \"from faster_whisper import WhisperModel; WhisperModel('$WHISPER_MODEL', device='cpu', compute_type='int8', download_root='$WHISPER_CACHE')\""
if [ -d "$WHISPER_CACHE" ] && [ "$(ls -A "$WHISPER_CACHE" 2>/dev/null)" ]; then
  echo "[download] Whisper cache already populated at $WHISPER_CACHE"
else
  echo "[download] Whisper cache empty — will download on first transcription"
fi

# ── Ollama (handled in entrypoint.sh) ──────────────────────────
echo "[download] Ollama model '${OLLAMA_MODEL:-phi3:mini}' is pulled idempotently in entrypoint.sh via ollama API"

echo "[download] Done. Models:"
ls -lh "$MODELS_DIR" 2>&1 || true
ls -lh "$PIPER_DIR" 2>&1 || true
echo "[download] Check: arecord -l and aplay -l inside container"
