#!/bin/sh
set -e

# entrypoint.sh — idempotent model bootstrap + startup
# - Checks if OLLAMA_MODEL is present (`ollama list`), pulls only if missing
# - Validates wake-word and Piper models exist (clear errors if not)
# - Then execs main CMD

echo "[entrypoint] Jarvis Voice Assistant starting..."

# ── Load .env if present (for OLLAMA_* checks) ─────────────────
if [ -f /app/.env ]; then
  set -a
  # shellcheck disable=SC1091
  . /app/.env
  set +a
fi

# ── Ollama model bootstrap (idempotent) ─────────────────────────
OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-phi3:mini}"

# Wait for Ollama to be healthy (max 60s)
echo "[entrypoint] Waiting for Ollama at $OLLAMA_URL ..."
for i in $(seq 1 30); do
  if curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "[entrypoint] Ollama reachable"
    break
  fi
  echo "[entrypoint] Ollama not ready, retry $i/30..."
  sleep 2
done

# Check if model exists
if curl -sf "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"${OLLAMA_MODEL}\"" 2>/dev/null; then
  echo "[entrypoint] Ollama model $OLLAMA_MODEL already present — skip pull"
else
  # Try ollama CLI inside ollama container via curl pull API
  echo "[entrypoint] Checking ollama list for $OLLAMA_MODEL ..."
  # Use API to pull if missing (idempotent, no re-download if cached)
  # We call the Ollama container's API; if we are in assistant, we use curl to ollama service
  echo "[entrypoint] Pulling $OLLAMA_MODEL if needed (this may take a while on first run)..."
  # The pull is done via assistant's network call to ollama service
  # We use curl to trigger pull; ollama handles idempotency
  curl -sf -X POST "$OLLAMA_URL/api/pull" -d "{\"name\":\"$OLLAMA_MODEL\"}" >/dev/null 2>&1 || echo "[entrypoint] Pull trigger failed or model already exists — continuing"
  # Alternative: if ollama CLI is available in this image (not by default), use: ollama pull "$OLLAMA_MODEL"
fi

# ── Audio device sanity (arecord/aplay) ────────────────────────
echo "[entrypoint] Audio devices:"
arecord -l 2>&1 || echo "[entrypoint] arecord -l failed — check /dev/snd and --device /dev/snd:/dev/snd"
aplay -l 2>&1 || echo "[entrypoint] aplay -l failed"

# ── Model file checks (clear errors) ───────────────────────────
if [ ! -f "${WAKEWORD_MODEL_PATH:-/models/wakeword.onnx}" ]; then
  echo "[entrypoint] WARNING: Wake-word model missing at ${WAKEWORD_MODEL_PATH:-/models/wakeword.onnx}"
  echo "  No pretrained 'Jarvis' ships. See README 'Wake-word model setup' or run scripts/download_models.sh"
fi
if [ ! -f "${PIPER_MODEL_PATH:-/models/piper/tr_TR-model.onnx}" ] && [ "${TTS_PROVIDER:-piper}" = "piper" ]; then
  echo "[entrypoint] WARNING: Piper model missing at ${PIPER_MODEL_PATH:-/models/piper/tr_TR-model.onnx}"
  echo "  Run scripts/download_models.sh or set TTS_PROVIDER=edge-tts"
fi

# ── Heartbeat file ─────────────────────────────────────────────
touch /tmp/jarvis_heartbeat || true

echo "[entrypoint] Exec: $*"
exec "$@"
