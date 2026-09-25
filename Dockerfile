# Dockerfile — Jarvis Voice Assistant
# Python 3.11-slim, ALSA/audio deps, non-root, deterministic installs
FROM python:3.11-slim

ARG AUDIO_GID=29
# AUDIO_GID: host `audio` group GID varies by distro (Debian/Ubuntu 29, Arch 92, etc.)
# Check host with: getent group audio | cut -d: -f3
# Pass at build: docker compose build --build-arg AUDIO_GID=$(getent group audio | cut -d: -f3)
# Or use group_add in compose (preferred, no rebuild).

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    alsa-utils \
    libasound2 \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Audio group ──────────────────────────────────────────────
# Create audio group with host GID so non-root user can access /dev/snd
RUN groupadd -r -g ${AUDIO_GID} audio || true

# ── Non-root user ────────────────────────────────────────────
RUN useradd -m -u 1000 -G audio jarvis \
    && mkdir -p /app /models /data /models/whisper /models/piper \
    && chown -R jarvis:jarvis /app /models /data

WORKDIR /app

# ── Python deps (deterministic) ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import openwakeword.utils; openwakeword.utils.download_models()" \
    && ls -lh /usr/local/lib/python3.11/site-packages/openwakeword/resources/models/ || true \
    && cp /usr/local/lib/python3.11/site-packages/openwakeword/resources/models/silero_vad.onnx /models/silero_vad.onnx \
    && chown jarvis:jarvis /models/silero_vad.onnx \
    && ls -lh /models/silero_vad.onnx

# ── App code ─────────────────────────────────────────────────
COPY --chown=jarvis:jarvis . .

# Ensure entrypoint and scripts are executable
RUN chmod +x entrypoint.sh scripts/download_models.sh || true \
    && chown -R jarvis:jarvis /app

USER jarvis

# ── Healthcheck (heartbeat file, no HTTP) ────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "main.py"]
