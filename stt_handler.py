"""
stt_handler.py — Faster-Whisper STT with secondary VAD cleanup.

Two-tier VAD reminder:
- Silero VAD (audio_handler.py) = real-time endpointing (when to stop recording).
- Whisper VAD filter here = secondary cleanup pass at transcription time — strips
  leading/trailing silence and reduces hallucinations on near-silent buffers.
  It is NOT the mechanism that decides when to stop recording.

Load-from-cache: model is cached at WHISPER_MODEL_CACHE and persisted via Docker volume.
"""
from __future__ import annotations

import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel  # type: ignore
    HAS_FASTER_WHISPER = True
except Exception as e:  # pragma: no cover
    WhisperModel = None  # type: ignore
    HAS_FASTER_WHISPER = False
    logger.debug(f"faster-whisper not available: {e}")

# Hallucinated / noise-only patterns to filter
_NOISE_PATTERNS = [
    r"^\s*$",
    r"^\[.*\]$",  # [Music], [ Silence ], etc.
    r"^\(.*\)$",
    r"^\W+$",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)

# Common Whisper hallucination phrases in Turkish/English silence
_HALLUCINATIONS = {
    "teşekkürler",
    "teşekkür ederim",
    "thank you",
    "you",
    "music",
    "silence",
    "applause",
    "laughs",
}


class STTHandler:
    def __init__(self, settings=None):
        from config import get_settings
        self.settings = settings or get_settings()
        self.model_name = self.settings.whisper_model
        self.device = self.settings.whisper_device
        self.compute_type = self.settings.whisper_compute_type
        self.language = self.settings.whisper_language
        self.beam_size = self.settings.whisper_beam_size
        self.vad_filter = self.settings.whisper_vad_filter
        self.min_silence_ms = self.settings.whisper_min_silence_ms
        self.cache_dir = Path(self.settings.whisper_model_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[WhisperModel] = None  # type: ignore
        self._loaded = False

    def load_model(self) -> None:
        """Lazy load model from cache; idempotent."""
        if self._loaded and self.model is not None:
            return
        if not HAS_FASTER_WHISPER:
            raise RuntimeError(
                "faster-whisper not installed. Install with: pip install faster-whisper==1.2.1\n"
                "In Docker, it is installed via requirements.txt"
            )
        try:
            logger.info(
                f"Loading Faster-Whisper model '{self.model_name}' "
                f"(device={self.device}, compute_type={self.compute_type}, "
                f"cache={self.cache_dir}, vad_filter={self.vad_filter})"
            )
            start = time.monotonic()
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.cache_dir),
            )
            elapsed = time.monotonic() - start
            logger.info(f"Whisper model loaded in {elapsed:.2f}s")
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{self.model_name}': {e}")
            raise RuntimeError(f"Whisper model load failed for '{self.model_name}': {e}") from e

    def _is_empty_or_noise(self, text: str) -> bool:
        if not text or not text.strip():
            return True
        t = text.strip()
        if _NOISE_RE.match(t):
            return True
        # single-word hallucinations
        if t.lower().strip(" .!?,") in _HALLUCINATIONS and len(t.split()) <= 2:
            # but allow if original audio was not near-silent — we already filtered via VAD
            # Here we treat very short hallucination as noise
            return True
        # Very short without meaningful chars
        if len(t) < 2:
            return True
        return False

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe PCM bytes (16-bit mono) to Turkish text.
        Returns cleaned text or None if empty/noise filtered.
        Logs latency.
        """
        if not audio_bytes or len(audio_bytes) < 1000:
            logger.debug("Audio too short to transcribe")
            return None

        if self.model is None:
            self.load_model()

        # Convert int16 bytes -> float32 [-1,1]
        audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Use temp file only if faster-whisper requires file path (it can take ndarray directly)
        # Prefer in-memory ndarray; fallback to temp wav if needed
        start = time.monotonic()
        try:
            # faster-whisper transcribe can take numpy array directly
            segments, info = self.model.transcribe(  # type: ignore
                audio_f32,
                language=self.language if self.language else None,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                vad_parameters={"min_silence_duration_ms": self.min_silence_ms} if self.vad_filter else None,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            latency = time.monotonic() - start
            logger.info(f"STT latency={latency:.2f}s, language={info.language if hasattr(info, 'language') else self.language}, text='{text[:80]}'")
        except Exception as e:
            latency = time.monotonic() - start
            logger.error(f"Transcription failed after {latency:.2f}s: {e}")
            return None

        # Filter empty/noise
        if self._is_empty_or_noise(text):
            logger.info(f"Discarding empty/noise transcription: '{text}'")
            return None

        # Truncate to MAX_INPUT_CHARS (enforced again in llm_handler, but early)
        from config import get_settings
        max_chars = get_settings().max_input_chars
        if len(text) > max_chars:
            logger.warning(f"Transcription too long ({len(text)} > {max_chars}), truncating")
            text = text[:max_chars]

        return text

    def transcribe_file(self, wav_path: str) -> Optional[str]:
        """Transcribe a WAV file path (convenience for testing)."""
        if self.model is None:
            self.load_model()
        start = time.monotonic()
        try:
            segments, info = self.model.transcribe(  # type: ignore
                wav_path,
                language=self.language if self.language else None,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                vad_parameters={"min_silence_duration_ms": self.min_silence_ms} if self.vad_filter else None,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            latency = time.monotonic() - start
            logger.info(f"STT file latency={latency:.2f}s, text='{text[:80]}'")
            if self._is_empty_or_noise(text):
                return None
            return text
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return None

    def unload(self) -> None:
        self.model = None
        self._loaded = False
        logger.info("STT model unloaded")
