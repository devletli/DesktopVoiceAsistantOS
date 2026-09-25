"""
config.py — Pydantic Settings with startup validation for Jarvis Voice Assistant.

All configuration is loaded via environment variables / .env.
Validation fails fast with clear messages.

Audio format: 16kHz, mono, 16-bit PCM is the default but configurable.
Two-tier VAD: Silero VAD (real-time endpointing) + Whisper VAD filter (cleanup) are separate.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # ── Wake word ──────────────────────────────────────────────
    wakeword_provider: Literal["openwakeword"] = Field(default="openwakeword")
    wakeword_model_path: str = Field(default="/models/wakeword.onnx")
    wakeword_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    wakeword_cooldown_seconds: float = Field(default=2.0, ge=0.1, le=10.0)
    wakeword_sample_rate: int = Field(default=16000, ge=8000, le=48000)
    wakeword_chunk_size: int = Field(default=1280, ge=128, le=8192)

    # ── Audio / VAD ────────────────────────────────────────────
    audio_sample_rate: int = Field(default=16000, ge=8000, le=48000)
    audio_channels: int = Field(default=1, ge=1, le=2)
    audio_sample_width: int = Field(default=2)  # validated: 1,2,4
    audio_chunk_size: int = Field(default=1024, ge=256, le=8192)
    silence_duration_ms: int = Field(default=900, ge=100, le=5000)
    speech_start_timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    max_recording_seconds: float = Field(default=20.0, ge=5.0, le=60.0)
    pre_roll_ms: int = Field(default=300, ge=0, le=2000)
    input_device: Optional[str] = Field(default=None)  # None = first available
    output_device: Optional[str] = Field(default=None)

    # ── STT ────────────────────────────────────────────────────
    whisper_model: str = Field(default="small")
    whisper_device: Literal["cpu", "cuda"] = Field(default="cpu")
    whisper_compute_type: str = Field(default="int8")
    whisper_language: str = Field(default="tr")
    whisper_beam_size: int = Field(default=1, ge=1, le=10)
    whisper_vad_filter: bool = Field(default=True)
    whisper_min_silence_ms: int = Field(default=500, ge=0, le=2000)
    whisper_model_cache: str = Field(default="/models/whisper")

    # ── LLM ────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "openai"] = Field(default="ollama")
    ollama_base_url: str = Field(default="http://ollama:11434")
    ollama_model: str = Field(default="phi3:mini")
    ollama_timeout_seconds: int = Field(default=60, ge=5, le=300)
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    openai_timeout_seconds: int = Field(default=60, ge=5, le=300)
    llm_fallback_enabled: bool = Field(default=False)
    llm_max_output_tokens: int = Field(default=300, ge=10, le=2000)
    max_input_chars: int = Field(default=1000, ge=100, le=10000)

    # ── Memory ─────────────────────────────────────────────────
    memory_backend: Literal["sqlite", "memory", "disabled"] = Field(default="sqlite")
    memory_db_path: str = Field(default="/data/memory.db")
    session_id: str = Field(default="default")
    max_history_messages: int = Field(default=12, ge=0, le=100)

    # ── TTS ────────────────────────────────────────────────────
    tts_provider: Literal["piper", "edge-tts"] = Field(default="piper")
    piper_model_path: str = Field(default="/models/piper/tr_TR-model.onnx")
    piper_config_path: str = Field(default="/models/piper/tr_TR-model.onnx.json")
    piper_speaker: Optional[int] = Field(default=None)
    edge_tts_voice: str = Field(default="tr-TR-EmelNeural")

    # ── Observability ──────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_format: Literal["text", "json"] = Field(default="text")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Validators ─────────────────────────────────────────────
    @field_validator("audio_sample_width")
    @classmethod
    def validate_sample_width(cls, v: int) -> int:
        if v not in (1, 2, 4):
            raise ValueError("AUDIO_SAMPLE_WIDTH must be 1, 2, or 4")
        return v

    @field_validator("whisper_compute_type")
    @classmethod
    def validate_compute_type(cls, v: str) -> str:
        allowed = {"int8", "int8_float16", "int8_float32", "float16", "float32"}
        if v not in allowed:
            raise ValueError(f"WHISPER_COMPUTE_TYPE must be one of {allowed}")
        return v

    @field_validator("piper_speaker", mode="before")
    @classmethod
    def validate_piper_speaker(cls, v):
        # Empty string from .env (PIPER_SPEAKER=) should be treated as None
        if v == "" or v is None:
            return None
        try:
            return int(v)  # type: ignore
        except Exception:
            raise ValueError("PIPER_SPEAKER must be integer or empty")

    @field_validator("input_device", "output_device", "openai_api_key", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "":
            return None
        return v

    @field_validator("ollama_base_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must start with http:// or https://")
        return v.rstrip("/")

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "Settings":
        # If LLM provider is openai, api key should be present (warn, not hard fail — connectivity check will handle)
        # But enforce that int8 is used on cpu; if cuda, float16 is recommended (warn)
        if self.whisper_device == "cpu" and self.whisper_compute_type == "float16":
            logger.warning("WHISPER_DEVICE=cpu with float16 may be slow; int8 recommended on CPU")
        if self.pre_roll_ms >= self.speech_start_timeout_seconds * 1000:
            raise ValueError("PRE_ROLL_MS must be less than SPEECH_START_TIMEOUT_SECONDS*1000")
        if self.max_history_messages % 2 == 1:
            # odd number would trim asymmetrically; allow but log
            logger.warning("MAX_HISTORY_MESSAGES is odd; even number recommended for user/assistant pairs")
        return self


# Singleton accessor with lazy load (no heavy work at import time)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Load and validate settings. Fail fast with clear message."""
    global _settings
    if _settings is None:
        try:
            _settings = Settings()  # type: ignore[call-arg]
        except Exception as e:
            # Re-raise with context for startup fail-fast
            raise RuntimeError(f"Configuration validation failed: {e}") from e
    return _settings


def load_settings() -> Settings:
    """Alias for get_settings — used by tests and main."""
    return get_settings()


def reset_settings() -> None:
    """Reset cached settings (for testing)."""
    global _settings
    _settings = None


# Backwards-compat helper: validate on import if RUN_VALIDATION=1
if __name__ == "__main__":
    s = get_settings()
    print(s.model_dump_json(indent=2))
