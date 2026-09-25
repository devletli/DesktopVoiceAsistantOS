"""
wakeword_handler.py — openWakeWord integration.

Rules:
- Never write raw audio to disk.
- Only short chunks in RAM (1280 samples = 80ms @16kHz).
- Cooldown to avoid repeated triggers.
- Fail fast with clear error if model file missing — no pretrained "Jarvis" ships.
- See README for how to supply/train a model.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from openwakeword.model import Model as OWWModel  # type: ignore
    HAS_OWW = True
except Exception as e:  # pragma: no cover
    OWWModel = None  # type: ignore
    HAS_OWW = False
    logger.debug(f"openwakeword not available: {e}")


class WakeWordError(RuntimeError):
    """Raised when wake-word model is missing or unusable."""


class WakeWordHandler:
    def __init__(self, settings=None):
        from config import get_settings
        self.settings = settings or get_settings()
        self.model_path = Path(self.settings.wakeword_model_path)
        self.threshold = self.settings.wakeword_threshold
        self.cooldown = self.settings.wakeword_cooldown_seconds
        self.sample_rate = self.settings.wakeword_sample_rate
        self.chunk_size = self.settings.wakeword_chunk_size

        self._last_trigger = 0.0
        self.model: Optional[OWWModel] = None
        self._load_model()

    def _load_model(self) -> None:
        if not HAS_OWW:
            raise WakeWordError(
                "openwakeword library not installed. Install with: pip install openwakeword==0.6.0\n"
                "If installed but import failed, check onnxruntime/dependencies."
            )
        if not self.model_path.exists():
            raise WakeWordError(
                f"Wake-word model not found at {self.model_path}\n"
                "No pretrained 'Jarvis' model ships by default. You must supply one:\n"
                "  1. Train custom model: https://github.com/dscripka/openWakeWord#training-a-custom-model\n"
                "  2. Use a bundled model (e.g. 'hey_jarvis' if available) and copy to /models/wakeword.onnx\n"
                "  3. Or set WAKEWORD_MODEL_PATH to an existing .onnx file\n"
                "  4. For Docker: place model at ./models/wakeword.onnx and mount via volume\n"
                f"Current setting: WAKEWORD_MODEL_PATH={self.model_path} (threshold={self.threshold})"
            )
        try:
            # Model expects list of paths; inference_framework onnx is default
            self.model = OWWModel(
                wakeword_models=[str(self.model_path)],
                inference_framework="onnx",
            )
            logger.info(f"Wake-word model loaded: {self.model_path} (threshold={self.threshold}, cooldown={self.cooldown}s)")
        except Exception as e:
            raise WakeWordError(f"Failed to load wake-word model {self.model_path}: {e}") from e

    def _is_cooldown(self) -> bool:
        return (time.monotonic() - self._last_trigger) < self.cooldown

    def detect(self, audio_bytes: bytes) -> bool:
        """
        Process a single chunk (bytes, 16-bit PCM mono).
        Returns True if wake word detected and not in cooldown.
        Never writes to disk; RAM only.
        """
        if self._is_cooldown():
            return False
        if self.model is None:
            return False

        # Convert bytes → int16 → expected format
        # openWakeWord expects 16-bit PCM 16kHz mono, chunk_size samples
        try:
            pcm = np.frombuffer(audio_bytes, dtype=np.int16)
            # If chunk length mismatches, pad/truncate
            if len(pcm) != self.chunk_size:
                if len(pcm) < self.chunk_size:
                    pcm = np.pad(pcm, (0, self.chunk_size - len(pcm)))
                else:
                    pcm = pcm[:self.chunk_size]

            # predict returns dict {model_name: score}
            scores = self.model.predict(pcm)  # type: ignore
            if not scores:
                return False
            # Get max score across models (usually one model)
            max_score = max(scores.values()) if isinstance(scores, dict) else float(scores)
            if max_score >= self.threshold:
                self._last_trigger = time.monotonic()
                logger.info(f"Wake word detected: score={max_score:.3f} >= {self.threshold}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Wake-word detection failed for chunk: {e}")
            return False

    def reset_cooldown(self) -> None:
        self._last_trigger = 0.0

    def close(self) -> None:
        # openWakeWord Model has no explicit close, but clear ref
        self.model = None
        logger.info("WakeWordHandler closed")
