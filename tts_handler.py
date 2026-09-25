"""
tts_handler.py — Piper (default, local) + edge-tts (fallback, network).

- Prefer Piper Python binding over CLI subprocess per utterance (latency)
- Sentence-splitting for long responses
- Edge-tts with network timeout (external service, requires internet)
- Error beep on failure, clear error if Piper model missing
- Temp files under /tmp, deleted immediately after use
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Try Piper binding
try:
    from piper.voice import PiperVoice  # type: ignore
    HAS_PIPER_BINDING = True
except Exception:
    PiperVoice = None  # type: ignore
    HAS_PIPER_BINDING = False

try:
    import edge_tts  # type: ignore
    HAS_EDGE_TTS = True
except Exception:
    import types as _types
    edge_tts = _types.ModuleType("edge_tts")  # dummy for patching in tests
    class _DummyCommunicate:
        def __init__(self, *a, **kw): pass
        async def save(self, path): 
            Path(path).write_bytes(b"RIFF")
    edge_tts.Communicate = _DummyCommunicate  # type: ignore
    HAS_EDGE_TTS = False


def split_sentences(text: str) -> List[str]:
    """
    Split long responses into sentences for incremental TTS.
    Keeps delimiters, handles Turkish: splits on .!? + whitespace
    """
    if not text.strip():
        return []
    # Don't split too aggressively; keep at least 10 chars per sentence
    # Use regex to split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    # Filter empty and very short fragments that are likely artifacts
    sentences = [p.strip() for p in parts if p.strip()]
    # If single long sentence > 300 chars, split by commas as well
    if len(sentences) == 1 and len(sentences[0]) > 300:
        sub = re.split(r",\s+", sentences[0])
        # Recombine to avoid too tiny chunks
        combined: List[str] = []
        buf = ""
        for piece in sub:
            if len(buf) + len(piece) < 200:
                buf = f"{buf}, {piece}" if buf else piece
            else:
                if buf:
                    combined.append(buf.strip(" ,"))
                buf = piece
        if buf:
            combined.append(buf.strip(" ,"))
        if combined:
            return combined
    return sentences


class TTSHandler:
    def __init__(self, settings=None, audio_handler=None):
        from config import get_settings
        self.settings = settings or get_settings()
        self.provider = self.settings.tts_provider
        self.piper_model = Path(self.settings.piper_model_path)
        self.piper_config = Path(self.settings.piper_config_path)
        self.piper_speaker = self.settings.piper_speaker
        self.edge_voice = self.settings.edge_tts_voice
        self.audio_handler = audio_handler  # optional, for playback & beep

        self._piper_voice: Optional[PiperVoice] = None  # type: ignore

        if self.provider == "piper":
            self._init_piper()

    def _init_piper(self) -> None:
        if not self.piper_model.exists():
            raise FileNotFoundError(
                f"Piper model not found at {self.piper_model}\n"
                f"Config expected at {self.piper_config}\n"
                "Download Turkish model:\n"
                "  mkdir -p /models/piper && \\\n"
                "  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/...\n"
                "  (see scripts/download_models.sh or README)\n"
                "Or set TTS_PROVIDER=edge-tts for network TTS."
            )
        if not self.piper_config.exists():
            logger.warning(f"Piper config not found at {self.piper_config}, but model exists — trying anyway")

        if HAS_PIPER_BINDING:
            try:
                logger.info(f"Loading Piper voice via Python binding: {self.piper_model}")
                self._piper_voice = PiperVoice.load(str(self.piper_model), config_path=str(self.piper_config) if self.piper_config.exists() else None)
                logger.info("Piper binding loaded")
            except Exception as e:
                logger.warning(f"Piper binding load failed: {e} — will fallback to CLI")
                self._piper_voice = None
        else:
            logger.info("Piper Python binding not available — will use CLI subprocess (piper binary required)")

    def _synthesize_piper_binding(self, text: str, wav_path: str) -> None:
        """Use Python binding (preferred, low latency)."""
        if self._piper_voice is None:
            raise RuntimeError("Piper binding not loaded")
        # PiperVoice.synthesize writes to wav file
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)  # Piper default, or use voice config
            # Synthesize — PiperVoice may have synthesize_wav or similar
            # Try both APIs
            try:
                # New API: voice.synthesize(text, wav_file)
                self._piper_voice.synthesize(text, wf)  # type: ignore
            except TypeError:
                # Old API: synthesize returns audio chunks
                for chunk in self._piper_voice.synthesize(text):  # type: ignore
                    wf.writeframes(chunk.audio_int16_bytes)  # type: ignore

    def _synthesize_piper_cli(self, text: str, wav_path: str) -> None:
        """Fallback to CLI subprocess."""
        cmd = ["piper", "--model", str(self.piper_model), "--output_file", wav_path]
        if self.piper_speaker is not None:
            cmd.extend(["--speaker", str(self.piper_speaker)])
        if self.piper_config.exists():
            cmd.extend(["--config", str(self.piper_config)])
        # piper reads text from stdin
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Piper CLI failed: {result.stderr.decode(errors='ignore')}")

    async def _synthesize_edge_tts(self, text: str, wav_path: str, timeout: float = 15.0) -> None:
        """Edge-tts with network timeout."""
        if not HAS_EDGE_TTS:
            raise RuntimeError("edge-tts not installed. Install with: pip install edge-tts==7.2.8")
        communicate = edge_tts.Communicate(text, self.edge_voice)  # type: ignore
        # Use asyncio.wait_for for timeout
        try:
            await asyncio.wait_for(communicate.save(wav_path), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"edge-tts timeout after {timeout}s — check internet")
        except Exception as e:
            raise RuntimeError(f"edge-tts failed: {e}") from e

    def synthesize(self, text: str, wav_path: Optional[str] = None) -> Optional[str]:
        """
        Synthesize text to WAV. Returns wav_path or None on failure.
        Handles sentence splitting for long responses — concatenates via temp files then merges?
        Simpler: synthesize each sentence to separate wav and return first (for testing) or play each sequentially
        Here we synthesize full text at once for simplicity, but split for playback.
        """
        if not text.strip():
            return None

        # Create temp wav if not provided (use system temp dir for cross-platform)
        if wav_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_path = tmp.name
            tmp.close()

        sentences = split_sentences(text)
        logger.info(f"TTS synthesizing {len(sentences)} sentence(s) via {self.provider}")

        try:
            if self.provider == "piper":
                # For long responses, synthesize sentence-by-sentence and concatenate?
                # For now, synthesize joined text; sentence splitting is used in speak() for incremental playback
                full_text = " ".join(sentences)
                if self._piper_voice is not None and HAS_PIPER_BINDING:
                    self._synthesize_piper_binding(full_text, wav_path)
                else:
                    self._synthesize_piper_cli(full_text, wav_path)
                return wav_path

            elif self.provider == "edge-tts":
                full_text = " ".join(sentences)
                # edge-tts is async
                try:
                    asyncio.run(self._synthesize_edge_tts(full_text, wav_path))
                except RuntimeError as e:
                    # If already in event loop (e.g., tests), use new loop
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(self._synthesize_edge_tts(full_text, wav_path))
                    finally:
                        loop.close()
                return wav_path

            else:
                raise ValueError(f"Unknown TTS_PROVIDER: {self.provider}")

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"TTS synthesis failed ({self.provider}): {e}")
            # Error beep
            if self.audio_handler is not None:
                try:
                    self.audio_handler.play_error_beep()
                except Exception:
                    pass
            else:
                # try lazy import beep
                try:
                    from audio_handler import AudioHandler
                    AudioHandler().play_error_beep()
                except Exception:
                    pass
            # Clean up temp file on failure
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def speak(self, text: str) -> bool:
        """
        Synthesize and play. Mutes wake-word during playback (caller must handle).
        Returns True on success, False on failure (error beep already played).
        Temp file deleted after play.
        """
        if not text.strip():
            return False

        sentences = split_sentences(text)
        overall_success = True

        for sentence in sentences:
            wav_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                wav_path = tmp.name
                tmp.close()

                result = self.synthesize(sentence, wav_path)
                if result is None or not Path(result).exists():
                    overall_success = False
                    continue

                # Play via audio_handler or fallback
                if self.audio_handler is not None and hasattr(self.audio_handler, "play_wav"):
                    self.audio_handler.play_wav(result)
                else:
                    # Fallback to aplay or sounddevice
                    try:
                        from audio_handler import AudioHandler
                        AudioHandler().play_wav(result)
                    except Exception as e:
                        logger.warning(f"Playback failed: {e}")
                        # Try aplay directly
                        try:
                            subprocess.run(["aplay", result], timeout=10, check=False)
                        except Exception:
                            pass

            except FileNotFoundError as e:
                logger.error(str(e))
                overall_success = False
                break
            except Exception as e:
                logger.error(f"Speak failed for sentence '{sentence[:30]}': {e}")
                overall_success = False
            finally:
                if wav_path:
                    try:
                        Path(wav_path).unlink(missing_ok=True)
                    except Exception:
                        pass

        return overall_success
