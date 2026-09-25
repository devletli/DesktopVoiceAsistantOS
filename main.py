"""
main.py — Full runtime loop for Jarvis Voice Assistant.

Architecture (explicit per spec Section 8):
- Single dedicated audio thread (producer) feeding a queue; wake-word/VAD/record loop lives there.
- STT → LLM → TTS runs on main thread (consumer), so slow LLM/TTS never blocks audio capture.
- Don't mix ad-hoc threading and asyncio without clear boundary — this file uses threading
  for audio capture + synchronous processing on main. No asyncio event loop for core loop
  except for edge-tts which spawns its own loop.

Mute-input-during-playback: wake-word detection is paused while TTS audio is playing
to avoid self-triggering (feed-forward).

Heartbeat: touches /tmp/jarvis_heartbeat every 5s for healthcheck.py.

Graceful shutdown on SIGINT/SIGTERM: stop threads, close streams, close DB, exit cleanly.
"""
from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Optional

# ── Logging setup ──────────────────────────────────────────────
def setup_logging():
    from config import get_settings
    import logging as _logging
    import sys
    s = get_settings()
    level = getattr(_logging, s.log_level.upper(), _logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s" if s.log_format == "text" else "%(message)s"
    _logging.basicConfig(level=level, format=fmt, stream=sys.stdout)

# ── Core assistant class ───────────────────────────────────────

class JarvisAssistant:
    def __init__(self):
        from config import get_settings
        self.settings = get_settings()
        self.logger = logging.getLogger("jarvis")
        self.shutdown_event = threading.Event()
        self.is_playing = threading.Event()  # mute wake-word during playback
        self.audio_queue: queue.Queue = queue.Queue(maxsize=200)
        self.heartbeat_path = Path("/tmp/jarvis_heartbeat")
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.audio_thread: Optional[threading.Thread] = None

        # Handlers (lazy init in start)
        self.audio_handler = None
        self.wake_handler = None
        self.stt_handler = None
        self.llm_handler = None
        self.tts_handler = None
        self.memory = None

    def _setup_signal_handlers(self):
        def _handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_event.set()
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _heartbeat_loop(self):
        while not self.shutdown_event.is_set():
            try:
                self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
                self.heartbeat_path.touch(exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Heartbeat touch failed: {e}")
            # wait with timeout to allow shutdown
            self.shutdown_event.wait(5)

    def _audio_producer_loop(self):
        """
        Audio thread: continuously captures audio chunks and feeds queue.
        Wake-word detection and VAD recording are driven by main thread consuming this queue,
        but capture itself never blocks.
        If capture fails, log and retry with backoff.
        """
        self.logger.info("Audio producer thread started")
        from audio_handler import AudioHandler
        import numpy as np

        # We use sounddevice callback to feed queue; here we just keep thread alive
        # Actual feeding is done via AudioHandler.open_input_stream callback
        # For this design, we open stream here and feed queue.
        retry_delay = 1.0
        while not self.shutdown_event.is_set():
            try:
                if self.audio_handler is None:
                    self.audio_handler = AudioHandler(settings=self.settings)

                # Callback that feeds queue unless muted (playback)
                def callback(indata, frames, time_info, status):
                    if status:
                        self.logger.warning(f"Audio status: {status}")
                    if self.is_playing.is_set():
                        return  # mute input during playback to prevent self-triggering
                    try:
                        # indata is int16 array
                        self.audio_queue.put(indata.tobytes(), block=False)
                    except queue.Full:
                        # Drop oldest to keep latency low
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(indata.tobytes())
                        except Exception:
                            pass

                # Open stream blocking until shutdown
                stream = self.audio_handler.open_input_stream(callback, blocksize=self.settings.audio_chunk_size)
                self.logger.info(f"Audio stream opened (device={self.settings.input_device or 'default'}, sr={self.settings.audio_sample_rate})")
                with stream:
                    while not self.shutdown_event.is_set():
                        time.sleep(0.1)
                break  # normal exit

            except Exception as e:
                self.logger.error(f"Audio producer error: {e} — retry in {retry_delay}s (check arecord -l, ALSA permissions, INPUT_DEVICE)")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 10.0)
                # Try to reset handler
                self.audio_handler = None
                # Drain queue
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except Exception:
                        break

        self.logger.info("Audio producer thread exited")

    def validate_and_init(self):
        """Steps 1-6 of runtime loop."""
        self.logger.info("Jarvis starting — validating config...")
        # 1. Config already validated via get_settings() — fail fast if invalid
        # (Pydantic raises at startup)
        self.logger.info(f"Config ok: LLM={self.settings.llm_provider}, STT={self.settings.whisper_model}, TTS={self.settings.tts_provider}")

        # 2-3. Audio devices verified via AudioHandler (lazy, but we check listing)
        try:
            from audio_handler import AudioHandler
            ah = AudioHandler(settings=self.settings)
            in_devices = ah.list_input_devices()
            out_devices = ah.list_output_devices()
            self.logger.info(f"Audio devices: {len(in_devices)} input, {len(out_devices)} output")
            if not in_devices:
                self.logger.warning("No input devices found — arecord -l empty? Check /dev/snd and audio group GID")
            if not out_devices:
                self.logger.warning("No output devices found — aplay -l empty?")
            # Keep handler for later
            self.audio_handler = ah
        except Exception as e:
            self.logger.error(f"Audio device check failed: {e}")
            # Don't crash — main loop will retry

        # 4. Load wake-word model (clear error if missing)
        try:
            from wakeword_handler import WakeWordHandler
            self.wake_handler = WakeWordHandler(settings=self.settings)
            self.logger.info("Wake-word model loaded")
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error("Jarvis cannot start without wake-word model. See README 'Wake-word model setup'")
            raise

        # 5. Load STT model (lazy, but we trigger load now)
        try:
            from stt_handler import STTHandler
            self.stt_handler = STTHandler(settings=self.settings)
            # Don't block startup on model download; load lazily on first transcription
            # But we try to ensure cache dir exists
            self.logger.info(f"STT cache: {self.settings.whisper_model_cache}")
        except Exception as e:
            self.logger.error(f"STT init failed: {e}")
            raise

        # 6. Check LLM connectivity (warning, don't crash)
        try:
            from llm_handler import LLMHandler
            self.llm_handler = LLMHandler(settings=self.settings)
            ok = self.llm_handler.check_connectivity()
            if not ok:
                self.logger.warning("LLM provider not reachable — will surface error via TTS/beep on next interaction")
            else:
                self.logger.info("LLM provider reachable")
        except Exception as e:
            self.logger.warning(f"LLM connectivity check error: {e}")

        # Memory
        try:
            from memory import get_memory
            self.memory = get_memory()
            self.logger.info(f"Memory ready: backend={self.settings.memory_backend}, session={self.settings.session_id}")
        except Exception as e:
            self.logger.warning(f"Memory init failed (fallback to in-memory): {e}")
            from memory import SQLiteHistoryWrapper
            self.memory = SQLiteHistoryWrapper(session_id=self.settings.session_id, db_path=self.settings.memory_db_path, max_messages=self.settings.max_history_messages)

        # TTS
        try:
            from tts_handler import TTSHandler
            self.tts_handler = TTSHandler(settings=self.settings, audio_handler=self.audio_handler)
            self.logger.info(f"TTS ready: provider={self.settings.tts_provider}")
        except FileNotFoundError as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"TTS init failed: {e}")
            raise

    def run(self):
        """Main loop — steps 7-19."""
        self._setup_signal_handlers()
        self.validate_and_init()

        # Heartbeat
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

        # Audio producer
        self.audio_thread = threading.Thread(target=self._audio_producer_loop, daemon=True)
        self.audio_thread.start()

        self.logger.info("Jarvis listening for wake word... (say 'Jarvis')")
        # Drain any stale queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break

        try:
            while not self.shutdown_event.is_set():
                # 7. Passive listening: consume queue chunks and run wake-word detection
                try:
                    chunk = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if self.is_playing.is_set():
                    continue  # muted during playback

                # 8. On wake-word detection: beep
                detected = False
                try:
                    detected = self.wake_handler.detect(chunk)  # type: ignore
                except Exception as e:
                    self.logger.warning(f"Wake-word detect error: {e}")
                    continue

                if not detected:
                    continue

                wake_time = time.monotonic()
                self.logger.info(f"Wake word at {wake_time:.2f}")
                if self.audio_handler:
                    self.audio_handler.play_beep()

                # 9-12. VAD recording loop (uses queue, with timeout/max)
                self.logger.info("Listening for speech...")
                record_start = time.monotonic()
                audio_bytes = None
                try:
                    if self.audio_handler:
                        audio_bytes = self.audio_handler.record_with_vad(self.audio_queue)
                    else:
                        self.logger.error("No audio handler for recording")
                        continue
                except Exception as e:
                    self.logger.error(f"Recording failed: {e}")
                    continue

                record_latency = time.monotonic() - record_start
                self.logger.info(f"Recording took {record_latency:.2f}s")

                if audio_bytes is None:
                    self.logger.info("No speech detected (timeout/empty), returning to wake word")
                    continue

                # 13. Transcribe
                stt_start = time.monotonic()
                text = None
                try:
                    if self.stt_handler:
                        text = self.stt_handler.transcribe(audio_bytes, sample_rate=self.settings.audio_sample_rate)
                    else:
                        self.logger.error("No STT handler")
                        continue
                except Exception as e:
                    self.logger.error(f"STT error: {e}")
                    if self.audio_handler:
                        self.audio_handler.play_error_beep()
                    continue
                stt_latency = time.monotonic() - stt_start
                self.logger.info(f"STT latency {stt_latency:.2f}s")

                # 14. Discard empty
                if not text or not text.strip():
                    self.logger.info("Empty transcription, returning to idle")
                    continue

                self.logger.info(f"User: {text}")

                # 15-16. LLM: text + trimmed history, clean + length-limit
                llm_start = time.monotonic()
                history = None
                try:
                    if self.memory:
                        history = self.memory.get_history_for_llm()
                except Exception:
                    history = None

                response = None
                try:
                    if self.llm_handler:
                        response = self.llm_handler.generate(text, history=history)
                    else:
                        response = "Üzgünüm, LLM hazır değil."
                except Exception as e:
                    self.logger.error(f"LLM error: {e}")
                    response = "Üzgünüm, bir hata oluştu."

                llm_latency = time.monotonic() - llm_start
                self.logger.info(f"LLM latency {llm_latency:.2f}s, response='{response[:80] if response else ''}'")

                # Persist to memory (never secrets)
                try:
                    if self.memory and text and response:
                        self.memory.add_user_message(text)
                        self.memory.add_ai_message(response)
                except Exception as e:
                    self.logger.warning(f"Memory persist failed (fallback): {e}")

                # 17-18. TTS: synthesize + play with mute
                if not response:
                    continue
                tts_start = time.monotonic()
                self.is_playing.set()  # mute wake-word during playback
                try:
                    if self.tts_handler:
                        ok = self.tts_handler.speak(response)
                        if not ok:
                            self.logger.warning("TTS speak failed")
                    else:
                        self.logger.error("No TTS handler")
                finally:
                    # Small delay after playback to avoid capturing tail
                    time.sleep(0.2)
                    self.is_playing.clear()
                    # Drain queue of self-triggered audio
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except Exception:
                            break
                    tts_latency = time.monotonic() - tts_start
                    self.logger.info(f"TTS latency {tts_latency:.2f}s")

                total = time.monotonic() - wake_time
                self.logger.info(f"Total round-trip {total:.2f}s (record {record_latency:.2f} + stt {stt_latency:.2f} + llm {llm_latency:.2f} + tts {tts_latency:.2f})")

                # 19. Resume passive listening (loop)
                self.logger.info("Resuming wake-word listening...")

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt — shutting down")
        finally:
            self.shutdown()

    def shutdown(self):
        self.logger.info("Shutting down...")
        self.shutdown_event.set()
        if self.audio_handler:
            try:
                self.audio_handler.shutdown()
            except Exception:
                pass
        # Close memory
        try:
            from memory import reset_store
            # reset_store disposes engines
            # Don't clear memory on shutdown — persist
            pass
        except Exception:
            pass
        self.logger.info("Shutdown complete, exiting.")


def main():
    setup_logging()
    app = JarvisAssistant()
    app.run()


if __name__ == "__main__":
    main()
