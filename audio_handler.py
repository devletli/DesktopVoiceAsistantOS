"""
audio_handler.py — Audio I/O + real-time Silero VAD endpointing.

Responsibilities:
- Device listing / selection (sounddevice over ALSA)
- Beep/chime playback
- Pre-roll buffer (so first word isn't clipped)
- Silero VAD-based recording loop:
    * wait for speech start (timeout → return None)
    * buffer speech in RAM
    * stop on SILENCE_DURATION_MS of silence
    * force-stop at MAX_RECORDING_SECONDS
- Clean shutdown, no temp files (RAM only, /tmp only if unavoidable)

Two-tier VAD design (IMPORTANT):
- Silero VAD here is REAL-TIME endpointing — decides when to stop recording.
- Faster-Whisper's VAD filter (in stt_handler.py) is SECONDARY cleanup only — strips
  leading/trailing silence at transcription time. They are separate concerns.

Concurrency note (see main.py):
- This module owns the single dedicated audio thread (producer) feeding a queue.
- STT → LLM → TTS runs on main/asyncio loop (consumer). Never block audio capture.
"""
from __future__ import annotations

import collections
import logging
import math
import queue
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd  # type: ignore
    HAS_SOUNDDEVICE = True
except Exception as e:  # pragma: no cover
    logger.warning(f"sounddevice not available: {e}")
    sd = None  # type: ignore
    HAS_SOUNDDEVICE = False

try:
    import onnxruntime as ort  # type: ignore
    HAS_ONNX = True
except Exception:
    ort = None  # type: ignore
    HAS_ONNX = False

# ── Silero VAD ─────────────────────────────────────────────────

class SileroVAD:
    """
    Silero VAD via onnxruntime. Falls back to energy-based VAD if model missing.

    Model expected: silero_vad.onnx exported from https://github.com/snakers4/silero-vad
    Input: 512 samples @16kHz (32ms), or 256/1024 depending on version.
    We support 512 and 1024 by trimming/padding.
    """
    def __init__(self, model_path: str = "/models/silero_vad.onnx", threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model_path = Path(model_path)
        self.session: Optional[Any] = None
        self.h = np.zeros((2, 1, 128), dtype=np.float32)
        self.c = np.zeros((2, 1, 128), dtype=np.float32)
        self._use_onnx = False

        if HAS_ONNX and self.model_path.exists():
            try:
                providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(str(self.model_path), providers=providers)
                self._use_onnx = True
                logger.info(f"Silero VAD loaded: {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load Silero VAD ONNX {self.model_path}: {e} — falling back to energy VAD")
        else:
            if not HAS_ONNX:
                logger.warning("onnxruntime not available — using energy-based VAD fallback")
            elif not self.model_path.exists():
                logger.warning(f"Silero VAD model not found at {self.model_path} — using energy-based VAD fallback. "
                               "Run scripts/download_models.sh to fetch it.")
            # energy fallback

    def reset(self):
        self.h = np.zeros((2, 1, 128), dtype=np.float32)
        self.c = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        audio_chunk: np.ndarray float32, shape (N,), range [-1, 1]
        Returns True if speech detected.
        """
        if self._use_onnx and self.session is not None:
            try:
                # Silero expects 512 samples for 16k
                chunk = audio_chunk.astype(np.float32)
                if len(chunk) < 512:
                    chunk = np.pad(chunk, (0, 512 - len(chunk)))
                elif len(chunk) > 512:
                    chunk = chunk[:512]
                # Add batch dim: (1, 512)
                inp = chunk[np.newaxis, :]
                sr = np.array(self.sample_rate, dtype=np.int64)
                ort_inputs = {
                    "input": inp,
                    "h": self.h,
                    "c": self.c,
                    "sr": sr,
                }
                # Some exports use different input names; try to adapt
                # If session input names differ, map by order
                input_names = [i.name for i in self.session.get_inputs()]
                feed = {}
                for name in input_names:
                    if name == "input" and "input" in ort_inputs:
                        feed[name] = ort_inputs["input"]
                    elif name == "h" and "h" in ort_inputs:
                        feed[name] = ort_inputs["h"]
                    elif name == "c" and "c" in ort_inputs:
                        feed[name] = ort_inputs["c"]
                    elif name == "sr" and "sr" in ort_inputs:
                        feed[name] = ort_inputs["sr"]
                # Fallback: if names mismatch, use positional
                if len(feed) != len(input_names):
                    # try simple mapping by index
                    vals = [inp, self.h, self.c]
                    feed = {n: v for n, v in zip(input_names, vals)}
                out = self.session.run(None, feed)
                # Output 0 is probability, 1 & 2 are updated h,c if present
                prob = float(out[0].squeeze())
                if len(out) >= 3:
                    self.h = out[1]
                    self.c = out[2]
                return prob > self.threshold
            except Exception as e:
                logger.debug(f"Silero ONNX inference failed: {e} — using energy fallback for this chunk")
                # fall through to energy
        # Energy fallback: RMS > threshold
        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2) + 1e-10))
        # threshold ~0.02 corresponds to ~ -34dB; tuned for 16-bit PCM
        energy_threshold = 0.02
        return rms > energy_threshold

# ── AudioHandler ───────────────────────────────────────────────

class AudioHandler:
    def __init__(self, settings=None):
        from config import get_settings
        self.settings = settings or get_settings()
        self.sample_rate = self.settings.audio_sample_rate
        self.channels = self.settings.audio_channels
        self.sample_width = self.settings.audio_sample_width
        self.chunk_size = self.settings.audio_chunk_size
        self.silence_ms = self.settings.silence_duration_ms
        self.start_timeout = self.settings.speech_start_timeout_seconds
        self.max_seconds = self.settings.max_recording_seconds
        self.pre_roll_ms = self.settings.pre_roll_ms
        self.input_device = self.settings.input_device
        self.output_device = self.settings.output_device

        self.vad = SileroVAD(threshold=0.5, sample_rate=self.sample_rate)
        self._stop_event = threading.Event()
        self._stream: Optional[Any] = None

        # Pre-roll buffer size in chunks
        pre_roll_chunks = max(1, int(self.pre_roll_ms / 1000 * self.sample_rate / self.chunk_size))
        self.pre_roll_buffer: collections.deque = collections.deque(maxlen=pre_roll_chunks)
        logger.info(f"AudioHandler init: {self.sample_rate}Hz, pre_roll_chunks={pre_roll_chunks}")

    # ── Device listing ─────────────────────────────────────────
    def list_devices(self) -> List[Dict[str, Any]]:
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice unavailable — cannot list devices")
            return []
        try:
            devices = sd.query_devices()  # type: ignore
            return [{"index": i, "name": d["name"], "max_input_channels": d["max_input_channels"],
                     "max_output_channels": d["max_output_channels"], "default_samplerate": d["default_samplerate"]}
                    for i, d in enumerate(devices)]
        except Exception as e:
            logger.error(f"Failed to query audio devices: {e}")
            return []

    def list_input_devices(self) -> List[Dict[str, Any]]:
        return [d for d in self.list_devices() if d["max_input_channels"] > 0]

    def list_output_devices(self) -> List[Dict[str, Any]]:
        return [d for d in self.list_devices() if d["max_output_channels"] > 0]

    def _resolve_device(self, device: Optional[str], kind: str) -> Optional[int]:
        """Resolve device string to index. None → first available. Supports index int string or name substring."""
        devices = self.list_devices()
        if not devices:
            logger.warning(f"No {kind} devices found — using default (None)")
            return None
        if device is None or device.strip() == "":
            # first available
            for d in devices:
                if kind == "input" and d["max_input_channels"] > 0:
                    return d["index"]
                if kind == "output" and d["max_output_channels"] > 0:
                    return d["index"]
            return None
        # try as integer index
        try:
            idx = int(device)
            return idx
        except ValueError:
            pass
        # try substring match
        for d in devices:
            if device.lower() in d["name"].lower():
                if kind == "input" and d["max_input_channels"] > 0:
                    return d["index"]
                if kind == "output" and d["max_output_channels"] > 0:
                    return d["index"]
        logger.warning(f"Device '{device}' not found for {kind}, using default")
        return None

    # ── Beep ───────────────────────────────────────────────────
    def _generate_beep(self, freq: float = 880.0, duration_ms: int = 180) -> np.ndarray:
        """Generate 16-bit PCM beep at freq."""
        n = int(self.sample_rate * duration_ms / 1000)
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        # sine with fade in/out to avoid click
        wave = 0.3 * np.sin(2 * math.pi * freq * t)
        fade = int(0.01 * self.sample_rate)
        wave[:fade] *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)
        # to int16
        pcm = (wave * 32767).astype(np.int16)
        return pcm

    def play_beep(self, freq: float = 880.0, duration_ms: int = 180) -> None:
        """Play beep via output device; fallback to logging if no output."""
        try:
            pcm = self._generate_beep(freq, duration_ms)
            if not HAS_SOUNDDEVICE:
                logger.info(f"[beep] {freq}Hz {duration_ms}ms (sounddevice unavailable, logged only)")
                return
            device = self._resolve_device(self.output_device, "output")
            # sounddevice expects float32 [-1,1]
            float_data = pcm.astype(np.float32) / 32768.0
            sd.play(float_data, samplerate=self.sample_rate, device=device)  # type: ignore
            sd.wait()  # type: ignore
            logger.debug(f"Beep played: {freq}Hz")
        except Exception as e:
            logger.warning(f"Beep playback failed: {e}")

    def play_error_beep(self) -> None:
        self.play_beep(freq=220.0, duration_ms=400)

    def play_wav(self, wav_path: str) -> None:
        """Play WAV/PCM file; used by TTS handler."""
        try:
            import soundfile as sf  # type: ignore
            import sounddevice as sd2  # type: ignore
            data, sr = sf.read(wav_path, dtype="float32")
            device = self._resolve_device(self.output_device, "output")
            sd2.play(data, samplerate=sr, device=device)
            sd2.wait()
        except ImportError:
            logger.warning("soundfile not available, falling back to aplay")
            import subprocess
            try:
                subprocess.run(["aplay", wav_path], check=True, timeout=10)
            except Exception as e:
                logger.error(f"WAV playback failed: {e}")
        except Exception as e:
            logger.error(f"WAV playback failed for {wav_path}: {e}")

    # ── Recording loop with VAD ────────────────────────────────
    def record_with_vad(self, stream_queue: Optional[queue.Queue] = None) -> Optional[bytes]:
        """
        Blocking VAD recording loop.
        - Waits for speech start within SPEECH_START_TIMEOUT_SECONDS.
        - Buffers audio in RAM (pre-roll + speech).
        - Stops after SILENCE_DURATION_MS of silence or MAX_RECORDING_SECONDS.
        Returns bytes (16-bit PCM mono) or None if timeout/no speech.
        If stream_queue provided, audio chunks are fed from queue (used by audio thread).
        Otherwise opens a temporary input stream directly (simpler for tests).
        """
        # This method is designed to be called from the audio producer thread
        # or main thread for testing; it handles its own stream if no queue.
        if stream_queue is not None:
            return self._record_from_queue(stream_queue)
        else:
            return self._record_from_mic()

    def _record_from_queue(self, q: queue.Queue) -> Optional[bytes]:
        """Consume chunks from an externally-fed queue (audio thread)."""
        pre_roll = collections.deque(maxlen=self.pre_roll_buffer.maxlen)
        buffer: List[bytes] = []
        speech_started = False
        silence_start: Optional[float] = None
        start_wait = time.monotonic()
        record_start: Optional[float] = None
        self.vad.reset()

        while not self._stop_event.is_set():
            try:
                chunk = q.get(timeout=0.1)
            except queue.Empty:
                chunk = None

            now = time.monotonic()

            # Timeout waiting for speech start
            if not speech_started and (now - start_wait) > self.start_timeout:
                logger.info("Speech start timeout — returning to wake word")
                return None

            # Force stop at max duration
            if speech_started and record_start is not None and (now - record_start) > self.max_seconds:
                logger.info(f"Max recording duration {self.max_seconds}s reached — force stop")
                break

            if chunk is None:
                continue

            # chunk is bytes (int16 PCM)
            # Convert to float for VAD
            audio_f32 = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

            # Pre-roll buffering before speech
            if not speech_started:
                pre_roll.append(chunk)
                if self.vad.is_speech(audio_f32):
                    speech_started = True
                    record_start = now
                    # Flush pre-roll into buffer
                    buffer.extend(list(pre_roll))
                    buffer.append(chunk)
                    silence_start = None
                    logger.debug("Speech started (VAD)")
                continue

            # Already in speech — buffer and check silence
            buffer.append(chunk)
            if self.vad.is_speech(audio_f32):
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = now
                elif (now - silence_start) * 1000 >= self.silence_ms:
                    logger.info(f"Silence {self.silence_ms}ms detected — stopping recording")
                    break

        if not speech_started or not buffer:
            return None
        return b"".join(buffer)

    def _record_from_mic(self) -> Optional[bytes]:
        """Direct mic recording (used when no queue/thread). Falls back gracefully if no device."""
        if not HAS_SOUNDDEVICE:
            logger.error("Cannot record: sounddevice unavailable and no queue provided")
            return None
        device = self._resolve_device(self.input_device, "input")
        if device is None:
            logger.error("No input device available")
            return None

        q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):  # type: ignore
            if status:
                logger.warning(f"Audio callback status: {status}")
            # indata is float32 or int16 depending on dtype; we request int16
            q.put(indata.tobytes())

        try:
            with sd.InputStream(  # type: ignore
                device=device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype="int16",
                callback=callback,
            ):
                return self._record_from_queue(q)
        except Exception as e:
            logger.error(f"Microphone open failed (device={device}): {e} — check ALSA permissions and INPUT_DEVICE")
            return None

    # ── Stream helpers for wake-word phase ─────────────────────
    def open_input_stream(self, callback, blocksize: Optional[int] = None) -> Any:
        """Open a sounddevice InputStream for wake-word detection. Caller must close."""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice not available — cannot open input stream")
        device = self._resolve_device(self.input_device, "input")
        if device is None:
            raise RuntimeError("No input device found — check arecord -l and INPUT_DEVICE env")
        return sd.InputStream(  # type: ignore
            device=device,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=blocksize or self.chunk_size,
            dtype="int16",
            callback=callback,
        )

    def shutdown(self) -> None:
        self._stop_event.set()
        logger.info("AudioHandler shutdown")

    def reset_stop(self) -> None:
        self._stop_event.clear()
