"""tests/test_config.py — validate Pydantic settings."""
import os
import pytest
from config import Settings, reset_settings


def test_default_load():
    reset_settings()
    # ensure env clean for defaults
    for k in list(os.environ.keys()):
        if k.startswith("WAKEWORD_") or k.startswith("AUDIO_") or k.startswith("WHISPER_"):
            pass
    s = Settings(_env_file=None)  # ignore .env
    assert s.audio_sample_rate == 16000
    assert s.whisper_language == "tr"
    assert s.llm_provider == "ollama"
    assert s.tts_provider == "piper"
    assert s.wakeword_threshold == 0.5


def test_invalid_threshold():
    with pytest.raises(Exception):
        Settings(_env_file=None, wakeword_threshold=2.0)


def test_invalid_sample_width():
    with pytest.raises(Exception):
        Settings(_env_file=None, audio_sample_width=3)


def test_invalid_compute_type():
    with pytest.raises(Exception):
        Settings(_env_file=None, whisper_compute_type="invalid")


def test_timeouts_bounded():
    with pytest.raises(Exception):
        Settings(_env_file=None, ollama_timeout_seconds=9999)
    with pytest.raises(Exception):
        Settings(_env_file=None, max_recording_seconds=999)


def test_pre_roll_validation():
    with pytest.raises(Exception):
        Settings(_env_file=None, pre_roll_ms=6000, speech_start_timeout_seconds=5.0)


def test_url_validation():
    with pytest.raises(Exception):
        Settings(_env_file=None, ollama_base_url="ftp://bad")
    s = Settings(_env_file=None, ollama_base_url="http://ollama:11434/")
    assert s.ollama_base_url == "http://ollama:11434"  # trailing slash stripped


def test_env_override(monkeypatch):
    monkeypatch.setenv("AUDIO_SAMPLE_RATE", "22050")
    monkeypatch.setenv("WHISPER_LANGUAGE", "en")
    s = Settings(_env_file=None)
    assert s.audio_sample_rate == 22050
    assert s.whisper_language == "en"
    monkeypatch.delenv("AUDIO_SAMPLE_RATE")
    monkeypatch.delenv("WHISPER_LANGUAGE")


def test_max_input_chars_bounds():
    with pytest.raises(Exception):
        Settings(_env_file=None, max_input_chars=5)
    with pytest.raises(Exception):
        Settings(_env_file=None, max_input_chars=20000)


def test_log_level_validation():
    with pytest.raises(Exception):
        Settings(_env_file=None, log_level="VERBOSE")
