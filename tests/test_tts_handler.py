"""tests/test_tts_handler.py — mocked synth, sentence-splitting, error-beep."""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from tts_handler import split_sentences, TTSHandler
from config import Settings


def make_settings(provider="piper", model_path="/tmp/fake.onnx"):
    return Settings(
        _env_file=None,
        tts_provider=provider,
        piper_model_path=model_path,
        piper_config_path=model_path + ".json",
        piper_speaker=None,
        edge_tts_voice="tr-TR-EmelNeural",
    )


def test_split_sentences_basic():
    assert split_sentences("Merhaba. Nasılsın? İyiyim!") == ["Merhaba.", "Nasılsın?", "İyiyim!"]
    assert split_sentences("Tek cümle") == ["Tek cümle"]
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_long_comma():
    long = "Bu çok uzun bir cümle, virgüllerle dolu, ve test için, birden fazla parçaya bölünmeli, çünkü TTS için uygun değil, ama yine de akıcı olmalı"
    long = (long + " ") * 3
    sentences = split_sentences(long)
    assert len(sentences) > 1
    for s in sentences:
        assert len(s) < 250


def test_tts_missing_model_raises():
    settings = make_settings(provider="piper", model_path="/nonexistent/model.onnx")
    with pytest.raises(FileNotFoundError, match="Piper model not found"):
        TTSHandler(settings=settings)


def test_synthesize_piper_binding_mock():
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "model.onnx"
        config = Path(tmp) / "model.onnx.json"
        model.write_bytes(b"fake")
        config.write_text("{}")
        settings = make_settings(provider="piper", model_path=str(model))
        settings.piper_config_path = str(config)
        with patch("tts_handler.HAS_PIPER_BINDING", True):
            handler = TTSHandler(settings=settings)
            handler._piper_voice = MagicMock()
            with patch.object(handler, "_synthesize_piper_binding") as mock_synth:
                def side_effect(text, path):
                    Path(path).write_bytes(b"RIFFfake")
                mock_synth.side_effect = side_effect
                wav = handler.synthesize("Merhaba dunya")
                assert wav is not None
                assert Path(wav).exists()
                Path(wav).unlink(missing_ok=True)
            # Also test cli fallback path
            handler._piper_voice = None
            with patch.object(handler, "_synthesize_piper_cli") as mock_cli:
                mock_cli.side_effect = lambda t, p: Path(p).write_bytes(b"RIFFfake")
                wav2 = handler.synthesize("Merhaba again")
                assert wav2 is not None
                Path(wav2).unlink(missing_ok=True)


def test_synthesize_edge_tts_mock():
    settings = make_settings(provider="edge-tts", model_path="/tmp/fake.onnx")
    with patch.object(TTSHandler, "_init_piper", return_value=None), \
         patch("tts_handler.HAS_EDGE_TTS", True):
        handler = TTSHandler(settings=settings)
        async def fake_save(path):
            Path(path).write_bytes(b"RIFFfake")
        with patch("tts_handler.edge_tts.Communicate") as MockComm:
            mock_comm = MagicMock()
            mock_comm.save = fake_save
            MockComm.return_value = mock_comm
            wav = handler.synthesize("Merhaba")
            assert wav is not None
            assert Path(wav).exists()
            Path(wav).unlink(missing_ok=True)


def test_speak_calls_synthesize_per_sentence():
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "model.onnx"
        model.write_bytes(b"fake")
        config = Path(tmp) / "model.onnx.json"
        config.write_text("{}")
        settings = make_settings(provider="piper", model_path=str(model))
        settings.piper_config_path = str(config)
        handler = TTSHandler(settings=settings)
        with patch.object(handler, "synthesize", return_value=None) as mock_synth:
            mock_audio = MagicMock()
            handler.audio_handler = mock_audio
            result = handler.speak("Hello world. Second sentence.")
            assert mock_synth.call_count == 2
            # overall success false because synthesize returned None
            assert result is False


def test_empty_text_no_synth():
    settings = make_settings(provider="edge-tts")
    with patch.object(TTSHandler, "_init_piper", return_value=None):
        handler = TTSHandler(settings=settings)
        assert handler.synthesize("") is None
        assert handler.synthesize("   ") is None
