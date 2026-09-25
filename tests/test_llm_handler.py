"""tests/test_llm_handler.py — mocked provider, timeout/retry/truncation."""
from unittest.mock import patch, MagicMock
import pytest
from config import Settings
from llm_handler import LLMHandler, SYSTEM_PROMPT


def make_settings(provider="ollama", fallback=False):
    return Settings(
        _env_file=None,
        llm_provider=provider,
        ollama_base_url="http://ollama:11434",
        ollama_model="phi3:mini",
        ollama_timeout_seconds=60,
        openai_api_key="sk-test" if fallback else None,
        openai_model="gpt-4o-mini",
        openai_timeout_seconds=60,
        llm_fallback_enabled=fallback,
        llm_max_output_tokens=20,
        max_input_chars=100,
        max_history_messages=12,
    )


def test_generate_truncates_input():
    settings = make_settings()
    handler = LLMHandler(settings=settings)
    long_input = "x" * 200  # >100
    with patch.object(handler, "_generate_ollama", return_value="ok") as mock_gen:
        handler.generate(long_input)
        called_messages = mock_gen.call_args[0][0]
        user_msg = called_messages[-1]["content"]
        assert len(user_msg) <= 101  # 100 + ellipsis
        assert "x" in user_msg


def test_generate_truncates_output():
    settings = make_settings()
    handler = LLMHandler(settings=settings)
    long_output = "y" * 500  # 20 tokens ~80 chars, so should truncate
    with patch.object(handler, "_generate_ollama", return_value=long_output):
        result = handler.generate("hello")
        assert len(result) <= 81


def test_retry_and_fallback(monkeypatch):
    settings = make_settings(provider="ollama", fallback=True)
    handler = LLMHandler(settings=settings)
    # First ollama fails, fallback to openai succeeds
    with patch.object(handler, "_generate_ollama", side_effect=Exception("ollama down")) as mock_ollama, \
         patch.object(handler, "_generate_openai", return_value="fallback ok") as mock_openai:
        result = handler.generate("hello")
        assert result == "fallback ok"
        assert mock_ollama.call_count >= 1
        assert mock_openai.call_count >= 1


def test_timeout_retry_returns_error_message():
    settings = make_settings(provider="ollama", fallback=False)
    handler = LLMHandler(settings=settings)
    with patch.object(handler, "_generate_ollama", side_effect=Exception("timeout")):
        result = handler.generate("hello")
        # Should return Turkish error message, not raise
        assert "Üzgünüm" in result


def test_system_prompt_in_messages():
    settings = make_settings()
    handler = LLMHandler(settings=settings)
    with patch.object(handler, "_generate_ollama", return_value="ok") as mock:
        handler.generate("test")
        msgs = mock.call_args[0][0]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SYSTEM_PROMPT


def test_history_trimming():
    settings = make_settings()
    settings.max_history_messages = 2
    handler = LLMHandler(settings=settings)
    from langchain_core.messages import HumanMessage, AIMessage
    history = [HumanMessage(content=f"msg {i}") for i in range(10)]
    with patch.object(handler, "_generate_ollama", return_value="ok") as mock:
        handler.generate("hello", history=history)
        msgs = mock.call_args[0][0]
        # system + 2 history + user = 4
        assert len(msgs) == 4
        # Should be last 2 history items
        assert "msg 8" in msgs[1]["content"]
        assert "msg 9" in msgs[2]["content"]
