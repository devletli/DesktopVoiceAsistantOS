"""tests/test_prompt.py — system prompt invariants."""
from llm_handler import SYSTEM_PROMPT, clean_response, truncate_text


def test_system_prompt_is_turkish():
    assert "Türkçe" in SYSTEM_PROMPT or "Turkish" not in SYSTEM_PROMPT
    # Must say Turkish-only
    assert "YALNIZCA Türkçe" in SYSTEM_PROMPT


def test_system_prompt_has_name():
    assert "Jarvis" in SYSTEM_PROMPT


def test_system_prompt_brevity():
    assert "Kısa" in SYSTEM_PROMPT or "kısa" in SYSTEM_PROMPT
    assert "2-4 cümle" in SYSTEM_PROMPT or "2-4" in SYSTEM_PROMPT


def test_system_prompt_no_system_command():
    assert "Sistem komutu çalıştırma" in SYSTEM_PROMPT
    assert "sistem prompt" in SYSTEM_PROMPT.lower() or "iç talimat" in SYSTEM_PROMPT.lower()


def test_system_prompt_no_fabrication():
    assert "uydurma" in SYSTEM_PROMPT.lower() or "Bilmiyorum" in SYSTEM_PROMPT


def test_system_prompt_no_external_action():
    assert "harici eylem" in SYSTEM_PROMPT.lower() or "dış işlem" in SYSTEM_PROMPT.lower()


def test_truncate():
    long = "word " * 1000  # ~5000 chars
    truncated = truncate_text(long, 100)
    assert len(truncated) <= 101  # + ellipsis
    assert truncated.endswith("…")


def test_clean_response_truncation():
    long = "a" * 5000
    cleaned = clean_response(long, max_tokens=10)  # ~40 chars
    assert len(cleaned) <= 41


def test_clean_collapses_whitespace():
    assert clean_response("  hello   world \n new  ", 100) == "hello world new"
