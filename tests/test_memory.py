"""tests/test_memory.py — memory trimming, fallback, no secrets."""
import tempfile
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from config import Settings, reset_settings
from memory import SQLiteHistoryWrapper, reset_store, get_memory


def make_settings(db_path, max_messages=12, backend="sqlite"):
    return Settings(
        _env_file=None,
        memory_backend=backend,
        memory_db_path=db_path,
        max_history_messages=max_messages,
        session_id="test",
    )


def test_add_and_trim_memory():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "mem.db")
    s = make_settings(db, max_messages=4)
    with patch("config.get_settings", return_value=s):
        reset_store()
        mem = SQLiteHistoryWrapper(session_id="test", db_path=db, max_messages=4)
        try:
            for i in range(6):
                mem.add_user_message(f"hello {i}")
                mem.add_ai_message(f"reply {i}")
            assert len(mem.messages) == 4
            assert "hello 5" in mem.messages[-2].content or "reply 5" in mem.messages[-1].content
        finally:
            mem.close()
            reset_store()
    # cleanup after dispose
    try:
        os.unlink(db)
        os.rmdir(tmp)
    except Exception:
        pass


def test_fallback_on_write_failure():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "mem.db")
    s = make_settings(db, max_messages=12)
    with patch("config.get_settings", return_value=s):
        reset_store()
        mem = SQLiteHistoryWrapper(session_id="test", db_path=db, max_messages=12)
        try:
            if mem._sql_history is not None:
                with patch.object(mem._sql_history, "add_message", side_effect=Exception("disk full")):
                    mem.add_user_message("should fallback")
                    assert len(mem._fallback.messages) >= 1
                    assert mem._use_sql is False
            else:
                mem.add_user_message("fallback direct")
                assert len(mem.messages) == 1
        finally:
            mem.close()
            reset_store()
    try:
        if os.path.exists(db):
            os.unlink(db)
        os.rmdir(tmp)
    except Exception:
        pass


def test_never_persist_secrets():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "mem.db")
    s = make_settings(db)
    with patch("config.get_settings", return_value=s):
        reset_store()
        mem = SQLiteHistoryWrapper(session_id="test", db_path=db, max_messages=12)
        try:
            mem.add_user_message("my api_key is sk-12345 secret")
            assert len(mem.messages) == 0
            mem.add_user_message("normal message")
            assert len(mem.messages) == 1
        finally:
            mem.close()
            reset_store()
    try:
        if os.path.exists(db):
            os.unlink(db)
        os.rmdir(tmp)
    except Exception:
        pass


def test_disabled_backend():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "mem.db")
    s = make_settings(db, backend="disabled")
    with patch("config.get_settings", return_value=s):
        reset_store()
        mem = SQLiteHistoryWrapper(session_id="test", db_path=db, max_messages=12)
        try:
            assert mem._use_sql is False
            mem.add_user_message("hello")
            assert len(mem.messages) == 1
            assert len(mem._fallback.messages) == 1
        finally:
            mem.close()
            reset_store()
    try:
        if os.path.exists(db):
            os.unlink(db)
        os.rmdir(tmp)
    except Exception:
        pass


def test_clear():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "mem.db")
    s = make_settings(db)
    with patch("config.get_settings", return_value=s):
        reset_store()
        mem = SQLiteHistoryWrapper(session_id="test", db_path=db, max_messages=12)
        try:
            mem.add_user_message("hello")
            mem.add_ai_message("hi")
            assert len(mem.messages) == 2
            mem.clear()
            assert len(mem.messages) == 0
        finally:
            mem.close()
            reset_store()
    try:
        if os.path.exists(db):
            os.unlink(db)
        os.rmdir(tmp)
    except Exception:
        pass


def test_max_history_zero():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "mem.db")
    s = make_settings(db, max_messages=0)
    with patch("config.get_settings", return_value=s):
        reset_store()
        mem = SQLiteHistoryWrapper(session_id="test", db_path=db, max_messages=0)
        try:
            mem.add_user_message("hello")
            assert len(mem.messages) == 0
        finally:
            mem.close()
            reset_store()
    try:
        if os.path.exists(db):
            os.unlink(db)
        os.rmdir(tmp)
    except Exception:
        pass
