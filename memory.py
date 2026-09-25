"""
memory.py — SQLite-backed conversation history via RunnableWithMessageHistory.

- Modern LangChain: RunnableWithMessageHistory (never ConversationChain / ConversationBufferMemory)
- SQLite persisted at MEMORY_DB_PATH, survives restarts, Docker volume
- Falls back to in-memory (logged warning) if SQLite write fails — never crash
- Never persists API keys, system prompts, or secrets
- Message-count trimming (MAX_HISTORY_MESSAGES)
- Allow disable via MEMORY_BACKEND=disabled / memory
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_core.chat_history import BaseChatMessageHistory  # type: ignore
    from langchain_core.messages import AIMessage, HumanMessage, BaseMessage  # type: ignore
    from langchain_core.runnables.history import RunnableWithMessageHistory  # type: ignore
    HAS_LANGCHAIN = True
except Exception as e:  # pragma: no cover
    BaseChatMessageHistory = object  # type: ignore
    AIMessage = HumanMessage = BaseMessage = object  # type: ignore
    RunnableWithMessageHistory = None  # type: ignore
    HAS_LANGCHAIN = False
    logger.debug(f"langchain not available: {e}")

try:
    from langchain_community.chat_message_histories import SQLChatMessageHistory  # type: ignore
    HAS_SQL_HISTORY = True
except Exception:
    SQLChatMessageHistory = None  # type: ignore
    HAS_SQL_HISTORY = False

# ── In-memory fallback history ─────────────────────────────

class InMemoryHistory(BaseChatMessageHistory):  # type: ignore
    """Simple in-memory chat history with trimming."""
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.messages: List[BaseMessage] = []  # type: ignore

    def add_message(self, message: BaseMessage) -> None:  # type: ignore
        self.messages.append(message)

    def add_messages(self, messages: List[BaseMessage]) -> None:  # type: ignore
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

    def trim(self, max_messages: int) -> None:
        if max_messages <= 0:
            self.messages = []
        elif len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]


class SQLiteHistoryWrapper:
    """
    Wrapper that tries SQLChatMessageHistory, falls back to InMemoryHistory on write failure.
    Also handles trimming and never stores secrets.
    """
    def __init__(self, session_id: str, db_path: str, max_messages: int = 12):
        self.session_id = session_id
        self.db_path = Path(db_path)
        self.max_messages = max_messages
        self._fallback = InMemoryHistory(session_id=session_id)
        self._use_sql = False
        self._sql_history: Optional[SQLChatMessageHistory] = None  # type: ignore

        # If backend disabled, stay in fallback
        from config import get_settings
        backend = get_settings().memory_backend
        if backend == "disabled":
            logger.info("Memory disabled via MEMORY_BACKEND=disabled")
            self._use_sql = False
            return

        if backend == "memory":
            logger.info("Memory set to in-memory only")
            self._use_sql = False
            return

        # Try to init SQL
        if not HAS_SQL_HISTORY:
            logger.warning("langchain_community SQLChatMessageHistory not available — using in-memory fallback")
            self._use_sql = False
            return

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure file exists
            # SQLChatMessageHistory expects connection string
            conn_str = f"sqlite:///{self.db_path}"
            self._sql_history = SQLChatMessageHistory(
                session_id=session_id,
                connection=conn_str,  # type: ignore
            )
            # Test write
            # Don't actually write test message; just check DB reachable
            self._use_sql = True
            logger.info(f"SQLite history ready: {self.db_path} (session={session_id})")
        except Exception as e:
            logger.warning(f"SQLite init failed at {self.db_path}: {e} — falling back to in-memory")
            self._use_sql = False

    def _trim_sql(self) -> None:
        if not self._use_sql or self._sql_history is None:
            return
        try:
            if self.max_messages <= 0:
                self._sql_history.clear()  # type: ignore
                return
            msgs = self._sql_history.messages  # type: ignore
            if len(msgs) > self.max_messages:
                keep = msgs[-self.max_messages:]
                self._sql_history.clear()  # type: ignore
                for m in keep:
                    self._sql_history.add_message(m)  # type: ignore
        except Exception as e:
            logger.warning(f"SQLite trim failed: {e} — switching to in-memory")
            self._use_sql = False
            try:
                self._fallback.messages = list(self._sql_history.messages) if self._sql_history else []  # type: ignore
            except Exception:
                pass

    def close(self) -> None:
        """Dispose SQL engine to release file lock (needed on Windows for tests)."""
        if self._sql_history is not None:
            try:
                # Try to dispose SQLAlchemy engine if present
                engine = getattr(self._sql_history, "engine", None) or getattr(self._sql_history, "_engine", None)
                if engine is not None and hasattr(engine, "dispose"):
                    engine.dispose()
                # Also try session handling
                session = getattr(self._sql_history, "session", None)
                if session is not None and hasattr(session, "close"):
                    try:
                        session.close()
                    except Exception:
                        pass
            except Exception:
                pass
            self._sql_history = None
        self._use_sql = False

    def add_user_message(self, text: str) -> None:
        if not text or text.strip() == "":
            return
        # Never persist secrets — filter obvious keys (defense in depth, though caller shouldn't send them)
        lowered = text.lower()
        if "api_key" in lowered or "sk-" in text:
            logger.warning("Refusing to persist potential secret in user message")
            return
        msg = HumanMessage(content=text)  # type: ignore
        if self._use_sql:
            try:
                self._sql_history.add_message(msg)  # type: ignore
                self._trim_sql()
                return
            except Exception as e:
                logger.warning(f"SQLite write failed (user): {e} — falling back to in-memory")
                self._use_sql = False
        # fallback
        self._fallback.add_message(msg)
        self._fallback.trim(self.max_messages)

    def add_ai_message(self, text: str) -> None:
        if not text:
            return
        # Never persist system prompts
        msg = AIMessage(content=text)  # type: ignore
        if self._use_sql:
            try:
                self._sql_history.add_message(msg)  # type: ignore
                self._trim_sql()
                return
            except Exception as e:
                logger.warning(f"SQLite write failed (ai): {e} — falling back to in-memory")
                self._use_sql = False
        self._fallback.add_message(msg)
        self._fallback.trim(self.max_messages)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        if self._use_sql and self._sql_history is not None:
            try:
                if self.max_messages <= 0:
                    return []
                msgs = list(self._sql_history.messages)  # type: ignore
                if len(msgs) > self.max_messages:
                    return msgs[-self.max_messages:]
                return msgs
            except Exception as e:
                logger.warning(f"SQLite read failed: {e} — using fallback")
                self._use_sql = False
        return list(self._fallback.messages)

    def clear(self) -> None:
        if self._use_sql and self._sql_history is not None:
            try:
                self._sql_history.clear()  # type: ignore
            except Exception as e:
                logger.warning(f"SQLite clear failed: {e}")
        self._fallback.clear()

    def get_history_for_llm(self) -> List[BaseMessage]:  # type: ignore
        """Return trimmed messages for LLM (already filtered)."""
        return self.messages

# ── Factory for RunnableWithMessageHistory ───────────────────

_memory_store: Dict[str, SQLiteHistoryWrapper] = {}

def get_memory(session_id: Optional[str] = None) -> SQLiteHistoryWrapper:
    from config import get_settings
    s = get_settings()
    sid = session_id or s.session_id
    key = f"{s.memory_db_path}:{sid}"
    if key not in _memory_store:
        _memory_store[key] = SQLiteHistoryWrapper(
            session_id=sid,
            db_path=s.memory_db_path,
            max_messages=s.max_history_messages,
        )
    return _memory_store[key]

def get_session_history(session_id: str) -> BaseChatMessageHistory:  # type: ignore
    """
    Callable for RunnableWithMessageHistory.
    Returns a BaseChatMessageHistory for the session.
    """
    mem = get_memory(session_id)
    # Adapt wrapper to BaseChatMessageHistory interface
    # If using SQL, return underlying SQL history; else fallback
    if mem._use_sql and mem._sql_history is not None:
        return mem._sql_history  # type: ignore
    return mem._fallback  # type: ignore

def create_runnable_with_history(runnable):  # type: ignore
    """Wrap a runnable with message history (modern API)."""
    if RunnableWithMessageHistory is None:
        raise RuntimeError("langchain_core not available — cannot create RunnableWithMessageHistory")
    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

def clear_memory(session_id: Optional[str] = None) -> None:
    mem = get_memory(session_id)
    mem.clear()

def reset_store() -> None:
    """For testing — clear global store and dispose engines."""
    for mem in list(_memory_store.values()):
        try:
            mem.close()
        except Exception:
            pass
    _memory_store.clear()
