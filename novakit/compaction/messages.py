"""Message accessors shared by the compaction stages."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from novakit.session.models import Message as StoredMessage
    from novakit.session.models import SessionContext


def is_summary(message: StoredMessage | dict[str, Any]) -> bool:
    """Whether *message* is a compaction summary rather than real history.

    Args:
        message: Stored message object or raw message dict.

    Returns:
        True when the message carries the summary marker.
    """
    if isinstance(message, dict):
        return bool(message.get("summary"))
    return bool(getattr(message, "summary", 0))


def contains_summary(messages: list) -> bool:
    """Whether any message in *messages* is a compaction summary.

    Args:
        messages: Stored message objects or raw message dicts.

    Returns:
        True when at least one message is a summary.
    """
    return any(is_summary(message) for message in messages)


def new_message_id() -> str:
    """Generate a fresh random message identifier.

    Returns:
        A UUID string.
    """
    return str(uuid.uuid4())


def content_of(msg: StoredMessage | dict[str, Any]) -> str:
    """Return the text content of a stored message or raw message dict.

    Args:
        msg: Stored message object or raw message dict.

    Returns:
        The message content, or an empty string when absent.
    """
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    return getattr(msg, "content", "") or ""


def tool_calls_of(msg: StoredMessage | dict[str, Any]) -> list:
    """Return the tool calls attached to a stored message or raw message dict.

    Args:
        msg: Stored message object or raw message dict.

    Returns:
        The attached tool calls, or an empty list when absent.
    """
    if isinstance(msg, dict):
        return msg.get("tool_calls", []) or []
    return getattr(msg, "tool_calls", []) or []


def role_of(msg: StoredMessage | dict[str, Any]) -> str:
    """Return the role of a stored message or raw message dict.

    Args:
        msg: Stored message object or raw message dict.

    Returns:
        The message role, or ``"?"`` when absent.
    """
    if isinstance(msg, dict):
        return msg.get("role", "?")
    return getattr(msg, "role", "?")


def id_of(msg: StoredMessage | dict[str, Any]) -> str:
    """Return the id of a stored message or raw message dict.

    Args:
        msg: Stored message object or raw message dict.

    Returns:
        The message id, or an empty string when absent.
    """
    if isinstance(msg, dict):
        return msg.get("id", "")
    return getattr(msg, "id", "")


def tool_call_ids_of(msg: StoredMessage | dict[str, Any]) -> list[str]:
    """Extract tool call IDs from an assistant message's tool_calls field.

    Accepts a stored message object or raw message dict; a JSON-encoded
    string field is decoded, and undecodable values yield no ids.

    Args:
        msg: Stored assistant message or raw message dict.

    Returns:
        Declared tool call ids in order, or an empty list.
    """
    tc = tool_calls_of(msg)
    if not tc:
        return []
    if isinstance(tc, str):
        try:
            tc = json.loads(tc)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(tc, list):
        return [t["id"] for t in tc if isinstance(t, dict) and t.get("id")]
    return []


def tool_call_id_of(msg: StoredMessage | dict[str, Any]) -> str:
    """Return the tool_call_id a tool response message answers.

    Args:
        msg: Stored message object or raw message dict.

    Returns:
        The answered tool call id, or an empty string when absent.
    """
    if isinstance(msg, dict):
        return msg.get("tool_call_id", "") or ""
    return getattr(msg, "tool_call_id", "") or ""


def created_at_of(msg: StoredMessage | dict[str, Any]) -> int:
    """Return the creation timestamp of a stored message or raw message dict.

    Args:
        msg: Stored message object or raw message dict.

    Returns:
        Creation time in milliseconds, or 0 when absent.
    """
    if isinstance(msg, dict):
        return int(msg.get("time_created") or 0)
    return int(getattr(msg, "time_created", 0) or 0)


def session_id_of(session: SessionContext | dict[str, Any] | None) -> str | None:
    """Return the id of a session context or raw session dict.

    Args:
        session: Session context, raw session dict, or None.

    Returns:
        The session id, or None when there is no session.
    """
    if session is None:
        return None
    if isinstance(session, dict):
        return session.get("id")
    return getattr(session, "id", None)


def compacted_at_of(
    session: SessionContext | dict[str, Any] | None,
) -> int | None:
    """Return the last successful compaction timestamp, if any.

    Args:
        session: Session context, raw session dict, or None.

    Returns:
        Compaction time in milliseconds, or None when never compacted.
    """
    if session is None:
        return None
    if isinstance(session, dict):
        return session.get("compacted_at")
    return getattr(session, "compacted_at", None)


def set_compacted_at(
    session: SessionContext | dict[str, Any] | None, value: int
) -> None:
    """Record a successful compaction timestamp on the session object in place.

    Args:
        session: Session context or raw session dict to stamp; None is a no-op.
        value: Compaction time in milliseconds.
    """
    if session is None:
        return
    if isinstance(session, dict):
        session["compacted_at"] = value
        return
    session.compacted_at = value


def tokens_input_of(msg: StoredMessage | dict[str, Any]) -> int:
    """Return the billed prompt tokens recorded on an assistant message.

    Args:
        msg: Stored message object or raw message dict.

    Returns:
        Prompt tokens charged for the request, or 0 when unrecorded.
    """
    if isinstance(msg, dict):
        return int(msg.get("tokens_input") or 0)
    return int(getattr(msg, "tokens_input", 0) or 0)


def tokens_output_of(msg: StoredMessage | dict[str, Any]) -> int:
    """Return the billed output tokens recorded on an assistant message.

    Args:
        msg: Stored message object or raw message dict.

    Returns:
        Output tokens charged for the request, or 0 when unrecorded.
    """
    if isinstance(msg, dict):
        return int(msg.get("tokens_output") or 0)
    return int(getattr(msg, "tokens_output", 0) or 0)


__all__ = [
    "compacted_at_of",
    "contains_summary",
    "content_of",
    "created_at_of",
    "id_of",
    "is_summary",
    "new_message_id",
    "role_of",
    "session_id_of",
    "set_compacted_at",
    "tokens_input_of",
    "tokens_output_of",
    "tool_call_id_of",
    "tool_call_ids_of",
    "tool_calls_of",
]
