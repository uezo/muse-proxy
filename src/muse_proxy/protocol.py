"""The supported text-only Chat Completions wire contract.

Like dots-bridge, only the latest user message is forwarded. Muse owns the
conversation; system instructions and earlier messages are not replayed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .errors import ProxyError
from .events import MuseMessage


def _invalid(message: str, param: str | None = None) -> ProxyError:
    return ProxyError(message, status_code=400, code="invalid_request_error", param=param)


def _unicode(value: str, param: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise _invalid("Text must contain valid Unicode.", param) from None
    return value


@dataclass(frozen=True)
class ChatRequest:
    text: str
    model: str
    stream: bool = False
    include_usage: bool = False
    session_id: str | None = None


def parse_chat(payload: Any, default_model: str, *, max_message_chars: int = 16000) -> ChatRequest:
    if not isinstance(payload, dict):
        raise _invalid("Request body must be a JSON object.")
    # Muse selects the backend through the paired account and VM, not this field.
    # Accept clients' model values for compatibility; report the configured alias.
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise _invalid("stream must be a boolean.", "stream")
    if "n" in payload and (type(payload["n"]) is not int or payload["n"] != 1):
        raise _invalid("Only n=1 is supported.", "n")
    for key in ("tools", "functions"):
        if payload.get(key) not in (None, []):
            raise _invalid("Client-supplied tools are not supported.", key)
    for key in ("tool_choice", "function_call"):
        if payload.get(key) not in (None, "none"):
            raise _invalid("Client-side tool calling is not supported.", key)
    if payload.get("modalities") not in (None, ["text"]):
        raise _invalid("Only text output is supported.", "modalities")
    if payload.get("audio") is not None:
        raise _invalid("Audio output is not supported.", "audio")
    if payload.get("response_format") not in (None, {"type": "text"}):
        raise _invalid("Structured output is not supported.", "response_format")
    if payload.get("stop") not in (None, []):
        raise _invalid("Custom stop sequences are not supported.", "stop")
    if payload.get("logprobs") not in (None, False):
        raise _invalid("Token probabilities are not available.", "logprobs")
    options = payload.get("stream_options")
    include_usage = False
    if options is not None:
        if not isinstance(options, dict):
            raise _invalid("stream_options must be an object.", "stream_options")
        include_usage = options.get("include_usage", False)
        if type(include_usage) is not bool:
            raise _invalid("include_usage must be a boolean.", "stream_options.include_usage")
    session_id = payload.get("session_id")
    if session_id is not None and (
        not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,64}", session_id)
    ):
        raise _invalid("session_id must contain 1 to 64 letters, digits or hyphens.", "session_id")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _invalid("messages must be a nonempty array.", "messages")
    for index in range(len(messages) - 1, -1, -1):
        item = messages[index]
        if isinstance(item, dict) and item.get("role") == "user":
            param = f"messages[{index}].content"
            content = item.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if (
                        not isinstance(part, dict)
                        or part.get("type") not in ("text", "input_text")
                        or not isinstance(part.get("text"), str)
                    ):
                        raise _invalid("Only text content parts are supported.", param)
                    parts.append(part["text"])
                text = "\n".join(parts)
            else:
                raise _invalid("User content must be a string or text parts.", param)
            if not text.strip() or len(text) > max_message_chars:
                raise _invalid(
                    f"The latest user message must contain 1 to {max_message_chars} characters.",
                    param,
                )
            return ChatRequest(
                _unicode(text, param), default_model, stream, include_usage, session_id
            )
    raise _invalid("messages must contain a user message.", "messages")


def chat_completion(text: str, *, id: str, created: int, model: str) -> dict:
    return {
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
    }


def chat_chunk(
    delta: dict, *, id: str, created: int, model: str, finish_reason: str | None = None
) -> dict:
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None}],
    }


def message_metadata(message: MuseMessage, *, request_id: str, sequence: int) -> dict:
    """Describe a complete Muse message without changing Chat Completions fields."""
    metadata = {
        "kind": message.kind,
        "request_id": request_id,
        "message_id": message.message_id or f"musemsg-{uuid4().hex}",
        "sequence": sequence,
    }
    if message.kind == "progress" and message.activity_code is not None:
        metadata["activity_code"] = message.activity_code
    if message.kind == "cancel":
        metadata["superseded_by"] = message.superseded_by
    return metadata
