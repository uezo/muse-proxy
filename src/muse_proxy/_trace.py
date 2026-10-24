"""Opt-in HTTP body diagnostics, independent of SDK and transport logging."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable

from .logging import TRACE_LEVEL

logger = logging.getLogger("muse_proxy.http")
MAX_BODY_BYTES = 1024 * 1024
MAX_DEPTH = 64
_REDACTED = "[REDACTED]"


def _credential_field(key: str) -> bool:
    name = re.sub(r"[^a-z0-9]", "", key.lower())
    return name in {"auth", "passwd", "sessionkey"} or name.endswith(
        (
            "token",
            "secret",
            "password",
            "apikey",
            "privatekey",
            "credential",
            "credentials",
            "authorization",
            "cookie",
        )
    )


class BodyTrace:
    """Keep at most one bounded response body/frame, never an entire SSE stream."""

    def __init__(self, http_id: str, redact_values: Iterable[str | None] = ()):
        self.http_id = http_id
        self.request_id = "-"
        values = sorted({value for value in redact_values if value}, key=len, reverse=True)
        self._secrets = (
            re.compile("|".join(re.escape(value) for value in values)) if values else None
        )
        self._format = "other"
        self._buffer = bytearray()
        self._sequence = 0
        self._stopped = False

    def _mask(self, value: str) -> str:
        return self._secrets.sub(lambda _: _REDACTED, value) if self._secrets else value

    def _redact(self, value, depth=0):
        if depth > MAX_DEPTH:
            raise ValueError("Body nesting exceeds the trace limit")
        if isinstance(value, dict):
            return {
                self._mask(key): _REDACTED
                if _credential_field(key)
                else self._redact(item, depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact(item, depth + 1) for item in value]
        if isinstance(value, str):
            return self._mask(value)
        if self._secrets and self._secrets.search(json.dumps(value, allow_nan=False)):
            return _REDACTED
        return value

    def _emit(self, event: str, format: str, value) -> None:
        if not logger.isEnabledFor(TRACE_LEVEL):
            return
        try:
            body = json.dumps(
                self._redact(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            # Escape lone surrogates and Unicode line separators without hiding Japanese text.
            body = body.encode("utf-8", "backslashreplace").decode("utf-8")
            body = body.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            if len(body.encode("utf-8")) > MAX_BODY_BYTES:
                body = f'{{"omitted":"size_limit","limit_bytes":{MAX_BODY_BYTES}}}'
                if event == "http.response.body":
                    self._stopped = True
                    self._buffer.clear()
        except (ValueError, TypeError, RecursionError):
            body = '{"omitted":"unserializable_or_nested_body"}'
        if event == "http.response.body":
            self._sequence += 1
        logger.log(
            TRACE_LEVEL,
            "%s http_id=%s request_id=%s format=%s sequence=%d body=%s",
            event,
            self.http_id,
            self.request_id,
            format,
            self._sequence,
            body,
        )

    def request(self, value=None, *, omitted: str | None = None) -> None:
        self._emit("http.request.body", "json", {"omitted": omitted} if omitted else value)

    def start_response(self, headers) -> None:
        content_type = next(
            (
                value.split(b";", 1)[0].strip().lower()
                for key, value in headers
                if key.lower() == b"content-type"
            ),
            b"",
        )
        if content_type == b"application/json":
            self._format = "json"
        elif content_type == b"text/event-stream":
            self._format = "sse"

    def _append(self, data: bytes) -> bool:
        if len(self._buffer) + len(data) > MAX_BODY_BYTES:
            self._emit(
                "http.response.body",
                self._format,
                {"omitted": "size_limit", "limit_bytes": MAX_BODY_BYTES},
            )
            self._buffer.clear()
            self._stopped = True
            return False
        self._buffer.extend(data)
        return True

    def _json(self, data: bytes | str) -> None:
        try:
            value = json.loads(data)
        except (ValueError, UnicodeError, RecursionError):
            value = {"omitted": "invalid_json"}
        self._emit("http.response.body", self._format, value)

    def _frame(self) -> None:
        frame = bytes(self._buffer)
        self._buffer.clear()
        if not frame:
            return
        try:
            text = frame.decode("utf-8")
        except UnicodeError:
            self._emit("http.response.body", "sse", {"omitted": "invalid_utf8"})
            return
        if text == ": keepalive":
            self._emit("http.response.body", "sse", text)
            return
        data = "\n".join(
            line[5:].removeprefix(" ") for line in text.split("\n") if line.startswith("data:")
        )
        if data == "[DONE]":
            self._emit("http.response.body", "sse", data)
        else:
            self._json(data)

    def response(self, data: bytes, *, more_body: bool) -> None:
        if self._stopped or not logger.isEnabledFor(TRACE_LEVEL):
            return
        if self._format == "json":
            if self._append(data) and not more_body:
                self._json(bytes(self._buffer))
                self._buffer.clear()
        elif self._format == "sse":
            position = 0
            # A frame delimiter can straddle two ASGI messages.
            if self._buffer.endswith(b"\n") and data.startswith(b"\n"):
                self._buffer.pop()
                self._frame()
                position = 1
            while position < len(data) and not self._stopped:
                boundary = data.find(b"\n\n", position)
                end = len(data) if boundary == -1 else boundary
                if not self._append(data[position:end]):
                    return
                if boundary == -1:
                    break
                self._frame()
                position = boundary + 2
        elif not more_body:
            self._emit("http.response.body", "other", {"omitted": "unsupported_content_type"})

    def finish(self) -> None:
        if self._buffer and not self._stopped:
            self._emit("http.response.body", self._format, {"omitted": "incomplete_body"})
            self._buffer.clear()
