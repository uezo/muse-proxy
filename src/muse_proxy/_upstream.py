"""Instance-local diagnostics for the SDK's plaintext Muse protocol boundary."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ._trace import MAX_BODY_BYTES, BodyTrace, _credential_field
from .logging import TRACE_LEVEL

logger = logging.getLogger("muse_proxy.upstream")
_TRACE: ContextVar[MuseTrace | None] = ContextVar("muse_proxy.upstream.trace", default=None)
_REQUEST_ID: ContextVar[str] = ContextVar("muse_proxy.upstream.request_id", default="-")
_ATOM = re.compile(r"[A-Za-z0-9._:/@-]+\Z")
_MAX_STREAMS = 256
_DIAGNOSTIC_HEADERS = {"x-request-id", "content-type", "accept", "x-app-id"}


def _observe(call, *args, **kwargs):
    """Diagnostic failures must not replace SDK return values or exceptions."""
    try:
        return call(*args, **kwargs)
    except Exception:
        return None


def current_trace() -> MuseTrace | None:
    return _TRACE.get()


@contextmanager
def trace_context(trace: MuseTrace | None, request_id: str = "-"):
    """Propagate a router and request through asyncio tasks and to_thread calls."""
    trace_token = _TRACE.set(trace)
    request_token = _REQUEST_ID.set(request_id)
    try:
        yield
    finally:
        _REQUEST_ID.reset(request_token)
        _TRACE.reset(trace_token)


def _body(data: bytes | bytearray | str):
    if isinstance(data, (bytes, bytearray)):
        data = bytes(data).decode("utf-8", errors="replace")
    try:
        return json.loads(data), "json"
    except (ValueError, TypeError, RecursionError):
        return data, "text"


def _headers(headers) -> dict:
    result = {}
    if isinstance(headers, dict):
        headers = headers.items()
    for header in headers or ():
        if isinstance(header, (tuple, list)) and len(header) == 2:
            name, value = header
        else:
            name, value = getattr(header, "key", None), getattr(header, "value", None)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(name, str):
            result[name.lower()] = value
    return result


class MuseTrace:
    """Redact protocol diagnostics without exposing SDK/network logger output."""

    def __init__(self, router_id: str, secrets: Iterable[str | None] = ()):
        self.router_id = router_id
        self._secrets: set[str] = set()
        self._secret_lock = RLock()
        self._redactor = BodyTrace(router_id)
        self.add_secrets(*secrets)

    def add_secrets(self, *values: str | None) -> None:
        added = set()
        for value in values:
            if isinstance(value, str) and value:
                added.add(value)
                # Refresh tokens can carry a hatch_refresh: prefix. Errors may echo
                # either representation; redact both before any diagnostic output.
                if ":" in value:
                    suffix = value.rsplit(":", 1)[-1]
                    if suffix:
                        added.add(suffix)
        with self._secret_lock:
            if added - self._secrets:
                self._secrets.update(added)
                self._redactor = BodyTrace(self.router_id, self._secrets)

    def credentials(self, credentials: Any) -> None:
        pairing = getattr(credentials, "pairing", {})
        if isinstance(pairing, dict):
            self.add_secrets(pairing.get("access_token"), pairing.get("refresh_token"))
        self.add_secrets(getattr(credentials, "sdk_token", None))

    def _learn_credentials(self, value, depth: int = 0) -> None:
        if depth > 64:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and _credential_field(key):
                    if isinstance(item, str):
                        self.add_secrets(item)
                else:
                    self._learn_credentials(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._learn_credentials(item, depth + 1)

    @staticmethod
    def _render(value) -> str:
        if isinstance(value, str) and _ATOM.fullmatch(value):
            return value
        result = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return (
            result.encode("utf-8", "backslashreplace")
            .decode("utf-8")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )

    def emit(self, event: str, **fields) -> None:
        if not logger.isEnabledFor(TRACE_LEVEL):
            return
        try:
            with self._secret_lock:
                self._learn_credentials(fields)
                request_id = fields.pop("request_id", _REQUEST_ID.get())
                safe = self._redactor._redact(
                    {"router_id": self.router_id, "request_id": request_id, **fields}
                )
            rendered = [f"{key}={self._render(value)}" for key, value in safe.items()]
            detail = " ".join(rendered)
            if len(detail.encode("utf-8")) > MAX_BODY_BYTES:
                # Keep correlation and status even when a body cannot fit.
                safe = {
                    key: value
                    for key, value in safe.items()
                    if key
                    in {
                        "router_id",
                        "request_id",
                        "upstream_request_id",
                        "stream_id",
                        "method",
                        "path",
                        "status",
                        "operation",
                        "error_type",
                        "format",
                    }
                }
                safe["body"] = {"omitted": "size_limit", "limit_bytes": MAX_BODY_BYTES}
                detail = " ".join(f"{key}={self._render(value)}" for key, value in safe.items())
            logger.log(TRACE_LEVEL, "%s %s", event, detail)
        except Exception:
            # Diagnostics must never alter the request, encrypted nonce sequence,
            # error, cancellation, or stream backpressure behavior they describe.
            _observe(logger.log, TRACE_LEVEL, '%s body={"omitted":"unserializable_body"}', event)

    def failure(self, operation: str, exc: BaseException, **fields) -> None:
        if not logger.isEnabledFor(TRACE_LEVEL):
            return
        if isinstance(exc, asyncio.CancelledError):
            self.emit("muse.connection.closed", operation=operation, outcome="cancelled", **fields)
            return
        try:
            self._failure(operation, exc, **fields)
        except Exception:
            self.emit(
                "muse.failure",
                operation=operation,
                error_type=type(exc).__name__,
                body={"omitted": "unreadable_exception"},
                **fields,
            )

    def _failure(self, operation: str, exc: BaseException, **fields) -> None:
        details = {"operation": operation, "error_type": type(exc).__name__, "message": str(exc)}
        for name in ("status", "status_code"):
            value = getattr(exc, name, None)
            if isinstance(value, int):
                details["status"] = value
                break
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if isinstance(status, int):
                details["status"] = status
            body = getattr(response, "body", None)
            if isinstance(body, (bytes, bytearray, str)):
                if len(body) > MAX_BODY_BYTES:
                    details["body"] = {"omitted": "size_limit", "limit_bytes": MAX_BODY_BYTES}
                else:
                    details["body"], details["format"] = _body(body)
        # websockets ConnectionClosed exposes received/sent close frames without
        # using the deprecated exception.code/reason compatibility properties.
        close = getattr(exc, "rcvd", None) or getattr(exc, "sent", None)
        if close is not None:
            details["close_code"] = getattr(close, "code", None)
            details["close_reason"] = getattr(close, "reason", None)
        else:
            for name in ("code", "reason"):
                value = getattr(exc, name, None)
                if isinstance(value, (int, str)):
                    details[f"close_{name}"] = value
        self.emit("muse.failure", **{**details, **fields})


@dataclass
class _Stream:
    method: str
    path: str
    request_id: str
    upstream_request_id: str
    stream_id: int
    request_buffer: bytearray = field(default_factory=bytearray)
    response_buffer: bytearray = field(default_factory=bytearray)
    request_stopped: bool = False
    response_stopped: bool = False
    status: int | None = None
    sequence: int = 0

    def fields(self) -> dict:
        return {
            "request_id": self.request_id,
            "upstream_request_id": self.upstream_request_id,
            "stream_id": self.stream_id,
            "method": self.method,
            "path": self.path,
        }


class _ObservedTransport:
    """Observe each plaintext frame while delegating encryption/decryption once."""

    def __init__(self, delegate, trace: MuseTrace):
        self._delegate = delegate
        self._trace = trace
        self._streams: OrderedDict[int, _Stream] = OrderedDict()

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def _request(self, delegate, method, path, body, *args, **kwargs):
        try:
            encrypted = delegate(method, path, *args, **kwargs)
        except BaseException as exc:
            self._trace.failure("encrypt_request", exc, method=method, path=path)
            raise
        _observe(self._record_request, encrypted, method, path, body, args, kwargs)
        return encrypted

    def _record_request(self, encrypted, method, path, body, args, kwargs):
        headers = _headers(kwargs.get("headers", args[1] if len(args) > 1 else ()))
        request_id = "-" if path in ("/chat/subscribe", "/link-control") else _REQUEST_ID.get()
        stream = _Stream(
            method,
            path,
            request_id,
            headers.get("x-request-id") or "-",
            encrypted.stream_id,
        )
        self._streams[encrypted.stream_id] = stream
        while len(self._streams) > _MAX_STREAMS:
            self._streams.popitem(last=False)
        self._trace.emit(
            "muse.request",
            **stream.fields(),
            headers={key: value for key, value in headers.items() if key in _DIAGNOSTIC_HEADERS},
        )
        if body is not None and path != "/link-control":
            self._append(stream, body, request=True, ended=True)

    def encrypt_http_request(self, method, path, body=b"", *args, **kwargs):
        return self._request(
            lambda method, path, *args, **kwargs: self._delegate.encrypt_http_request(
                method, path, body, *args, **kwargs
            ),
            method,
            path,
            body,
            *args,
            **kwargs,
        )

    def start_stream_request(self, method, path, *args, **kwargs):
        return self._request(
            self._delegate.start_stream_request, method, path, None, *args, **kwargs
        )

    def encrypt_body_chunk(self, stream_id, data, *args, **kwargs):
        frames = self._delegate.encrypt_body_chunk(stream_id, data, *args, **kwargs)
        stream = self._streams.get(stream_id)
        if stream is not None and stream.path != "/link-control":
            ended = kwargs.get("end_body", args[1] if len(args) > 1 else False)
            _observe(self._append, stream, data, request=True, ended=ended)
        return frames

    def encrypt_reset(self, stream_id, *args, **kwargs):
        frames = self._delegate.encrypt_reset(stream_id, *args, **kwargs)
        _observe(self._reset, stream_id, args, kwargs)
        return frames

    def _reset(self, stream_id, args, kwargs):
        stream = self._streams.pop(stream_id, None)
        if stream is not None:
            self._flush(stream, incomplete=True)
            self._trace.emit(
                "muse.request.reset",
                **stream.fields(),
                reason=kwargs.get("reason", args[1] if len(args) > 1 else ""),
            )

    def _append(self, stream: _Stream, data, *, request: bool, ended: bool) -> None:
        if request:
            buffer, stopped = stream.request_buffer, stream.request_stopped
            event = "muse.request.body"
        else:
            buffer, stopped = stream.response_buffer, stream.response_stopped
            event = "muse.response.body"
        if stopped:
            return
        ndjson = not request and stream.path == "/chat/subscribe" and stream.status is not None
        ndjson = ndjson and 200 <= stream.status < 300
        if ndjson:
            self._ndjson(stream, data, ended)
            return
        if len(buffer) + len(data) > MAX_BODY_BYTES:
            buffer.clear()
            if request:
                stream.request_stopped = True
            else:
                stream.response_stopped = True
            self._trace.emit(
                event,
                **stream.fields(),
                body={"omitted": "size_limit", "limit_bytes": MAX_BODY_BYTES},
            )
            return
        buffer.extend(data)
        if ended:
            value, format = _body(buffer)
            buffer.clear()
            if not request:
                stream.sequence += 1
            self._trace.emit(
                event, **stream.fields(), format=format, sequence=stream.sequence, body=value
            )

    def _ndjson(self, stream: _Stream, data: bytes, ended: bool) -> None:
        position = 0
        while position < len(data):
            boundary = data.find(b"\n", position)
            end = len(data) if boundary == -1 else boundary + 1
            part = data[position:end]
            if len(stream.response_buffer) + len(part) > MAX_BODY_BYTES:
                stream.response_buffer.clear()
                stream.response_stopped = True
                self._trace.emit(
                    "muse.response.body",
                    **stream.fields(),
                    body={"omitted": "size_limit", "limit_bytes": MAX_BODY_BYTES},
                )
                return
            stream.response_buffer.extend(part)
            if boundary != -1:
                self._ndjson_line(stream)
            position = end
        if ended and stream.response_buffer:
            self._ndjson_line(stream)

    def _ndjson_line(self, stream: _Stream) -> None:
        data = bytes(stream.response_buffer)
        stream.response_buffer.clear()
        if data.strip():
            value, _ = _body(data)
            stream.sequence += 1
            self._trace.emit(
                "muse.response.body",
                **stream.fields(),
                format="ndjson",
                sequence=stream.sequence,
                body=value,
            )

    def _flush(self, stream: _Stream, *, incomplete: bool) -> None:
        if stream.response_buffer:
            value, format = _body(stream.response_buffer)
            stream.response_buffer.clear()
            self._trace.emit(
                "muse.response.body",
                **stream.fields(),
                format=format,
                body=value,
                incomplete=incomplete,
            )

    def decrypt_frame(self, data):
        try:
            frame = self._delegate.decrypt_frame(data)
        except BaseException as exc:
            self._trace.failure("decrypt_response", exc)
            raise
        if frame is None:
            return frame
        _observe(self._received, frame)
        return frame

    def _received(self, frame):
        stream = self._streams.get(frame.stream_id)
        if stream is None:
            self._trace.emit(
                "muse.response.unmatched",
                request_id="-",
                stream_id=frame.stream_id,
                kind=frame.kind,
            )
            return
        if frame.kind == "reset":
            self._flush(stream, incomplete=True)
            self._trace.emit(
                "muse.response.reset",
                **stream.fields(),
                reason=getattr(frame.value, "reason", None),
                code=getattr(frame.value, "code", None),
            )
            self._streams.pop(frame.stream_id, None)
            return
        if frame.kind == "response":
            stream.status = frame.value.status
            self._trace.emit(
                "muse.response",
                **stream.fields(),
                status=stream.status,
                headers={
                    key: value
                    for key, value in _headers(frame.value.headers).items()
                    if key in _DIAGNOSTIC_HEADERS
                },
            )
            body, ended = frame.value.body, frame.value.end_body
        elif frame.kind == "body_chunk":
            body, ended = frame.value.data, frame.value.end_body
        else:
            return
        if stream.path != "/link-control":
            self._append(stream, body, request=False, ended=ended)
        if ended:
            self._streams.pop(frame.stream_id, None)

    def finish(self) -> None:
        for stream in self._streams.values():
            _observe(self._flush, stream, incomplete=True)
        self._streams.clear()


class _ObservedWebSocket:
    def __init__(self, delegate, trace: MuseTrace):
        self._delegate = delegate
        self._trace = trace

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    async def recv(self, *args, **kwargs):
        try:
            return await self._delegate.recv(*args, **kwargs)
        except BaseException as exc:
            self._trace.failure("websocket_receive", exc, request_id="-")
            raise


def instrument_session(session, trace: MuseTrace):
    """Attach observers to this session only, never to SDK classes or globals."""
    if not logger.isEnabledFor(TRACE_LEVEL) or getattr(session, "_muse_proxy_trace", None):
        return session
    try:
        session._muse_proxy_trace = trace
    except (AttributeError, TypeError):
        return session
    trace.add_secrets(getattr(session, "_token", None))

    def attach(name, wrapper):
        def bind():
            original = getattr(session, name, None)
            if callable(original):
                setattr(session, name, wraps(original)(wrapper(original)))

        _observe(bind)

    def handshake(original):
        async def observed(*args, **kwargs):
            try:
                transport = await original(*args, **kwargs)
            except BaseException as exc:
                trace.failure("noise_handshake", exc, request_id="-")
                raise
            trace.emit("muse.connection.handshake", request_id="-", outcome="established")
            return _ObservedTransport(transport, trace)

        return observed

    attach("_handshake", handshake)

    def open_socket(original):
        async def observed(*args, **kwargs):
            target = _observe(getattr, session, "_url")
            if isinstance(target, str):
                parts = _observe(urlsplit, target)
                target = (
                    urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")) if parts else None
                )
            trace.emit("muse.connection.open", request_id="-", url=target)
            try:
                websocket = await original(*args, **kwargs)
            except BaseException as exc:
                trace.failure("websocket_open", exc, request_id="-")
                raise
            response = _observe(getattr, websocket, "response", None)
            trace.emit(
                "muse.connection.opened",
                request_id="-",
                status=_observe(getattr, response, "status_code", None),
            )
            return _ObservedWebSocket(websocket, trace)

        return observed

    attach("_open", open_socket)

    def chat(original):
        async def observed(body, *args, **kwargs):
            trace.emit("muse.chat.request", path="/chat/stream", body=body)
            try:
                response = await original(body, *args, **kwargs)
            except BaseException as exc:
                trace.failure("send_chat", exc, path="/chat/stream")
                raise
            trace.emit(
                "muse.chat.response",
                path="/chat/stream",
                status=_observe(response.get, "status") if isinstance(response, dict) else None,
                body=response,
            )
            return response

        return observed

    if callable(getattr(session, "_post_chat", None)):
        attach("_post_chat", chat)
    else:

        def send_chat(original):
            async def observed(message, session_id=None, **kwargs):
                body = {"message": message, "session_id": session_id, **kwargs}

                async def delegated(_):
                    return await original(message, session_id, **kwargs)

                return await chat(delegated)(body)

            return observed

        attach("send_chat", send_chat)

    def send_control(original):
        async def observed(message, *args, **kwargs):
            trace.emit("muse.control.request", request_id="-", body=message)
            return await original(message, *args, **kwargs)

        return observed

    attach("send", send_control)

    def handle_control(original):
        def observed(message, *args, **kwargs):
            trace.emit("muse.control.response", request_id="-", body=message)
            return original(message, *args, **kwargs)

        return observed

    attach("_handle", handle_control)

    def run(original):
        async def observed(*args, **kwargs):
            with trace_context(trace, _REQUEST_ID.get()):
                try:
                    outcome = await original(*args, **kwargs)
                except BaseException as exc:
                    trace.failure("connection", exc, request_id="-")
                    raise
                else:
                    trace.emit(
                        "muse.connection.outcome",
                        request_id="-",
                        outcome=_observe(getattr, outcome, "value", outcome),
                    )
                    return outcome
                finally:
                    transport = _observe(getattr, session, "_transport", None)
                    if isinstance(transport, _ObservedTransport):
                        _observe(transport.finish)

        return observed

    attach("run", run)
    return session
