"""Request diagnostics that never inspect or log client or Muse content."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from uuid import uuid4

from starlette.requests import ClientDisconnect, Request

from .errors import ProxyError

logger = logging.getLogger("muse_proxy.http")
_SCOPE_KEY = "muse_proxy.request_log"
_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
_PATHS = {
    "/",
    "/health",
    "/v1",
    "/v1/models",
    "/v1/chat/completions",
    # Common client configuration mistakes, kept as fixed labels.
    "/models",
    "/chat/completions",
    "/responses",
    "/v1/responses",
    "/v1/v1/models",
    "/v1/v1/chat/completions",
}
_CODES = {
    "invalid_request_error",
    "invalid_api_key",
    "origin_rejected",
    "model_not_found",
    "body_too_large",
    "muse_busy",
    "muse_credentials_error",
    "muse_sdk_unavailable",
    "muse_unavailable",
    "muse_auth_error",
    "muse_request_error",
    "muse_connection_error",
    "muse_protocol_error",
    "muse_task_error",
    "muse_empty_reply",
    "muse_timeout",
    "muse_reply_timeout",
    "muse_error",
}
_PARAMS = {
    "model",
    "messages",
    "stream",
    "n",
    "tools",
    "functions",
    "tool_choice",
    "function_call",
    "modalities",
    "audio",
    "response_format",
    "stop",
    "logprobs",
    "stream_options",
    "stream_options.include_usage",
    "session_id",
}


def _path_label(path: str) -> str:
    label = path.rstrip("/") or "/"
    if label in _PATHS:
        return label
    if label.startswith("/v1/models/") and "/" not in label[len("/v1/models/") :]:
        return "/v1/models/{model}"
    return "<unmatched>"


def _param_label(param: str | None) -> str:
    if param is None:
        return "-"
    if not isinstance(param, str):
        return "<other>"
    if param in _PARAMS or re.fullmatch(r"messages\[[0-9]{1,6}\]\.content", param):
        return param
    return "<other>"


@dataclass
class _RequestLog:
    http_id: str = field(default_factory=lambda: f"http-{uuid4().hex}")
    request_id: str = "-"
    status: int | None = None
    result: str | None = None
    code: str = "-"
    param: str = "-"
    progress: int = 0
    replies: int = 0
    disconnected: bool = False
    complete: bool = False


class RequestLoggingMiddleware:
    """Observe ASGI messages without consuming, buffering, or changing them."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        record = _RequestLog()
        scope[_SCOPE_KEY] = record
        method = scope.get("method")
        method = method if method in _METHODS else "OTHER"
        path = _path_label(scope.get("path", ""))
        started = time.monotonic()
        logger.info("http.request http_id=%s method=%s path=%s", record.http_id, method, path)

        async def logged_receive():
            message = await receive()
            if message["type"] == "http.disconnect" and not record.complete:
                record.disconnected = True
            return message

        async def logged_send(message):
            if message["type"] == "http.response.start":
                record.status = message["status"]
            try:
                await send(message)
            except (OSError, ClientDisconnect):
                record.complete = False
                record.disconnected = True
                raise
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                record.complete = True

        try:
            await self.app(scope, logged_receive, logged_send)
        except ClientDisconnect:
            record.disconnected = True
            raise
        except asyncio.CancelledError:
            record.result = "interrupted"
            raise
        except Exception:
            if not record.disconnected:
                record.result = "error"
                if record.status is None:
                    record.status = 500
                # Exception messages and tracebacks can contain credentials or text.
                logger.error(
                    "request.failed http_id=%s request_id=%s code=internal_error",
                    record.http_id,
                    record.request_id,
                )
            raise
        finally:
            result = (
                "disconnected"
                if record.disconnected
                else record.result
                or (
                    "rejected"
                    if (record.status or 0) >= 400
                    else "ok"
                    if record.complete
                    else "interrupted"
                )
            )
            logger.info(
                "http.response http_id=%s method=%s path=%s status=%s result=%s "
                "duration_ms=%.1f request_id=%s code=%s param=%s progress=%d replies=%d",
                record.http_id,
                method,
                path,
                record.status or "-",
                result,
                (time.monotonic() - started) * 1000,
                record.request_id,
                record.code,
                record.param,
                record.progress,
                record.replies,
            )


def request_accepted(
    request: Request, *, request_id: str, stream: bool, session_source: str, input_chars: int
) -> None:
    record = request.scope[_SCOPE_KEY]
    record.request_id = request_id
    logger.info(
        "chat.accepted http_id=%s request_id=%s stream=%s session_source=%s input_chars=%d",
        record.http_id,
        request_id,
        stream,
        session_source,
        input_chars,
    )


def message_received(request: Request, kind: str) -> None:
    record = request.scope[_SCOPE_KEY]
    if kind == "progress":
        record.progress += 1
    elif kind == "reply":
        record.replies += 1
    elif kind == "cancel":
        record.result = "cancel"
    logger.debug(
        "chat.event http_id=%s request_id=%s kind=%s",
        record.http_id,
        record.request_id,
        kind,
    )


def request_failed(request: Request, error: ProxyError) -> None:
    record = request.scope[_SCOPE_KEY]
    record.result = "error"
    record.code = error.code if isinstance(error.code, str) and error.code in _CODES else "<other>"
    record.param = _param_label(error.param)
    logger.warning(
        "request.failed http_id=%s request_id=%s http_status=%s error_status=%d code=%s param=%s",
        record.http_id,
        record.request_id,
        record.status or "-",
        error.status_code,
        record.code,
        record.param,
    )
