"""OpenAI-compatible HTTP boundary for a single Muse account."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from uuid import uuid4

from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .config import Settings
from .errors import ProxyError
from .events import MuseMessage
from .protocol import ChatRequest, chat_chunk, chat_completion, message_metadata, parse_chat

logger = logging.getLogger(__name__)
HEARTBEAT_SECONDS = 15.0


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _frame(value: dict) -> str:
    return f"data: {_json(value)}\n\n"


def _safe_error(error: Exception) -> ProxyError:
    if isinstance(error, ProxyError):
        return error
    if isinstance(error, TimeoutError):
        return ProxyError(
            "Timed out waiting for Muse. The request may still be running; do not resend it "
            "without checking Muse.",
            status_code=504,
            code="muse_reply_timeout",
        )
    # Network libraries can include bearer tokens, remote URLs or user text in
    # their exception messages. They must not reach the HTTP client or logs.
    return ProxyError(
        "Muse communication failed. The request may have been accepted; "
        "check Muse before retrying.",
        code="muse_unavailable",
    )


def _error_response(error: Exception, request_id: str | None = None) -> JSONResponse:
    safe = _safe_error(error)
    headers = {"x-should-retry": "false", "Cache-Control": "no-store"}
    if safe.status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    if request_id:
        headers["x-request-id"] = request_id
    return JSONResponse({"error": safe.as_dict()}, status_code=safe.status_code, headers=headers)


def _authorize(request: Request, settings: Settings) -> None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        token.encode(), settings.api_key.encode()
    ):
        raise ProxyError("Invalid API key.", status_code=401, code="invalid_api_key")
    origin = request.headers.get("origin")
    if origin is not None:
        expected = f"{request.url.scheme}://{request.url.netloc}"
        if origin.rstrip("/") != expected:
            raise ProxyError(
                "Cross-origin requests are not allowed.", status_code=403, code="origin_rejected"
            )


def _invalid_constant(value: str) -> None:
    raise ValueError("Non-finite JSON")


async def _payload(request: Request, max_bytes: int) -> dict:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ProxyError("Request body is too large.", status_code=413, code="body_too_large")
    try:
        return json.loads(body, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ProxyError(
            "Expected a valid JSON object.", status_code=400, code="invalid_request_error"
        ) from None


async def _disconnected(request: Request) -> None:
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


async def _reply_parts(
    request: Request,
    client,
    chat: ChatRequest,
    timeout_seconds: float,
    *,
    request_id: str,
) -> AsyncIterator[MuseMessage | None]:
    """Race every upstream wait against downstream disconnect and a total deadline.

    None is a keepalive. Both JSON and SSE paths close the upstream iterator,
    including when a response write raises or the ASGI task is cancelled.
    """
    watcher = asyncio.create_task(_disconnected(request))
    pending = None
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)

    async def produce() -> None:
        # One task owns the complete upstream iterator. Its timeout contexts
        # and cancellation bookkeeping must not cross tasks between chunks.
        try:
            stream_request = getattr(client, "stream_request", None)
            stream_events = getattr(client, "stream_events", None)
            if callable(stream_request):
                source = stream_request(
                    chat.text, session_id=chat.session_id, request_id=request_id
                )
            elif callable(stream_events):
                source = stream_events(chat.text, session_id=chat.session_id)
            else:
                source = client.stream_chat(chat.text, session_id=chat.session_id)
            async with aclosing(source) as stream:
                async for part in stream:
                    if isinstance(part, str):
                        part = MuseMessage("reply", text=part)
                    if (
                        not isinstance(part, MuseMessage)
                        or part.kind not in {"progress", "reply", "cancel"}
                        or not isinstance(part.text, str)
                    ):
                        raise ProxyError(
                            "Muse returned an invalid text response.", code="muse_protocol_error"
                        )
                    await queue.put(("part", part))
                    if part.kind == "cancel":
                        return
        except Exception as exc:
            await queue.put(("error", exc))
        else:
            await queue.put(("done", None))

    producer = asyncio.create_task(produce())
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    try:
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError
            if pending is None:
                pending = asyncio.create_task(queue.get())
            waiting = {pending, watcher}
            if not producer.done():
                waiting.add(producer)
            done, _ = await asyncio.wait(
                waiting,
                timeout=min(HEARTBEAT_SECONDS, remaining),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if watcher in done:
                raise ClientDisconnect
            if producer.done() and producer.cancelled():
                raise ProxyError("Muse proxy is shutting down.", status_code=503)
            if pending in done:
                kind, part = pending.result()
                if kind == "done":
                    return
                if kind == "error":
                    raise part
                pending = None
                if part.kind != "reply" or part.text:
                    yield part
                if part.kind == "cancel":
                    return
            else:
                yield None
    finally:
        tasks = [watcher, producer, *([pending] if pending is not None else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class _EventStreamResponse(StreamingResponse):
    """The generator owns disconnect observation on every ASGI version."""

    async def __call__(self, scope, receive, send) -> None:
        try:
            await self.stream_response(send)
        except OSError:
            raise ClientDisconnect from None
        finally:
            if hasattr(self.body_iterator, "aclose"):
                await self.body_iterator.aclose()


def create_app(settings: Settings, *, client=None) -> Starlette:
    """Build the proxy. Injected clients remain owned by their caller.

    Initial pairing is deliberately external. Creating the app and calling
    /health or /v1/models does not open credential files or contact Muse.
    """
    owned = client is None
    if owned:
        from .muse import MuseClient

        client = MuseClient(settings)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if owned:
                await client.close()

    async def health(request):
        return JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})

    def model_object() -> dict:
        return {"id": settings.model, "object": "model", "created": 0, "owned_by": "muse"}

    async def models(request: Request):
        _authorize(request, settings)
        return JSONResponse({"object": "list", "data": [model_object()]})

    async def model(request: Request):
        _authorize(request, settings)
        if request.path_params["model"] != settings.model:
            raise ProxyError(
                "The requested model is not available.", status_code=404, code="model_not_found"
            )
        return JSONResponse(model_object())

    async def completions(request: Request):
        _authorize(request, settings)
        payload = await _payload(request, settings.max_request_bytes)
        chat = parse_chat(payload, settings.model, max_message_chars=settings.max_message_chars)
        identity = {
            "id": f"chatcmpl-{uuid4().hex}",
            "created": int(time.time()),
            "model": chat.model,
        }
        headers = {
            "x-request-id": identity["id"],
            "x-should-retry": "false",
            "Cache-Control": "no-store",
        }
        parts = _reply_parts(
            request,
            client,
            chat,
            settings.request_timeout_seconds,
            request_id=identity["id"],
        )
        if not chat.stream:
            try:
                replies = []
                sequence = 0
                async for part in parts:
                    if part is None:
                        continue
                    sequence += 1
                    if part.kind == "cancel":
                        result = chat_completion("", **identity)
                        result["muse_proxy"] = message_metadata(
                            part, request_id=identity["id"], sequence=sequence
                        )
                        return JSONResponse(result, headers=headers)
                    if part.kind == "reply":
                        replies.append(part.text)
                text = "".join(replies)
                if not text:
                    raise ProxyError(
                        "Muse completed without a text reply.", code="muse_empty_reply"
                    )
                return JSONResponse(chat_completion(text, **identity), headers=headers)
            except ClientDisconnect:
                return JSONResponse({}, status_code=499, headers=headers)
            except Exception as exc:
                return _error_response(exc, identity["id"])
            finally:
                await parts.aclose()

        async def events():
            received_text = False
            cancelled = False
            sequence = 0
            try:
                yield _frame(chat_chunk({"role": "assistant", "content": ""}, **identity))
                async for part in parts:
                    if part is None:
                        yield ": keepalive\n\n"
                    else:
                        sequence += 1
                        text = part.text
                        if part.kind == "progress" and text:
                            text += "\n\n"
                        elif part.kind == "cancel":
                            text = ""
                            cancelled = True
                        elif part.kind == "reply" and text:
                            received_text = True
                        chunk = chat_chunk({"content": text}, **identity)
                        chunk["muse_proxy"] = message_metadata(
                            part, request_id=identity["id"], sequence=sequence
                        )
                        yield _frame(chunk)
                        if cancelled:
                            break
                if not received_text and not cancelled:
                    raise ProxyError(
                        "Muse completed without a text reply.", code="muse_empty_reply"
                    )
                yield _frame(chat_chunk({}, **identity, finish_reason="stop"))
                if chat.include_usage:
                    yield _frame(
                        {
                            **{
                                key: value
                                for key, value in chat_chunk({}, **identity).items()
                                if key != "choices"
                            },
                            "choices": [],
                            "usage": None,
                        }
                    )
                yield "data: [DONE]\n\n"
            except ClientDisconnect:
                return
            except Exception as exc:
                yield _frame({"error": _safe_error(exc).as_dict()})
                yield "data: [DONE]\n\n"
            finally:
                await parts.aclose()

        return _EventStreamResponse(
            events(),
            media_type="text/event-stream",
            headers={**headers, "X-Accel-Buffering": "no"},
        )

    async def proxy_error(request, error):
        return _error_response(error)

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/models/{model}", model),
            Route("/v1/chat/completions", completions, methods=["POST"]),
        ],
        lifespan=lifespan,
        exception_handlers={ProxyError: proxy_error},
    )
    app.state.client = client
    return app
