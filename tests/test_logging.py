"""Safe diagnostics for third-party HTTP clients, using no live credentials or services."""

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager

import httpx
import pytest

from muse_proxy.app import create_app
from muse_proxy.config import Settings
from muse_proxy.errors import ProxyError
from muse_proxy.events import MuseMessage
from muse_proxy.observability import RequestLoggingMiddleware

KEY = "test-proxy-logging-placeholder"
PRIVATE = "PRIVATE-SENTINEL-9c14b"
AUTH = {"Authorization": f"Bearer {KEY}"}


class Backend:
    def __init__(self, *, parts=None, failure=None, blocked=False):
        self.parts = [MuseMessage("reply", text=PRIVATE)] if parts is None else parts
        self.failure = failure
        self.blocked = blocked
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.calls = 0

    async def stream_request(self, text, session_id=None, *, request_id):
        self.calls += 1
        self.started.set()
        try:
            for part in self.parts:
                yield part
            if self.blocked:
                await self.release.wait()
            if self.failure:
                raise self.failure
        finally:
            self.closed.set()


@asynccontextmanager
async def proxy(tmp_path, backend=None, **settings):
    backend = backend or Backend()
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path, **settings), client=backend)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://proxy.test",
            headers=AUTH,
        ) as http:
            yield http, app, backend


def payload(**options):
    return {"messages": [{"role": "user", "content": PRIVATE}], **options}


def messages(caplog):
    return [record.getMessage() for record in caplog.records if record.name == "muse_proxy.http"]


def assert_safe(caplog):
    for record in caplog.records:
        if record.name.startswith("muse_proxy"):
            assert PRIVATE not in record.getMessage()
            assert KEY not in record.getMessage()
            assert record.exc_info is None


@pytest.mark.parametrize("stream", [False, True])
async def test_success_logs_arrival_identity_counts_and_full_response(tmp_path, caplog, stream):
    caplog.set_level(logging.DEBUG, logger="muse_proxy")
    backend = Backend(
        parts=[
            MuseMessage("progress", text=PRIVATE, activity_code=PRIVATE),
            MuseMessage("reply", text=PRIVATE, message_id=PRIVATE),
            MuseMessage("reply", text="second reply"),
        ]
    )
    async with proxy(tmp_path, backend) as (http, _, _):
        response = await http.post(
            "/v1/chat/completions?key=" + PRIVATE,
            headers={"X-Request-ID": PRIVATE},
            json=payload(model=PRIVATE, stream=stream, session_id=PRIVATE),
        )
    assert response.status_code == 200
    logs = messages(caplog)
    assert sum(line.startswith("http.request ") for line in logs) == 1
    assert sum(line.startswith("http.response ") for line in logs) == 1
    ids = {re.search(r"http_id=(http-[a-f0-9]{32})", line)[1] for line in logs}
    assert len(ids) == 1
    request_id = response.headers["x-request-id"]
    assert all(f"request_id={request_id}" in line for line in logs[1:])
    assert "status=200 result=ok" in logs[-1]
    assert "progress=1 replies=2" in logs[-1]
    assert "duration_ms=" in logs[-1]
    assert "session_source=request" in logs[1]
    assert_safe(caplog)


@pytest.mark.parametrize(
    "case,status,code,param",
    [
        ("auth", 401, "invalid_api_key", "-"),
        ("origin", 403, "origin_rejected", "-"),
        ("tools", 400, "invalid_request_error", "tools"),
        ("messages", 400, "invalid_request_error", "messages"),
        ("content", 400, "invalid_request_error", "messages[0].content"),
    ],
)
async def test_rejected_requests_log_safe_code_and_parameter_without_changing_headers(
    tmp_path,
    caplog,
    case,
    status,
    code,
    param,
):
    caplog.set_level(logging.DEBUG, logger="muse_proxy")
    body = payload()
    headers = {}
    if case == "auth":
        headers["Authorization"] = "Bearer " + PRIVATE
    elif case == "origin":
        headers["Origin"] = "https://" + PRIVATE + ".example"
    elif case == "tools":
        body["tools"] = [{"type": "function", "function": {"name": PRIVATE}}]
    elif case == "messages":
        body["messages"] = PRIVATE
    else:
        body["messages"][0]["content"] = [{"type": "image_url", "image_url": PRIVATE}]
    async with proxy(tmp_path) as (http, _, backend):
        response = await http.post("/v1/chat/completions", json=body, headers=headers)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "x-request-id" not in response.headers
    assert backend.calls == 0
    logs = messages(caplog)
    assert f"code={code} param={param}" in logs[-1]
    assert "request_id=-" in logs[-1]
    assert any(line.startswith("request.failed ") for line in logs)
    assert_safe(caplog)


@pytest.mark.parametrize(
    "path,label",
    [
        ("/" + PRIVATE, "<unmatched>"),
        ("/v1/models/" + PRIVATE, "/v1/models/{model}"),
        ("/v1/responses", "/v1/responses"),
        ("/chat/completions", "/chat/completions"),
        ("/v1/v1/chat/completions", "/v1/v1/chat/completions"),
    ],
)
async def test_paths_and_query_values_are_redacted_but_common_url_mistakes_are_visible(
    tmp_path,
    caplog,
    path,
    label,
):
    caplog.set_level(logging.DEBUG, logger="muse_proxy")
    async with proxy(tmp_path) as (http, _, _):
        response = await http.get(path + "?token=" + PRIVATE)
    assert response.status_code == 404
    assert all(f"path={label}" in line for line in messages(caplog) if line.startswith("http."))
    assert_safe(caplog)


async def test_unknown_method_is_redacted_and_browser_preflight_status_is_logged(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="muse_proxy")
    async with proxy(tmp_path) as (http, _, _):
        for method in (PRIVATE, "OPTIONS"):
            assert (await http.request(method, "/v1/chat/completions")).status_code == 405
    logs = messages(caplog)
    assert any("method=OTHER" in line and "status=405" in line for line in logs)
    assert any("method=OPTIONS" in line and "status=405" in line for line in logs)
    assert_safe(caplog)


@pytest.mark.parametrize("stream", [False, True])
async def test_upstream_error_is_logged_even_when_sse_http_status_is_200(tmp_path, caplog, stream):
    caplog.set_level(logging.DEBUG, logger="muse_proxy")
    backend = Backend(parts=[], failure=RuntimeError(PRIVATE))
    async with proxy(tmp_path, backend) as (http, _, _):
        response = await http.post("/v1/chat/completions", json=payload(stream=stream))
    status = 200 if stream else 502
    assert response.status_code == status
    logs = messages(caplog)
    assert any(
        "code=muse_unavailable" in line and line.startswith("request.failed ") for line in logs
    )
    assert f"status={status} result=error" in logs[-1]
    if stream:
        assert any("http_status=200 error_status=502" in line for line in logs)
        assert '"error":' in response.text and "data: [DONE]" in response.text
    assert backend.closed.is_set()
    assert backend.calls == 1
    assert_safe(caplog)


async def test_arbitrary_error_code_and_parameter_are_not_logged(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="muse_proxy")
    backend = Backend(parts=[], failure=ProxyError(PRIVATE, code=PRIVATE, param=PRIVATE))
    async with proxy(tmp_path, backend) as (http, _, _):
        await http.post("/v1/chat/completions", json=payload())
    assert "code=<other> param=<other>" in messages(caplog)[-1]
    assert_safe(caplog)


@pytest.mark.parametrize("stream", [False, True])
async def test_cancel_is_logged_separately_from_success_or_error(tmp_path, caplog, stream):
    caplog.set_level(logging.DEBUG, logger="muse_proxy")
    backend = Backend(
        parts=[MuseMessage("reply", text=PRIVATE), MuseMessage("cancel", superseded_by=PRIVATE)]
    )
    async with proxy(tmp_path, backend) as (http, _, _):
        response = await http.post("/v1/chat/completions", json=payload(stream=stream))
    assert response.status_code == 200
    assert "status=200 result=cancel" in messages(caplog)[-1]
    assert not any(line.startswith("request.failed ") for line in messages(caplog))
    assert backend.closed.is_set()
    assert_safe(caplog)


@pytest.mark.parametrize("stream", [False, True])
async def test_timeout_is_logged_and_closes_upstream(tmp_path, caplog, stream):
    caplog.set_level(logging.DEBUG, logger="muse_proxy")
    backend = Backend(parts=[MuseMessage("progress", text=PRIVATE)], blocked=True)
    async with proxy(tmp_path, backend, request_timeout_seconds=0.01) as (http, _, _):
        response = await http.post("/v1/chat/completions", json=payload(stream=stream))
    assert response.status_code == (200 if stream else 504)
    assert "result=error" in messages(caplog)[-1]
    assert "code=muse_reply_timeout" in messages(caplog)[-1]
    assert backend.closed.is_set()
    assert backend.calls == 1
    assert_safe(caplog)


def scope_for(body, spec="2.4"):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "server": ("proxy.test", 80),
        "client": ("127.0.0.1", 12345),
        "headers": [(b"authorization", ("Bearer " + KEY).encode()), (b"host", b"proxy.test")],
    }


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
@pytest.mark.parametrize("stream", [False, True])
async def test_disconnect_logging_preserves_upstream_cleanup(tmp_path, caplog, spec, stream):
    caplog.set_level(logging.DEBUG, logger="muse_proxy")
    backend = Backend(parts=[], blocked=True)
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path), client=backend)
    body = json.dumps(payload(stream=stream)).encode()
    supplied = False

    async def receive():
        nonlocal supplied
        if not supplied:
            supplied = True
            return {"type": "http.request", "body": body, "more_body": False}
        await backend.started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    async with asyncio.timeout(1):
        await app(scope_for(body, spec), receive, send)
    assert backend.closed.is_set()
    assert backend.calls == 1
    assert "result=disconnected" in messages(caplog)[-1]
    assert sum(line.startswith("http.response ") for line in messages(caplog)) == 1
    assert_safe(caplog)


async def test_arrival_is_visible_while_stream_is_waiting_and_finish_waits_for_body(
    tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger="muse_proxy")
    backend = Backend(blocked=True)
    async with proxy(tmp_path, backend) as (http, _, _):
        task = asyncio.create_task(http.post("/v1/chat/completions", json=payload(stream=True)))
        try:
            await asyncio.wait_for(backend.started.wait(), 1)
            assert any(line.startswith("http.request ") for line in messages(caplog))
            assert any(line.startswith("chat.accepted ") for line in messages(caplog))
            assert not any(line.startswith("http.response ") for line in messages(caplog))
            assert not task.done()
        finally:
            backend.release.set()
            await asyncio.wait_for(task, 1)
    assert "status=200 result=ok" in messages(caplog)[-1]
    assert_safe(caplog)


async def test_middleware_records_failed_send_as_disconnect_without_logging_exception(caplog):
    caplog.set_level(logging.INFO, logger="muse_proxy")

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError(PRIVATE)

    with pytest.raises(OSError):
        await RequestLoggingMiddleware(app)(scope_for(b""), receive, send)
    assert "result=disconnected" in messages(caplog)[-1]
    assert_safe(caplog)


async def test_disconnect_during_terminal_send_is_observed(caplog):
    caplog.set_level(logging.INFO, logger="muse_proxy")
    terminal = asyncio.Event()
    disconnected = asyncio.Event()

    async def app(scope, receive, send):
        async def observe_disconnect():
            await terminal.wait()
            await receive()
            disconnected.set()

        observer = asyncio.create_task(observe_disconnect())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        await observer

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            terminal.set()
            await disconnected.wait()

    async with asyncio.timeout(1):
        await RequestLoggingMiddleware(app)(scope_for(b""), receive, send)
    assert "result=disconnected" in messages(caplog)[-1]


async def test_path_labels_are_bounded_even_with_many_trailing_slashes(caplog):
    caplog.set_level(logging.INFO, logger="muse_proxy")

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 307, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def unused():
        raise AssertionError("Logging must not consume the request")

    async def send(message):
        pass

    scope = scope_for(b"")
    scope["path"] = "/health" + "/" * 2000
    await RequestLoggingMiddleware(app)(scope, unused, send)
    assert all("path=/health" in line and len(line) < 500 for line in messages(caplog))
