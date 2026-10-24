"""Opt-in HTTP body diagnostics use only mock services and placeholder credentials."""

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager

import httpx
import pytest

from muse_proxy._trace import MAX_BODY_BYTES, BodyTrace
from muse_proxy.app import create_app
from muse_proxy.config import Settings
from muse_proxy.events import MuseMessage
from muse_proxy.logging import TRACE_LEVEL
from muse_proxy.observability import RequestLoggingMiddleware

KEY = "placeholder-proxy-trace-key"
SDK_TOKEN = KEY + "-sdk-longer"
TEXT = "札幌のラーメン店を教えて"
REPLY = "味噌ラーメンのお店です。"


class Backend:
    def __init__(self, parts=None, *, failure=None, blocked=False):
        self.parts = [MuseMessage("reply", text=REPLY)] if parts is None else parts
        self.failure = failure
        self.blocked = blocked
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.calls = []

    async def stream_request(self, text, session_id=None, *, request_id):
        self.calls.append((text, session_id, request_id))
        self.started.set()
        try:
            for part in self.parts:
                yield part
            if self.blocked:
                await self.release.wait()
            if self.failure is not None:
                raise self.failure
        finally:
            self.closed.set()


@asynccontextmanager
async def proxy(tmp_path, backend=None, **settings):
    backend = backend or Backend()
    app = create_app(
        Settings(**{"api_key": KEY, "sdk_token": SDK_TOKEN, "state_dir": tmp_path, **settings}),
        client=backend,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://proxy.test",
            headers={"Authorization": f"Bearer {KEY}"},
        ) as http:
            yield http, app, backend


def payload(**options):
    return {"messages": [{"role": "user", "content": TEXT}], **options}


def records(caplog, event=None):
    return [
        record
        for record in caplog.records
        if record.name == "muse_proxy.http"
        and (event is None or record.getMessage().startswith(event + " "))
    ]


def bodies(caplog, event="http.response.body"):
    return [
        json.loads(record.getMessage().split(" body=", 1)[1]) for record in records(caplog, event)
    ]


def wire_frames(response):
    return [
        "[DONE]" if frame == "data: [DONE]" else json.loads(frame.removeprefix("data: "))
        for frame in response.text.strip().split("\n\n")
    ]


@pytest.fixture(autouse=True)
def trace_level(caplog):
    caplog.set_level(TRACE_LEVEL, logger="muse_proxy")


async def test_trace_json_includes_original_request_and_actual_response(tmp_path, caplog):
    body = payload(model="arbitrary-client-model", session_id="shared-chat")
    body["messages"].insert(0, {"role": "system", "content": "history visible in trace"})
    async with proxy(tmp_path) as (http, _, backend):
        response = await http.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert bodies(caplog, "http.request.body") == [body]
    assert bodies(caplog) == [response.json()]
    assert backend.calls == [(TEXT, "shared-chat", response.headers["x-request-id"])]
    request_record = records(caplog, "http.request.body")[0]
    response_record = records(caplog, "http.response.body")[0]
    assert request_record.levelname == response_record.levelname == "TRACE"
    assert "request_id=- format=json sequence=0" in request_record.getMessage()
    assert f"request_id={response.headers['x-request-id']} format=json sequence=1" in (
        response_record.getMessage()
    )
    assert (
        re.search(r"http_id=(http-[a-f0-9]{32})", request_record.getMessage())[1]
        == re.search(r"http_id=(http-[a-f0-9]{32})", response_record.getMessage())[1]
    )
    assert response.json()["model"] == "muse"


async def test_trace_sse_matches_progress_replies_stop_usage_and_done(tmp_path, caplog):
    backend = Backend(
        [
            MuseMessage("progress", text="検索しています", activity_code="search"),
            MuseMessage("reply", text=REPLY, message_id="reply-1"),
            MuseMessage("reply", text="追加の回答です。", message_id="reply-2"),
        ]
    )
    async with proxy(tmp_path, backend) as (http, _, _):
        response = await http.post(
            "/v1/chat/completions",
            json=payload(stream=True, stream_options={"include_usage": True}),
        )
    assert response.status_code == 200
    expected = wire_frames(response)
    assert bodies(caplog) == expected
    assert [
        part["muse_proxy"]["kind"]
        for part in expected
        if isinstance(part, dict) and "muse_proxy" in part
    ] == ["progress", "reply", "reply"]
    assert expected[-1] == "[DONE]"
    response_records = records(caplog, "http.response.body")
    assert all("format=sse" in record.getMessage() for record in response_records)
    assert [
        int(re.search(r"sequence=(\d+)", record.getMessage())[1]) for record in response_records
    ] == list(range(1, len(expected) + 1))
    assert backend.closed.is_set()


@pytest.mark.parametrize("stream", [False, True])
async def test_trace_cancel_records_actual_cancel_metadata(tmp_path, caplog, stream):
    backend = Backend([MuseMessage("cancel", superseded_by="chatcmpl-newer")])
    async with proxy(tmp_path, backend) as (http, _, _):
        response = await http.post("/v1/chat/completions", json=payload(stream=stream))
    expected = wire_frames(response) if stream else [response.json()]
    assert bodies(caplog) == expected
    cancel = next(value for value in expected if isinstance(value, dict) and "muse_proxy" in value)
    assert cancel["muse_proxy"]["kind"] == "cancel"
    assert cancel["muse_proxy"]["superseded_by"] == "chatcmpl-newer"
    assert backend.closed.is_set()


@pytest.mark.parametrize("stream", [False, True])
async def test_trace_error_records_sanitized_response_without_exception(tmp_path, caplog, stream):
    backend = Backend([], failure=RuntimeError("do-not-log-raw-backend-exception"))
    async with proxy(tmp_path, backend) as (http, _, _):
        response = await http.post("/v1/chat/completions", json=payload(stream=stream))
    expected = wire_frames(response) if stream else [response.json()]
    assert bodies(caplog) == expected
    assert any(
        isinstance(value, dict) and value.get("error", {}).get("code") == "muse_unavailable"
        for value in expected
    )
    assert "do-not-log-raw-backend-exception" not in caplog.text
    assert all(record.exc_info is None for record in records(caplog))
    assert backend.closed.is_set()


async def test_trace_logs_authenticated_request_before_protocol_rejection(tmp_path, caplog):
    body = payload(tools=[{"type": "function", "function": {"name": "unsupported-tool"}}])
    async with proxy(tmp_path) as (http, _, backend):
        response = await http.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert bodies(caplog, "http.request.body") == [body]
    assert bodies(caplog) == [response.json()]
    assert not backend.calls


async def test_trace_masks_credentials_recursively_but_does_not_modify_traffic(tmp_path, caplog):
    hidden = "nested-credential-value-never-log"
    request = payload()
    request["metadata"] = {
        "nested": [{"Authorization": hidden}, {"access_token": hidden}, {"api-key": hidden}],
        "refreshToken": hidden,
        "private_key": hidden,
        "password": hidden,
        "credentials": {"anything": hidden},
        "ordinary": "metadata remains visible",
        SDK_TOKEN: "secret in key is masked",
    }
    request["messages"][0]["content"] = f"{TEXT}: {SDK_TOKEN}, {KEY}"
    backend = Backend([MuseMessage("reply", text=f"{REPLY}: {KEY}, {SDK_TOKEN}")])
    async with proxy(tmp_path, backend) as (http, _, _):
        response = await http.post(
            "/v1/chat/completions?query=private-query-value",
            headers={"X-Private-Header": "private-header-value", "Cookie": "private-cookie-value"},
            json=request,
        )
    assert response.status_code == 200
    assert backend.calls[0][0] == request["messages"][0]["content"]
    assert SDK_TOKEN in response.json()["choices"][0]["message"]["content"]
    for forbidden in (
        KEY,
        SDK_TOKEN,
        hidden,
        "private-query-value",
        "private-header-value",
        "private-cookie-value",
    ):
        assert forbidden not in caplog.text
    logged_request = bodies(caplog, "http.request.body")[0]
    assert logged_request["messages"][0]["content"] == f"{TEXT}: [REDACTED], [REDACTED]"
    assert logged_request["metadata"]["ordinary"] == "metadata remains visible"
    assert logged_request["metadata"]["[REDACTED]"] == "secret in key is masked"
    assert "[REDACTED]-sdk-longer" not in caplog.text
    assert bodies(caplog)[0]["choices"][0]["message"]["content"] == (
        f"{REPLY}: [REDACTED], [REDACTED]"
    )


async def test_trace_does_not_read_unauthorized_request_body(tmp_path, caplog):
    async with proxy(tmp_path) as (http, _, backend):
        response = await http.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-key"},
            content=b"unauthorized-body-never-log",
        )
    assert response.status_code == 401
    assert not bodies(caplog, "http.request.body")
    assert bodies(caplog) == [response.json()]
    assert "unauthorized-body-never-log" not in caplog.text
    assert not backend.calls


async def test_debug_remains_content_free(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="muse_proxy")
    async with proxy(tmp_path) as (http, _, _):
        response = await http.post("/v1/chat/completions", json=payload())
    assert response.status_code == 200
    assert not bodies(caplog, "http.request.body")
    assert not bodies(caplog)
    assert TEXT not in caplog.text and REPLY not in caplog.text


async def test_numeric_credential_is_masked_even_in_ignored_model_and_nested_scalars(
    tmp_path, caplog
):
    body = payload(model=91743, metadata={"numeric": [91743, {"ordinary": 91743}]})
    async with proxy(tmp_path, sdk_token="91743") as (http, _, _):
        response = await http.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    logged = bodies(caplog, "http.request.body")[0]
    assert logged["model"] == "[REDACTED]"
    assert logged["metadata"] == {"numeric": ["[REDACTED]", {"ordinary": "[REDACTED]"}]}
    assert "91743" not in records(caplog, "http.request.body")[0].getMessage()


async def test_unicode_line_separators_remain_valid_sse_and_single_line_logs(tmp_path, caplog):
    reply = "日本語\u2028line separator\u2029paragraph separator\nnewline\rcarriage return"
    backend = Backend([MuseMessage("reply", text=reply)])
    async with proxy(tmp_path, backend) as (http, _, _):
        response = await http.post("/v1/chat/completions", json=payload(stream=True))
    assert response.status_code == 200
    assert bodies(caplog) == wire_frames(response)
    assert any(
        value.get("choices", [{}])[0].get("delta", {}).get("content") == reply
        for value in bodies(caplog)
        if isinstance(value, dict)
    )
    reply_log = next(
        record.getMessage()
        for record in records(caplog, "http.response.body")
        if "line separator" in record.getMessage()
    )
    assert "\\u2028" in reply_log and "\\u2029" in reply_log
    assert all(character not in reply_log for character in ("\u2028", "\u2029", "\n", "\r"))


@pytest.mark.parametrize(
    "raw,limit,status,reason",
    [
        (b'{"messages": "malformed-body-never-log"', 1024, 400, "invalid_json"),
        (b'{"messages": NaN}', 1024, 400, "invalid_json"),
        (b"oversized-body-never-log" * 10, 32, 413, "body_too_large"),
        (b"[" * 1500 + b"0" + b"]" * 1500, 4096, 400, "invalid_json"),
    ],
)
async def test_unparseable_requests_log_only_omission_marker(
    tmp_path, caplog, raw, limit, status, reason
):
    async with proxy(tmp_path, max_request_bytes=limit) as (http, _, backend):
        response = await http.post("/v1/chat/completions", content=raw)
    assert response.status_code == status
    assert bodies(caplog, "http.request.body") == [{"omitted": reason}]
    assert "malformed-body-never-log" not in caplog.text
    assert "oversized-body-never-log" not in caplog.text
    assert not backend.calls


async def test_trace_nesting_limit_does_not_break_valid_api_request(tmp_path, caplog):
    nested = "nested-private-data-never-log"
    for _ in range(70):
        nested = [nested]
    request = payload(metadata=nested)
    async with proxy(tmp_path) as (http, _, backend):
        response = await http.post("/v1/chat/completions", json=request)
    assert response.status_code == 200
    assert backend.calls[0][0] == TEXT
    assert bodies(caplog, "http.request.body") == [{"omitted": "unserializable_or_nested_body"}]
    assert bodies(caplog) == [response.json()]
    assert "nested-private-data-never-log" not in caplog.text


def test_sse_trace_handles_split_utf8_secrets_delimiters_and_multiple_frames(caplog):
    trace = BodyTrace("http-test", (KEY, SDK_TOKEN))
    trace.start_response([(b"content-type", b"text/event-stream; charset=utf-8")])
    first = json.dumps({"content": f"日本語 {SDK_TOKEN}"}, ensure_ascii=False).encode()
    wire = b"data: " + first + b"\n\n: keepalive\n\ndata: [DONE]\n\n"
    # Feed individual bytes through a full frame, then several frames in one ASGI send.
    end = wire.index(b"\n\n") + 2
    for byte in wire[:end]:
        trace.response(bytes([byte]), more_body=True)
    assert bodies(caplog) == [{"content": "日本語 [REDACTED]"}]
    trace.response(wire[end:], more_body=False)
    trace.finish()
    assert bodies(caplog) == [{"content": "日本語 [REDACTED]"}, ": keepalive", "[DONE]"]
    assert KEY not in caplog.text and SDK_TOKEN not in caplog.text


def test_json_trace_waits_for_complete_successfully_sent_body(caplog):
    trace = BodyTrace("http-test", (KEY,))
    trace.start_response([(b"content-type", b"application/json")])
    wire = json.dumps({"text": f"日本語 {KEY}"}, ensure_ascii=False).encode()
    split = wire.index("本".encode()) + 1
    trace.response(wire[:split], more_body=True)
    assert not bodies(caplog)
    trace.response(wire[split:], more_body=False)
    trace.finish()
    assert bodies(caplog) == [{"text": "日本語 [REDACTED]"}]
    assert KEY not in caplog.text


def test_sse_unknown_comment_with_credentials_does_not_fall_back_to_raw_text(caplog):
    trace = BodyTrace("http-test")
    trace.start_response([(b"content-type", b"text/event-stream")])
    trace.response(b': {"access_token":"raw-comment-credential"}\n\n', more_body=False)
    trace.finish()
    assert bodies(caplog) == [{"omitted": "invalid_json"}]
    assert "raw-comment-credential" not in caplog.text


def test_size_after_redaction_stops_following_frames_in_same_send(caplog):
    trace = BodyTrace("http-test", ("z",))
    trace.start_response([(b"content-type", b"text/event-stream")])
    first = json.dumps({"text": "z" * (MAX_BODY_BYTES // 9)}).encode()
    trace.response(
        b"data: " + first + b'\n\ndata: {"later":"not-logged-after-expansion"}\n\n',
        more_body=False,
    )
    trace.finish()
    assert bodies(caplog) == [{"omitted": "size_limit", "limit_bytes": MAX_BODY_BYTES}]
    assert "not-logged-after-expansion" not in caplog.text


@pytest.mark.parametrize("media_type", [b"application/json", b"text/event-stream"])
def test_oversized_response_logs_once_and_stops_following_frames(caplog, media_type):
    trace = BodyTrace("http-test")
    trace.start_response([(b"content-type", media_type)])
    trace.response(b"private-large-body" * (MAX_BODY_BYTES // 10), more_body=True)
    trace.response(b'\n\ndata: {"later":"also-not-logged"}\n\n', more_body=False)
    trace.finish()
    assert bodies(caplog) == [{"omitted": "size_limit", "limit_bytes": MAX_BODY_BYTES}]
    assert "private-large-body" not in caplog.text and "also-not-logged" not in caplog.text


@pytest.mark.parametrize(
    "media_type,wire,more,reason",
    [
        (b"application/json", b'{"truncated":"raw-private-body"', True, "incomplete_body"),
        (b"application/json", b"raw-private-body", False, "invalid_json"),
        (b"text/event-stream", b'data: {"value":"raw-private-body"}', False, "incomplete_body"),
        (b"text/event-stream", b"data: raw-private-body\n\n", False, "invalid_json"),
        (b"text/event-stream", b"data: \xffraw-private-body\n\n", False, "invalid_utf8"),
        (b"text/plain", b"raw-private-body", False, "unsupported_content_type"),
    ],
)
def test_response_parse_failures_never_fall_back_to_raw_text(
    caplog, media_type, wire, more, reason
):
    trace = BodyTrace("http-test")
    trace.start_response([(b"content-type", media_type)])
    trace.response(wire, more_body=more)
    trace.finish()
    assert bodies(caplog) == [{"omitted": reason}]
    assert "raw-private-body" not in caplog.text


def scope_for():
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
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


@pytest.mark.parametrize("media_type", [b"application/json", b"text/event-stream"])
async def test_failed_sends_do_not_log_unsent_response_contents(caplog, media_type):
    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", media_type)],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"content":"unsent-private-body"}',
                "more_body": False,
            }
        )

    async def unused():
        raise AssertionError("Body logging must not read the request")

    async def failing_send(message):
        if message["type"] == "http.response.body":
            raise OSError("unsent-private-body")

    with pytest.raises(OSError):
        await RequestLoggingMiddleware(app)(scope_for(), unused, failing_send)
    assert not bodies(caplog)
    assert "unsent-private-body" not in caplog.text
    assert "result=disconnected" in records(caplog)[-1].getMessage()


async def test_sse_trace_is_visible_before_stream_finishes_and_upstream_closes(tmp_path, caplog):
    progress_logged = asyncio.Event()

    class ProgressObserver(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("http.response.body ") and (
                "working-before-completion" in record.getMessage()
            ):
                progress_logged.set()

    observer = ProgressObserver()
    logger = logging.getLogger("muse_proxy.http")
    logger.addHandler(observer)
    backend = Backend([MuseMessage("progress", text="working-before-completion")], blocked=True)
    try:
        async with proxy(tmp_path, backend) as (http, _, _):
            task = asyncio.create_task(http.post("/v1/chat/completions", json=payload(stream=True)))
            try:
                await asyncio.wait_for(progress_logged.wait(), 1)
                assert not task.done()
                assert "[DONE]" not in bodies(caplog)
                assert not records(caplog, "http.response")
            finally:
                backend.release.set()
                await asyncio.wait_for(task, 1)
    finally:
        logger.removeHandler(observer)
    assert backend.closed.is_set()
    assert bodies(caplog)[-1] == "[DONE]"


async def test_trace_disconnect_keeps_upstream_cleanup_and_logs_only_sent_frames(tmp_path, caplog):
    backend = Backend([], blocked=True)
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path), client=backend)
    body = json.dumps(payload(stream=True)).encode()
    supplied = False
    sent = []

    async def receive():
        nonlocal supplied
        if not supplied:
            supplied = True
            return {"type": "http.request", "body": body, "more_body": False}
        await backend.started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async with asyncio.timeout(1):
        await app(scope_for(), receive, send)
    assert backend.closed.is_set()
    wire = b"".join(message.get("body", b"") for message in sent)
    assert len(bodies(caplog)) == wire.count(b"data: ")
    assert "[DONE]" not in bodies(caplog)
    assert "result=disconnected" in records(caplog)[-1].getMessage()
