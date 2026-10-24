"""Exercise the public HTTP interface without Muse credentials or live services."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from muse_proxy.app import create_app
from muse_proxy.config import Settings
from muse_proxy.errors import ProxyError

KEY = "test-api-key-for-muse-proxy"
AUTH = {"Authorization": f"Bearer {KEY}"}


def chat_payload(text="こんにちは🌏", **options):
    return {"messages": [{"role": "user", "content": text}], **options}


class FakeMuse:
    """Each iterator owns its result and cleanup, even when requests overlap."""

    def __init__(self, *, chunks=None, failures=None, blocked=()):
        self.chunks = chunks
        self.failures = failures or {}
        self.blocked = set(blocked)
        self.sent = []
        self.finished = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def stream_chat(self, text, session_id=None):
        self.sent.append((text, session_id))
        self.started.set()
        try:
            if text in self.blocked:
                await self.release.wait()
            if failure := self.failures.get(text):
                raise failure
            for chunk in self.chunks if self.chunks is not None else ["返信: ", text]:
                await asyncio.sleep(0)
                yield chunk
        finally:
            self.finished.append(text)

    async def close(self):
        self.closed = True


@asynccontextmanager
async def proxy(tmp_path, muse=None, **options):
    muse = muse or FakeMuse()
    settings = Settings(api_key=KEY, state_dir=tmp_path, model="muse", **options)
    app = create_app(settings, client=muse)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy.test",
            headers=AUTH,
        ) as http:
            yield http, muse
    assert not muse.closed, "Injected clients remain caller-owned"


def sse_frames(response):
    return [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]


async def test_health_is_public_and_does_not_expose_state_directory(tmp_path):
    async with proxy(tmp_path) as (http, muse):
        response = await http.get("/health", headers={"Authorization": ""})
    assert response.status_code == 200
    assert str(tmp_path) not in response.text
    assert KEY not in response.text
    assert muse.sent == []


async def test_server_runs_without_initial_authentication_and_reports_missing_state(tmp_path):
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test", headers=AUTH
        ) as http:
            assert (await http.get("/health")).status_code == 200
            assert (await http.get("/v1/models")).status_code == 200
            response = await http.post("/v1/chat/completions", json=chat_payload())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "muse_credentials_error"
    assert str(tmp_path) not in response.text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "authorization", ["", "Bearer wrong-key", "Basic " + KEY, "Bearer", "Bearer " + KEY + " extra"]
)
@pytest.mark.parametrize("path", ["/v1/models", "/v1/chat/completions"])
async def test_authentication_fails_before_upstream_requests(tmp_path, authorization, path):
    async with proxy(tmp_path) as (http, muse):
        options = {"headers": {"Authorization": authorization}}
        response = (
            await http.get(path, **options)
            if path == "/v1/models"
            else await http.post(path, json=chat_payload(), **options)
        )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    assert KEY not in response.text
    assert muse.sent == []


async def test_models_uses_configured_public_model_and_case_insensitive_bearer_scheme(tmp_path):
    async with proxy(tmp_path) as (http, muse):
        response = await http.get("/v1/models", headers={"Authorization": "bearer " + KEY})
    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert [model["id"] for model in response.json()["data"]] == ["muse"]
    assert muse.sent == []


async def test_normal_completion_joins_deltas_and_forwards_only_latest_user(tmp_path):
    payload = chat_payload(
        messages=[
            {"role": "system", "content": "Private local instructions"},
            {"role": "user", "content": "Previous request"},
            {"role": "assistant", "content": "Previous response"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "最新"}, {"type": "text", "text": "依頼🌏"}],
            },
        ],
        session_id="side-chat-1",
        temperature=0.1,
        top_p=0.2,
        max_tokens=1,
    )
    async with proxy(tmp_path) as (http, muse):
        response = await http.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["id"].startswith("chatcmpl-")
    assert result["object"] == "chat.completion"
    assert result["model"] == "muse"
    assert isinstance(result["created"], int)
    assert result["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "返信: 最新\n依頼🌏"},
            "finish_reason": "stop",
            "logprobs": None,
        }
    ]
    assert "usage" not in result
    assert muse.sent == [("最新\n依頼🌏", "side-chat-1")]
    assert muse.finished == ["最新\n依頼🌏"]


@pytest.mark.parametrize("include_usage", [False, True])
async def test_sse_preserves_identity_unicode_order_and_terminal_frame(tmp_path, include_usage):
    muse = FakeMuse(chunks=["一行目\n", "こんにちは", "🌏"])
    async with proxy(tmp_path, muse) as (http, _):
        response = await http.post(
            "/v1/chat/completions",
            json=chat_payload(stream=True, stream_options={"include_usage": include_usage}),
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = sse_frames(response)
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert (
        "".join(
            choice["delta"].get("content", "") for chunk in chunks for choice in chunk["choices"]
        )
        == "一行目\nこんにちは🌏"
    )
    finish_chunks = [
        chunk for chunk in chunks if chunk["choices"] and chunk["choices"][0]["finish_reason"]
    ]
    assert len(finish_chunks) == 1
    assert finish_chunks[0]["choices"][0]["finish_reason"] == "stop"
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert len({chunk["created"] for chunk in chunks}) == 1
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert all(chunk["model"] == "muse" for chunk in chunks)
    if include_usage:
        assert all(chunk.get("usage") is None for chunk in chunks)
        assert any("usage" in chunk for chunk in chunks)
    else:
        assert all("usage" not in chunk for chunk in chunks)


async def test_unknown_model_returns_404_before_sending(tmp_path):
    async with proxy(tmp_path) as (http, muse):
        response = await http.post("/v1/chat/completions", json=chat_payload(model="other"))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert muse.sent == []


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        b"[]",
        b'{"messages":NaN}',
        b'{"messages":[{"role":"user","content":"\\ud800"}]}',
    ],
)
async def test_malformed_json_and_unicode_never_send_requests(tmp_path, raw):
    async with proxy(tmp_path) as (http, muse):
        response = await http.post(
            "/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 400
    assert "error" in response.json()
    assert muse.sent == []


async def test_body_limit_rejects_oversized_history_even_when_latest_text_is_short(tmp_path):
    payload = chat_payload(
        messages=[
            {"role": "assistant", "content": "x" * (300 * 1024)},
            {"role": "user", "content": "small"},
        ]
    )
    async with proxy(tmp_path) as (http, muse):
        response = await http.post("/v1/chat/completions", json=payload)
    assert response.status_code == 413
    assert "error" in response.json()
    assert muse.sent == []


@pytest.mark.parametrize(
    "failure,expected_status,expected_code",
    [
        (ProxyError("Muse is busy", status_code=429, code="muse_busy"), 429, "muse_busy"),
        (RuntimeError("secret-token-and-private-server"), 502, None),
    ],
)
async def test_upstream_failure_is_safe_and_disables_automatic_retry(
    tmp_path, failure, expected_status, expected_code
):
    muse = FakeMuse(failures={"fail": failure})
    async with proxy(tmp_path, muse) as (http, _):
        response = await http.post("/v1/chat/completions", json=chat_payload("fail"))
    assert response.status_code == expected_status
    assert response.headers["x-should-retry"] == "false"
    error = response.json()["error"]
    assert isinstance(error["message"], str)
    if expected_code:
        assert error["code"] == expected_code
    assert "secret-token-and-private-server" not in response.text
    assert muse.sent == [("fail", None)]
    assert muse.finished == ["fail"]


async def test_sse_failure_after_partial_text_is_safe_and_never_claims_success(tmp_path):
    class FailingMuse(FakeMuse):
        async def stream_chat(self, text, session_id=None):
            self.sent.append((text, session_id))
            try:
                yield "partial response"
                raise RuntimeError("secret-token-and-private-server")
            finally:
                self.finished.append(text)

    muse = FailingMuse()
    async with proxy(tmp_path, muse) as (http, _):
        response = await http.post("/v1/chat/completions", json=chat_payload(stream=True))
    assert response.status_code == 200
    assert response.headers["x-should-retry"] == "false"
    assert "secret-token-and-private-server" not in response.text
    frames = sse_frames(response)
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert "error" in chunks[-1]
    assert any(
        choice["delta"].get("content") == "partial response"
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert not any(
        choice.get("finish_reason") == "stop"
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert len(muse.sent) == 1
    assert muse.finished == ["こんにちは🌏"]


@pytest.mark.parametrize("stream", [False, True])
async def test_timeout_closes_iterator_and_does_not_resend(tmp_path, stream):
    muse = FakeMuse(blocked={"pending"})
    async with proxy(tmp_path, muse, request_timeout_seconds=0.02) as (http, _):
        response = await http.post(
            "/v1/chat/completions", json=chat_payload("pending", stream=stream)
        )
    assert response.headers["x-should-retry"] == "false"
    if stream and response.status_code == 200:
        frames = sse_frames(response)
        assert frames[-1] == "[DONE]"
        assert "error" in json.loads(frames[-2])
    else:
        assert response.status_code == 504
        assert "error" in response.json()
    assert muse.sent == [("pending", None)]
    assert muse.finished == ["pending"]


async def test_concurrent_requests_do_not_share_response_text_or_completion_id(tmp_path):
    muse = FakeMuse(blocked={"first"})
    async with proxy(tmp_path, muse) as (http, _):
        first = asyncio.create_task(http.post("/v1/chat/completions", json=chat_payload("first")))
        try:
            async with asyncio.timeout(1):
                await muse.started.wait()
                second = await http.post("/v1/chat/completions", json=chat_payload("second"))
            assert not first.done()
            assert second.json()["choices"][0]["message"]["content"] == "返信: second"
        finally:
            muse.release.set()
        async with asyncio.timeout(1):
            first_result = await first
    assert first_result.json()["choices"][0]["message"]["content"] == "返信: first"
    assert first_result.json()["id"] != second.json()["id"]


async def test_request_cancellation_closes_pending_upstream_iterator(tmp_path):
    muse = FakeMuse(blocked={"cancel-me"})
    async with proxy(tmp_path, muse) as (http, _):
        pending = asyncio.create_task(
            http.post("/v1/chat/completions", json=chat_payload("cancel-me"))
        )
        async with asyncio.timeout(1):
            await muse.started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert muse.sent == [("cancel-me", None)]
    assert muse.finished == ["cancel-me"]


@pytest.mark.parametrize("stream", [False, True])
async def test_downstream_disconnect_cancels_upstream_without_a_second_send(tmp_path, stream):
    muse = FakeMuse(blocked={"disconnect-me"})
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path, model="muse"), client=muse)
    body = json.dumps(chat_payload("disconnect-me", stream=stream)).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
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
    supplied = False

    async def receive():
        nonlocal supplied
        if not supplied:
            supplied = True
            return {"type": "http.request", "body": body, "more_body": False}
        await muse.started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    async with app.router.lifespan_context(app):
        async with asyncio.timeout(1):
            await app(scope, receive, send)
    assert muse.sent == [("disconnect-me", None)]
    assert muse.finished == ["disconnect-me"]
