"""Run the real OpenAI Python client against the ASGI app, without a network."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import AsyncOpenAI, AuthenticationError, InternalServerError, NotFoundError
from starlette.requests import ClientDisconnect

from muse_proxy.app import create_app
from muse_proxy.config import Settings

KEY = "test-openai-compatible-muse-proxy-key"


class FakeMuse:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.sent = []
        self.finished = []
        self.closed = False

    async def stream_chat(self, text, session_id=None):
        self.sent.append((text, session_id))
        try:
            if self.failure:
                raise self.failure
            for part in ["返信: ", text, " 🌏"]:
                await asyncio.sleep(0)
                yield part
        finally:
            self.finished.append(text)

    async def close(self):
        self.closed = True


@asynccontextmanager
async def openai_client(tmp_path, muse=None, *, key=KEY):
    muse = muse or FakeMuse()
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path), client=muse)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
            async with AsyncOpenAI(
                api_key=key,
                base_url="http://proxy.test/v1",
                http_client=http,
                max_retries=3,
            ) as client:
                yield client, muse
    assert not muse.closed


async def test_sdk_lists_and_retrieves_configured_model_without_opening_muse(tmp_path):
    async with openai_client(tmp_path) as (client, muse):
        page = await client.models.list()
        assert page.object == "list"
        assert [model.id for model in page.data] == ["muse"]
        model = await client.models.retrieve("muse")
        assert model.id == "muse"
        assert model.object == "model"
        assert model.owned_by == "muse"
    assert muse.sent == []


async def test_sdk_normal_completion_parses_reply_and_forwards_extra_body_session_id(tmp_path):
    async with openai_client(tmp_path) as (client, muse):
        completion = await client.chat.completions.create(
            model="muse",
            messages=[
                {"role": "system", "content": "Local instructions"},
                {"role": "user", "content": "Previous request"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "最新の依頼"},
            ],
            extra_body={"session_id": "side-chat-1"},
        )
    assert completion.object == "chat.completion"
    assert completion.id.startswith("chatcmpl-")
    assert completion.model == "muse"
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content == "返信: 最新の依頼 🌏"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.choices[0].logprobs is None
    assert completion.usage is None
    assert muse.sent == [("最新の依頼", "side-chat-1")]
    assert muse.finished == ["最新の依頼"]


async def test_sdk_stream_aggregates_deltas_and_accepts_unknown_usage(tmp_path):
    async with openai_client(tmp_path) as (client, muse):
        stream = await client.chat.completions.create(
            model="muse",
            messages=[{"role": "user", "content": "テキスト\n会話"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async with stream:
            chunks = [chunk async for chunk in stream]
    assert chunks[0].choices[0].delta.role == "assistant"
    assert (
        "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
        == "返信: テキスト\n会話 🌏"
    )
    assert len({chunk.id for chunk in chunks}) == 1
    assert all(chunk.model == "muse" for chunk in chunks)
    assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
    assert [
        choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason
    ] == ["stop"]
    assert chunks[-1].choices == []
    assert chunks[-1].usage is None
    assert muse.sent == [("テキスト\n会話", None)]
    assert muse.finished == ["テキスト\n会話"]


async def test_sdk_maps_invalid_proxy_key_to_authentication_error(tmp_path):
    async with openai_client(tmp_path, key="wrong-proxy-api-key") as (client, muse):
        with pytest.raises(AuthenticationError) as failure:
            await client.chat.completions.create(
                model="muse", messages=[{"role": "user", "content": "Do not send"}]
            )
    assert failure.value.status_code == 401
    assert failure.value.code == "invalid_api_key"
    assert failure.value.response.headers["x-should-retry"] == "false"
    assert muse.sent == []


async def test_sdk_maps_unknown_model_to_not_found_error_without_sending(tmp_path):
    async with openai_client(tmp_path) as (client, muse):
        with pytest.raises(NotFoundError) as failure:
            await client.chat.completions.create(
                model="missing", messages=[{"role": "user", "content": "Do not send"}]
            )
    assert failure.value.status_code == 404
    assert failure.value.code == "model_not_found"
    assert muse.sent == []


async def test_sdk_honors_no_retry_header_for_upstream_error(tmp_path):
    muse = FakeMuse(failure=RuntimeError("private-token-and-upstream-host"))
    async with openai_client(tmp_path, muse) as (client, _):
        with pytest.raises(InternalServerError) as failure:
            await client.chat.completions.create(
                model="muse", messages=[{"role": "user", "content": "Once only"}]
            )
    assert failure.value.status_code == 502
    assert failure.value.code == "muse_unavailable"
    assert failure.value.response.headers["x-should-retry"] == "false"
    assert "private-token-and-upstream-host" not in str(failure.value)
    assert muse.sent == [("Once only", None)], "SDK max_retries=3 must not duplicate a Muse request"
    assert muse.finished == ["Once only"]


@pytest.mark.parametrize("asgi_version", ["2.3", "2.4"])
async def test_sse_write_failure_closes_iterator_after_partial_reply(tmp_path, asgi_version):
    """A lost socket during send must close both generator layers and their tasks."""

    class WaitingMuse(FakeMuse):
        async def stream_chat(self, text, session_id=None):
            self.sent.append((text, session_id))
            try:
                yield "partial reply"
                await asyncio.Event().wait()
            finally:
                self.finished.append(text)

    muse = WaitingMuse()
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path), client=muse)
    body = json.dumps(
        {"model": "muse", "messages": [{"role": "user", "content": "Once"}], "stream": True}
    ).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": asgi_version},
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
    written = []

    async def receive():
        nonlocal supplied
        if not supplied:
            supplied = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        written.append(message)
        if b"partial reply" in message.get("body", b""):
            raise OSError("Downstream socket closed")

    async with app.router.lifespan_context(app):
        async with asyncio.timeout(1):
            with pytest.raises(ClientDisconnect):
                await app(scope, receive, send)
    assert muse.sent == [("Once", None)]
    assert muse.finished == ["Once"]
    assert not muse.closed
    assert any(b"partial reply" in message.get("body", b"") for message in written)
