"""Structured Muse events cross the public Chat Completions boundary safely."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import AsyncOpenAI

from muse_proxy.app import create_app
from muse_proxy.config import Settings
from muse_proxy.events import MuseMessage

KEY = "test-progress-cancel-proxy-key"
AUTH = {"Authorization": "Bearer " + KEY}


def payload(*, stream=False):
    return {
        "model": "muse",
        "session_id": "progress-chat",
        "messages": [{"role": "user", "content": "調べてください🌏"}],
        "stream": stream,
    }


class EventMuse:
    """Track consumption and generator cleanup independently of HTTP delivery."""

    def __init__(self, events, *, failure=None):
        self.events = events
        self.failure = failure
        self.sent = []
        self.finished = []
        self.advanced_after_cancel = False
        self.closed = False

    async def stream_events(self, text, session_id=None):
        self.sent.append((text, session_id))
        try:
            for event in self.events:
                yield event
                if event.kind == "cancel":
                    self.advanced_after_cancel = True
                    raise AssertionError("The cancelled upstream iterator must not resume")
            if self.failure:
                raise self.failure
        finally:
            self.finished.append(text)

    async def stream_chat(self, text, session_id=None):
        raise AssertionError("Structured events must take precedence over text-only streaming")
        yield  # pragma: no cover

    async def close(self):
        self.closed = True


@asynccontextmanager
async def proxy(tmp_path, muse, **options):
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path, **options), client=muse)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy.test",
            headers=AUTH,
        ) as http:
            yield http
    assert not muse.closed, "An injected client remains caller-owned"


def frames(text):
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


def chunks(response):
    data = frames(response.text)
    assert data[-1] == "[DONE]"
    assert data.count("[DONE]") == 1
    return [json.loads(frame) for frame in data[:-1]]


def content_chunks(result):
    return [item for item in result if "muse_proxy" in item]


@pytest.mark.parametrize("include_usage", [False, True])
async def test_sse_progress_and_final_have_ordered_metadata_and_one_normal_finish(
    tmp_path, include_usage
):
    muse = EventMuse(
        [
            MuseMessage("progress", "資料を検索中🌏", activity_code="search"),
            MuseMessage("progress", "候補が見つかりました", message_id="status-2"),
            MuseMessage("reply", "最終回答\n日本語です", message_id="reply-1"),
        ]
    )
    request = payload(stream=True)
    request["stream_options"] = {"include_usage": include_usage}
    async with proxy(tmp_path, muse) as http:
        response = await http.post("/v1/chat/completions", json=request)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    result = chunks(response)
    opening = result[0]
    assert opening["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert "muse_proxy" not in opening
    messages = content_chunks(result)
    assert [item["choices"][0]["delta"]["content"] for item in messages] == [
        "資料を検索中🌏\n\n",
        "候補が見つかりました\n\n",
        "最終回答\n日本語です",
    ]
    generated_id = messages[0]["muse_proxy"]["message_id"]
    assert isinstance(generated_id, str) and generated_id.startswith("musemsg-")
    assert len({item["muse_proxy"]["message_id"] for item in messages}) == 3
    assert [item["muse_proxy"] for item in messages] == [
        {
            "kind": "progress",
            "request_id": opening["id"],
            "message_id": generated_id,
            "sequence": 1,
            "activity_code": "search",
        },
        {
            "kind": "progress",
            "request_id": opening["id"],
            "message_id": "status-2",
            "sequence": 2,
        },
        {
            "kind": "reply",
            "request_id": opening["id"],
            "message_id": "reply-1",
            "sequence": 3,
        },
    ]
    assert len({item["id"] for item in result}) == 1
    assert response.headers["x-request-id"] == opening["id"]
    assert [
        choice["finish_reason"]
        for item in result
        for choice in item.get("choices", [])
        if choice["finish_reason"]
    ] == ["stop"]
    if include_usage:
        assert result[-1]["choices"] == []
        assert result[-1]["usage"] is None
    assert muse.sent == [("調べてください🌏", "progress-chat")]
    assert muse.finished == ["調べてください🌏"]


async def test_rest_discards_progress_and_returns_only_final_reply(tmp_path):
    muse = EventMuse(
        [
            MuseMessage("progress", "この進捗を回答へ混ぜない"),
            MuseMessage("reply", "回答🌏", message_id="final"),
        ]
    )
    async with proxy(tmp_path, muse) as http:
        response = await http.post("/v1/chat/completions", json=payload())
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "回答🌏",
    }
    assert "この進捗を回答へ混ぜない" not in response.text
    assert muse.finished == ["調べてください🌏"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("replacement", [None, "newer-request"])
async def test_cancel_is_empty_success_closes_upstream_and_preserves_unknown_replacement(
    tmp_path, stream, replacement
):
    muse = EventMuse(
        [
            MuseMessage("progress", "調査中"),
            MuseMessage("cancel", message_id="cancel-1", superseded_by=replacement),
            MuseMessage("reply", "絶対に配信しない回答"),
        ]
    )
    async with proxy(tmp_path, muse) as http:
        response = await http.post("/v1/chat/completions", json=payload(stream=stream))
    assert response.status_code == 200
    assert "絶対に配信しない回答" not in response.text
    assert not muse.advanced_after_cancel
    assert muse.finished == ["調べてください🌏"]
    if stream:
        result = chunks(response)
        messages = content_chunks(result)
        assert [item["muse_proxy"]["kind"] for item in messages] == ["progress", "cancel"]
        cancel = messages[-1]
        metadata = cancel["muse_proxy"]
        assert cancel["choices"][0]["delta"]["content"] == ""
        assert metadata["message_id"] == "cancel-1"
        assert metadata["sequence"] == 2
        assert result[-1]["choices"][0]["finish_reason"] == "stop"
        assert all("error" not in item for item in result)
        completion_id = result[0]["id"]
    else:
        result = response.json()
        assert result["choices"][0]["message"] == {"role": "assistant", "content": ""}
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "調査中" not in response.text
        metadata = result["muse_proxy"]
        completion_id = result["id"]
    assert metadata["kind"] == "cancel"
    assert metadata["request_id"] == completion_id
    assert "superseded_by" in metadata
    assert metadata["superseded_by"] == replacement


@pytest.mark.parametrize("reply_before_error", [False, True])
async def test_failure_after_progress_never_claims_normal_completion(tmp_path, reply_before_error):
    messages = [MuseMessage("progress", "進捗🌏", activity_code="research")]
    if reply_before_error:
        messages.append(MuseMessage("reply", "確定前の応答"))
    muse = EventMuse(messages, failure=RuntimeError("private-token-and-upstream-host"))
    async with proxy(tmp_path, muse) as http:
        response = await http.post("/v1/chat/completions", json=payload(stream=True))
    result = chunks(response)
    assert content_chunks(result)[0]["muse_proxy"]["kind"] == "progress"
    assert result[-1]["error"]["code"] == "muse_unavailable"
    assert not any(
        choice["finish_reason"] == "stop" for item in result for choice in item.get("choices", [])
    )
    assert "private-token-and-upstream-host" not in response.text
    assert muse.sent == [("調べてください🌏", "progress-chat")]
    assert muse.finished == ["調べてください🌏"]


@pytest.mark.parametrize("stream", [False, True])
async def test_progress_without_reply_or_cancel_is_not_a_successful_answer(tmp_path, stream):
    muse = EventMuse([MuseMessage("progress", "まだ作業中")])
    async with proxy(tmp_path, muse) as http:
        response = await http.post("/v1/chat/completions", json=payload(stream=stream))
    if stream:
        result = chunks(response)
        assert "error" in result[-1]
        assert not any(
            choice["finish_reason"] == "stop"
            for item in result
            for choice in item.get("choices", [])
        )
    else:
        assert response.status_code == 502
        assert "error" in response.json()
    assert muse.finished == ["調べてください🌏"]


@pytest.mark.parametrize("stream", [False, True])
async def test_continuous_progress_does_not_extend_total_request_timeout(tmp_path, stream):
    class LongRunningMuse(EventMuse):
        async def stream_events(self, text, session_id=None):
            self.sent.append((text, session_id))
            try:
                for _ in range(100):
                    yield MuseMessage("progress", "調査を続けています")
                    await asyncio.sleep(0.01)
                yield MuseMessage("reply", "遅すぎる回答")
            finally:
                self.finished.append(text)

    muse = LongRunningMuse([])
    async with proxy(tmp_path, muse, request_timeout_seconds=0.05) as http:
        async with asyncio.timeout(1):
            response = await http.post("/v1/chat/completions", json=payload(stream=stream))
    if stream:
        result = chunks(response)
        assert result[-1]["error"]["code"] == "muse_reply_timeout"
        assert any(item.get("muse_proxy", {}).get("kind") == "progress" for item in result)
    else:
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "muse_reply_timeout"
    assert "遅すぎる回答" not in response.text
    assert muse.finished == ["調べてください🌏"]


async def test_sse_delivers_progress_before_muse_finishes_the_request(tmp_path):
    release = asyncio.Event()
    progress_delivered = asyncio.Event()

    class WaitingMuse(EventMuse):
        async def stream_events(self, text, session_id=None):
            self.sent.append((text, session_id))
            try:
                yield MuseMessage("progress", "途中の進捗🌏")
                await release.wait()
                yield MuseMessage("reply", "最後の回答")
            finally:
                self.finished.append(text)

    muse = WaitingMuse([])
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path), client=muse)
    body = json.dumps(payload(stream=True)).encode()
    scope = {
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
    supplied = False
    written = []

    async def receive():
        nonlocal supplied
        if not supplied:
            supplied = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        written.append(message.get("body", b""))
        if "途中の進捗🌏".encode() in message.get("body", b""):
            progress_delivered.set()

    async with app.router.lifespan_context(app):
        pending = asyncio.create_task(app(scope, receive, send))
        try:
            async with asyncio.timeout(1):
                await progress_delivered.wait()
            assert not pending.done()
            assert "最後の回答".encode() not in b"".join(written)
            assert b"[DONE]" not in b"".join(written)
        finally:
            release.set()
            async with asyncio.timeout(1):
                await pending
    assert "最後の回答".encode() in b"".join(written)
    assert b"[DONE]" in b"".join(written)
    assert muse.finished == ["調べてください🌏"]


async def test_openai_sdk_preserves_custom_progress_and_cancel_metadata(tmp_path):
    muse = EventMuse(
        [
            MuseMessage("progress", "調査中🌏", activity_code="search"),
            MuseMessage("cancel", message_id="cancel-1"),
        ]
    )
    async with proxy(tmp_path, muse) as http:
        async with AsyncOpenAI(
            api_key=KEY, base_url="http://proxy.test/v1", http_client=http, max_retries=0
        ) as client:
            stream = await client.chat.completions.create(
                model="muse",
                messages=[{"role": "user", "content": "調べてください🌏"}],
                extra_body={"session_id": "progress-chat"},
                stream=True,
            )
            async with stream:
                result = [item async for item in stream]
    structured = [item for item in result if "muse_proxy" in item.model_dump()]
    assert [item.model_dump()["muse_proxy"]["kind"] for item in structured] == [
        "progress",
        "cancel",
    ]
    assert structured[0].choices[0].delta.content == "調査中🌏\n\n"
    assert structured[0].model_dump()["muse_proxy"]["activity_code"] == "search"
    assert structured[1].choices[0].delta.content == ""
    assert structured[1].model_dump()["muse_proxy"]["superseded_by"] is None
    assert result[-1].choices[0].finish_reason == "stop"
    assert muse.finished == ["調べてください🌏"]


async def test_text_only_injected_client_remains_compatible_without_stream_events(tmp_path):
    class TextMuse:
        def __init__(self):
            self.finished = False
            self.closed = False

        async def stream_chat(self, text, session_id=None):
            try:
                yield "従来の"
                yield "テキスト応答🌏"
            finally:
                self.finished = True

        async def close(self):
            self.closed = True

    muse = TextMuse()
    async with proxy(tmp_path, muse) as http:
        response = await http.post("/v1/chat/completions", json=payload(stream=True))
    result = chunks(response)
    assert (
        "".join(
            choice["delta"].get("content", "")
            for item in result
            for choice in item.get("choices", [])
        )
        == "従来のテキスト応答🌏"
    )
    assert result[-1]["choices"][0]["finish_reason"] == "stop"
    assert muse.finished
