"""Observe the real OpenAI Python SDK when the proxy emits its cancel extension."""

import json
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk

from muse_proxy.app import create_app
from muse_proxy.config import Settings
from muse_proxy.events import MuseMessage

KEY = "test-sdk-cancel-proxy-key"
OLD_ID = "chatcmpl-cancel-mock"
NEW_ID = "chatcmpl-replacement-mock"
PARTIAL = "途中まで受信済み"


class CancelMuse:
    """Only Muse is mocked; the proxy and SDK process the actual HTTP/SSE wire."""

    def __init__(self, prefix=None):
        self.prefix = prefix
        self.sent = []
        self.closed = False
        self.advanced_after_cancel = False

    async def stream_request(self, text, session_id=None, *, request_id):
        self.sent.append((text, session_id, request_id))
        try:
            if self.prefix is not None:
                yield self.prefix
            yield MuseMessage("cancel", message_id="cancel-old", superseded_by=NEW_ID)
            self.advanced_after_cancel = True
            raise AssertionError("The HTTP boundary must stop after cancellation")
        finally:
            self.closed = True


@asynccontextmanager
async def proxy_sdk(tmp_path, muse):
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path), client=muse)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
            async with AsyncOpenAI(
                api_key=KEY,
                base_url="http://proxy.test/v1",
                http_client=http,
                max_retries=3,
            ) as client:
                yield client


@pytest.mark.parametrize(
    "prefix",
    [None, MuseMessage("progress", "検索中"), MuseMessage("reply", PARTIAL)],
    ids=["empty", "after-progress", "after-partial-reply"],
)
async def test_async_sdk_cancel_is_a_normal_chunk_not_an_exception_or_retry(tmp_path, prefix):
    muse = CancelMuse(prefix)
    async with proxy_sdk(tmp_path, muse) as client:
        stream = await client.chat.completions.create(
            model="muse",
            messages=[{"role": "user", "content": "前の要求"}],
            extra_body={"session_id": "same-session"},
            stream=True,
        )
        async with stream:
            chunks = [chunk async for chunk in stream]

    # Successful iteration means neither an API exception nor CancelledError was raised.
    cancel = next(
        chunk
        for chunk in chunks
        if chunk.model_dump().get("muse_proxy", {}).get("kind") == "cancel"
    )
    metadata = cancel.model_dump()["muse_proxy"]
    assert cancel.model_extra["muse_proxy"] == metadata
    assert metadata["superseded_by"] == NEW_ID
    assert metadata["request_id"] == cancel.id == muse.sent[0][2]
    assert cancel.choices[0].delta.content == ""
    assert cancel.choices[0].finish_reason is None
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert len(chunks) == 3 + (prefix is not None), "[DONE] is consumed by the SDK"
    assert {chunk.id for chunk in chunks} == {cancel.id}
    assembled = "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
    expected = "" if prefix is None else prefix.text + ("\n\n" if prefix.kind == "progress" else "")
    assert assembled == expected, "The SDK does not clear previously received text on cancel"
    assert len(muse.sent) == 1, "Even max_retries=3 must not replay this successful HTTP request"
    assert muse.sent[0][:2] == ("前の要求", "same-session")
    assert muse.closed and not muse.advanced_after_cancel


async def test_async_sdk_rest_cancel_is_an_empty_success_with_extra_metadata(tmp_path):
    muse = CancelMuse(MuseMessage("reply", PARTIAL))
    async with proxy_sdk(tmp_path, muse) as client:
        completion = await client.chat.completions.create(
            model="muse", messages=[{"role": "user", "content": "前の要求"}]
        )
    assert completion.choices[0].message.content == ""
    assert completion.choices[0].finish_reason == "stop"
    assert completion.model_extra["muse_proxy"]["kind"] == "cancel"
    assert completion.model_dump()["muse_proxy"]["superseded_by"] == NEW_ID
    assert completion.model_extra["muse_proxy"]["request_id"] == completion.id
    assert len(muse.sent) == 1
    assert muse.closed and not muse.advanced_after_cancel


def mock_stream_transport(prefix):
    """Canned SSE for sync SDK and its high-level streaming accumulator."""
    identity = {"id": OLD_ID, "object": "chat.completion.chunk", "created": 0, "model": "muse"}

    def chunk(delta, *, finish=None, metadata=None):
        item = {
            **identity,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
        }
        if metadata is not None:
            item["muse_proxy"] = metadata
        return item

    chunks = [chunk({"role": "assistant", "content": ""})]
    if prefix:
        chunks.append(chunk({"content": prefix}))
    chunks.extend(
        [
            chunk(
                {"content": ""},
                metadata={
                    "kind": "cancel",
                    "request_id": OLD_ID,
                    "message_id": "cancel-old",
                    "sequence": 1,
                    "superseded_by": NEW_ID,
                },
            ),
            chunk({}, finish="stop"),
        ]
    )
    wire = "".join("data: " + json.dumps(item) + "\n\n" for item in chunks) + "data: [DONE]\n\n"
    calls = []

    def handle(request):
        calls.append(request)
        assert request.url == "http://proxy.test/v1/chat/completions"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-request-id": OLD_ID},
            content=wire.encode(),
        )

    return httpx.MockTransport(handle), calls


@pytest.mark.parametrize("prefix", ["", PARTIAL], ids=["empty", "after-partial-reply"])
def test_sync_sdk_cancel_preserves_metadata_and_finishes_normally(prefix):
    transport, calls = mock_stream_transport(prefix)
    with httpx.Client(transport=transport) as http:
        with OpenAI(
            api_key=KEY, base_url="http://proxy.test/v1", http_client=http, max_retries=3
        ) as client:
            with client.chat.completions.create(
                model="muse", messages=[{"role": "user", "content": "前の要求"}], stream=True
            ) as stream:
                chunks = list(stream)
    cancel = next(chunk for chunk in chunks if chunk.model_extra.get("muse_proxy"))
    assert cancel.model_extra["muse_proxy"]["kind"] == "cancel"
    assert cancel.model_dump()["muse_proxy"]["superseded_by"] == NEW_ID
    assert cancel.choices[0].delta.content == ""
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert len(chunks) == 3 + bool(prefix)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == prefix
    assert len(calls) == 1


def assert_high_level_cancel(events, completion):
    raw = [event.chunk for event in events if event.type == "chunk"]
    cancel = next(chunk for chunk in raw if chunk.model_extra.get("muse_proxy"))
    assert cancel.model_extra["muse_proxy"]["kind"] == "cancel"
    assert cancel.model_extra["muse_proxy"]["superseded_by"] == NEW_ID
    assert [event.content for event in events if event.type == "content.done"] == [PARTIAL]
    assert completion.choices[0].message.content == PARTIAL
    assert completion.choices[0].finish_reason == "stop"
    assert "muse_proxy" not in completion.model_extra


def test_sync_stream_helper_requires_raw_chunk_to_detect_cancel():
    transport, calls = mock_stream_transport(PARTIAL)
    with httpx.Client(transport=transport) as http:
        with OpenAI(
            api_key=KEY, base_url="http://proxy.test/v1", http_client=http, max_retries=3
        ) as client:
            with client.chat.completions.stream(
                model="muse", messages=[{"role": "user", "content": "前の要求"}]
            ) as stream:
                events = list(stream)
                completion = stream.get_final_completion()
    assert_high_level_cancel(events, completion)
    assert len(calls) == 1


async def test_async_stream_helper_requires_raw_chunk_to_detect_cancel():
    transport, calls = mock_stream_transport(PARTIAL)
    async with httpx.AsyncClient(transport=transport) as http:
        async with AsyncOpenAI(
            api_key=KEY, base_url="http://proxy.test/v1", http_client=http, max_retries=3
        ) as client:
            async with client.chat.completions.stream(
                model="muse", messages=[{"role": "user", "content": "前の要求"}]
            ) as stream:
                events = [event async for event in stream]
                completion = await stream.get_final_completion()
    assert_high_level_cancel(events, completion)
    assert len(calls) == 1
