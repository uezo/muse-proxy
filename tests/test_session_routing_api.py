"""HTTP-layer invariants for independently pending requests and multi-message replies.

The controlled backend emits routing decisions explicitly. Real SDK correlation,
acceptance ordering and session isolation are exercised in backend tests instead.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest

from muse_proxy.app import create_app
from muse_proxy.config import Settings
from muse_proxy.events import MuseMessage

KEY = "test-session-routing-key"
END = object()


@dataclass
class Pending:
    request_id: str
    session_id: str | None
    messages: asyncio.Queue = field(default_factory=asyncio.Queue)
    closed: asyncio.Event = field(default_factory=asyncio.Event)


class ControlledBackend:
    """Deliver explicit backend decisions without implementing Muse routing rules."""

    def __init__(self):
        self.requests = {}
        self.changed = asyncio.Condition()

    async def stream_request(self, text, session_id=None, *, request_id):
        assert request_id.startswith("chatcmpl-")
        pending = Pending(request_id, session_id)
        async with self.changed:
            self.requests[text] = pending
            self.changed.notify_all()
        try:
            while True:
                message = await pending.messages.get()
                if message is END:
                    return
                yield message
        finally:
            pending.closed.set()

    async def stream_events(self, text, session_id=None):
        raise AssertionError("Request-aware streaming must take precedence")
        yield  # pragma: no cover

    async def wait_for(self, text):
        async with asyncio.timeout(1):
            async with self.changed:
                await self.changed.wait_for(lambda: text in self.requests)
        return self.requests[text]


class Exchange:
    """Observe each ASGI body write before the complete HTTP response is buffered."""

    def __init__(self, app, text, *, stream=True, session_id="same-session"):
        self.response_headers = {}
        self.status = None
        self.written = bytearray()
        self.changed = asyncio.Condition()
        self.incoming = asyncio.Queue()
        self.incoming.put_nowait(
            {
                "type": "http.request",
                "body": json.dumps(
                    {
                        "model": "muse",
                        "session_id": session_id,
                        "messages": [{"role": "user", "content": text}],
                        "stream": stream,
                    }
                ).encode(),
                "more_body": False,
            }
        )
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
        self.task = asyncio.create_task(app(scope, self.incoming.get, self.send))

    async def send(self, message):
        async with self.changed:
            if message["type"] == "http.response.start":
                self.status = message["status"]
                self.response_headers = {
                    key.decode(): value.decode() for key, value in message["headers"]
                }
            elif message["type"] == "http.response.body":
                self.written.extend(message.get("body", b""))
            self.changed.notify_all()

    @property
    def body(self):
        return self.written.decode()

    async def wait_for_text(self, text):
        async with asyncio.timeout(1):
            async with self.changed:
                await self.changed.wait_for(lambda: text in self.body)

    async def finish(self):
        async with asyncio.timeout(1):
            await self.task

    async def disconnect(self):
        await self.incoming.put({"type": "http.disconnect"})
        await self.finish()

    async def cleanup(self):
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def frames(exchange):
    return [line[6:] for line in exchange.body.splitlines() if line.startswith("data: ")]


def chunks(exchange):
    return [json.loads(item) for item in frames(exchange) if item != "[DONE]"]


def metadata_chunks(exchange):
    return [item for item in chunks(exchange) if "muse_proxy" in item]


def assert_pending(exchange):
    assert not exchange.task.done()
    assert "[DONE]" not in exchange.body
    assert not any(
        choice["finish_reason"] for chunk in chunks(exchange) for choice in chunk.get("choices", [])
    )


def assert_finished_once(exchange):
    assert exchange.status == 200
    assert frames(exchange).count("[DONE]") == 1
    assert [
        choice["finish_reason"]
        for chunk in chunks(exchange)
        for choice in chunk.get("choices", [])
        if choice["finish_reason"]
    ] == ["stop"]


@asynccontextmanager
async def running(tmp_path):
    backend = ControlledBackend()
    app = create_app(Settings(api_key=KEY, state_dir=tmp_path), client=backend)
    exchanges = []

    def start(text, **options):
        exchange = Exchange(app, text, **options)
        exchanges.append(exchange)
        return exchange

    async with app.router.lifespan_context(app):
        try:
            yield backend, start
        finally:
            for exchange in exchanges:
                await exchange.cleanup()


async def test_sse_delivers_multiple_normal_replies_before_root_task_finishes(tmp_path):
    async with running(tmp_path) as (backend, start):
        exchange = start("札幌と博多を順に")
        pending = await backend.wait_for("札幌と博多を順に")
        await pending.messages.put(MuseMessage("reply", "札幌の回答🌏", message_id="sapporo"))
        await exchange.wait_for_text("札幌の回答🌏")
        assert_pending(exchange)
        assert not pending.closed.is_set()

        # Separators belong to the backend, not an HTTP-side progress conversion.
        await pending.messages.put(MuseMessage("reply", "\n\n博多の回答", message_id="hakata"))
        await exchange.wait_for_text("博多の回答")
        assert_pending(exchange)
        assert [item["muse_proxy"]["kind"] for item in metadata_chunks(exchange)] == [
            "reply",
            "reply",
        ]
        await pending.messages.put(END)
        await exchange.finish()

        assert_finished_once(exchange)
        messages = metadata_chunks(exchange)
        assert "".join(item["choices"][0]["delta"]["content"] for item in messages) == (
            "札幌の回答🌏\n\n博多の回答"
        )
        assert [item["muse_proxy"]["message_id"] for item in messages] == ["sapporo", "hakata"]
        assert [item["muse_proxy"]["sequence"] for item in messages] == [1, 2]
        assert {item["id"] for item in chunks(exchange)} == {pending.request_id}
        assert {item["muse_proxy"]["request_id"] for item in messages} == {pending.request_id}
        assert exchange.response_headers["x-request-id"] == pending.request_id
        assert pending.closed.is_set()


async def test_rest_accumulates_normal_replies_and_waits_for_whole_iterator(tmp_path):
    async with running(tmp_path) as (backend, start):
        exchange = start("二つの地域", stream=False)
        pending = await backend.wait_for("二つの地域")
        await pending.messages.put(MuseMessage("progress", "回答に混ぜない進捗"))
        await pending.messages.put(MuseMessage("reply", "札幌", message_id="one"))
        await pending.messages.put(MuseMessage("reply", "\n\n博多", message_id="two"))
        assert not exchange.task.done()
        await pending.messages.put(END)
        await exchange.finish()
        result = json.loads(exchange.body)
        assert exchange.status == 200
        assert result["id"] == pending.request_id
        assert result["choices"][0]["message"] == {"role": "assistant", "content": "札幌\n\n博多"}
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "回答に混ぜない進捗" not in exchange.body


@pytest.mark.parametrize("old_stream", [False, True])
async def test_cancelled_old_http_request_finishes_while_replacement_keeps_streaming(
    tmp_path, old_stream
):
    async with running(tmp_path) as (backend, start):
        old = start("札幌を調べて", stream=old_stream)
        first = await backend.wait_for("札幌を調べて")
        await first.messages.put(MuseMessage("reply", "古い要求の途中回答", message_id="old-part"))
        if old_stream:
            await old.wait_for_text("古い要求の途中回答")
            assert_pending(old)
        latest = start("味噌専門店のみで")
        second = await backend.wait_for("味噌専門店のみで")
        assert first.session_id == second.session_id == "same-session"
        assert first.request_id != second.request_id

        # The real backend chooses the routing. HTTP must retain its decision,
        # including the exact replacement completion ID supplied to the backend.
        await first.messages.put(
            MuseMessage("cancel", message_id="supersession", superseded_by=second.request_id)
        )
        await old.finish()
        assert first.closed.is_set()
        assert not second.closed.is_set()
        assert_pending(latest)
        assert old.response_headers["x-request-id"] == first.request_id
        if old_stream:
            assert_finished_once(old)
            assert [item["muse_proxy"]["kind"] for item in metadata_chunks(old)] == [
                "reply",
                "cancel",
            ]
            cancel = metadata_chunks(old)[-1]
            assert cancel["choices"][0]["delta"]["content"] == ""
            metadata = cancel["muse_proxy"]
        else:
            result = json.loads(old.body)
            assert old.status == 200
            assert result["choices"][0]["message"]["content"] == ""
            metadata = result["muse_proxy"]
        assert metadata["kind"] == "cancel"
        assert metadata["request_id"] == first.request_id
        assert metadata["superseded_by"] == second.request_id

        await second.messages.put(MuseMessage("reply", "味噌専門店の回答", message_id="latest"))
        await latest.wait_for_text("味噌専門店の回答")
        assert_pending(latest)
        await second.messages.put(END)
        await latest.finish()
        assert_finished_once(latest)
        assert latest.response_headers["x-request-id"] == metadata["superseded_by"]
        assert {item["id"] for item in chunks(latest)} == {second.request_id}
        assert second.closed.is_set()


async def test_old_http_disconnect_closes_only_its_own_request_iterator(tmp_path):
    async with running(tmp_path) as (backend, start):
        old = start("先の要求")
        first = await backend.wait_for("先の要求")
        latest = start("後の要求")
        second = await backend.wait_for("後の要求")
        await old.disconnect()
        assert first.closed.is_set()
        assert not second.closed.is_set()
        assert_pending(latest)

        await second.messages.put(MuseMessage("reply", "残った要求の回答", message_id="survives"))
        await latest.wait_for_text("残った要求の回答")
        assert_pending(latest)
        await second.messages.put(END)
        await latest.finish()
        assert_finished_once(latest)
