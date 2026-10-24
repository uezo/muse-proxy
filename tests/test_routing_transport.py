"""Concurrent request routing through the real encrypted Muse SDK transport."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from musegadget.link_client import LinkSession, MessageDecoder, encode_message
from musegadget.noise import (
    ApplicationResponse,
    BodyChunk,
    Header,
    NoiseXXResponder,
    ServiceFrame,
)
from test_transport import TEST_SIDE, MemoryWebSocket, MusePeer

from muse_proxy.muse import MuseClient, _load_sdk


class RoutingPeer(MusePeer):
    """Two accepted requests, two replies, one root task, over actual Noise frames."""

    def __init__(self, websocket, suffix, terminal_before_last_reply=False):
        super().__init__(websocket, lose_ack=False)
        self.suffix = suffix
        self.terminal_before_last_reply = terminal_before_last_reply
        self.first_accepted = asyncio.Event()
        self.release_second = asyncio.Event()

    async def emit(self, subscribe, *events):
        data = b"".join(json.dumps(event, ensure_ascii=False).encode() + b"\n" for event in events)
        await self.send(ServiceFrame.body_chunk(subscribe.stream_id, BodyChunk(data=data)))

    async def acknowledge(self, chat, message_id):
        ack = {"message_id": message_id, "session_id": TEST_SIDE, "is_thread": True}
        await self.send(
            ServiceFrame.response(
                chat.stream_id,
                ApplicationResponse(200, body=json.dumps(ack).encode(), end_body=True),
            )
        )

    async def run(self):
        try:
            responder = NoiseXXResponder()
            responder.initialize()
            hello = await self.websocket.requests.get()
            await self.websocket.responses.put(responder.read_message1_and_write_message2(hello))
            responder.read_message3(await self.websocket.requests.get())
            self.outgoing, self.incoming = responder.split()

            control = await self.read()
            assert control.value.path == "/link-control"
            await self.send(ServiceFrame.response(control.stream_id, ApplicationResponse(200)))
            registration_frame = await self.read()
            registration = MessageDecoder().feed(registration_frame.value.data)[0]
            assert registration["method"] == "link.register"
            assert registration["params"]["node_id"] == "homelink-abcdef"
            await self.send(
                ServiceFrame.body_chunk(
                    control.stream_id,
                    BodyChunk(data=encode_message({"id": registration["id"], "ok": True})),
                )
            )

            subscribe = await self.read()
            self.requests.append(subscribe.value.path)
            assert subscribe.value.path == "/chat/subscribe"
            assert json.loads(subscribe.value.body) == {"session_id": TEST_SIDE}
            await self.send(
                ServiceFrame.response(
                    subscribe.stream_id,
                    ApplicationResponse(
                        200, headers=[Header("Content-Type", "application/x-ndjson")]
                    ),
                )
            )

            # Subscription snapshots may report a prior idle/completed agent before
            # this batch's own work starts, using the same stable task identifier.
            await self.emit(
                subscribe,
                self.event("task.status", status="completed", task_id="task-root"),
            )
            first = await self.read()
            self.requests.append(first.value.path)
            assert first.value.path == "/chat/stream"
            assert json.loads(first.value.body)["message"] == "札幌のラーメン屋さんを調べて"
            await self.acknowledge(first, f"user-sapporo-{self.suffix}")
            await self.emit(
                subscribe,
                self.event("task.status", status="running", task_id="task-root"),
            )
            self.first_accepted.set()

            second = await self.read()
            self.requests.append(second.value.path)
            assert second.value.path == "/chat/stream"
            assert json.loads(second.value.body)["message"] == "博多も調べて。札幌を先に回答して"
            await self.acknowledge(second, f"user-hakata-{self.suffix}")
            await self.emit(
                subscribe,
                self.event("task.status", status="running", task_id="task-child"),
                self.event("delta.message_start", message_id=f"reply-sapporo-{self.suffix}"),
                self.event(
                    "delta.text_append",
                    message_id=f"reply-sapporo-{self.suffix}",
                    parent_message_id=f"reply-sapporo-{self.suffix}",
                    text="札幌の回答",
                ),
                self.event(
                    "delta.message_done",
                    message_id=f"reply-sapporo-{self.suffix}",
                    content="札幌の回答",
                ),
                self.event("task.status", status="completed", task_id="task-child"),
            )
            await self.release_second.wait()
            second_reply = [
                self.event("delta.message_start", message_id=f"reply-hakata-{self.suffix}"),
                self.event(
                    "delta.message_done",
                    message_id=f"reply-hakata-{self.suffix}",
                    content="博多の回答",
                ),
            ]
            terminal = self.event("task.status", status="completed", task_id="task-root")
            events = (
                [terminal, *second_reply]
                if self.terminal_before_last_reply
                else [*second_reply, terminal]
            )
            await self.emit(subscribe, *events)
            while (frame := await self.read()) is not None:
                if frame.kind == "reset":
                    self.reset_streams.append(frame.stream_id)
            assert subscribe.stream_id in self.reset_streams
        finally:
            self.finished = True


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_before_last_reply", [False, True])
async def test_real_noise_routes_latest_request_and_keeps_multiple_replies_open(
    tmp_path, monkeypatch, terminal_before_last_reply
):
    monkeypatch.delenv("MUSEGADGET_SDK_TOKEN", raising=False)
    if terminal_before_last_reply:
        # Model a scheduler resuming after the grace period while reordered
        # final-message events are already queued by the subscription reader.
        monkeypatch.setattr("muse_proxy._routing._TERMINAL_GRACE", 0)
    (tmp_path / "identity.json").write_text(json.dumps({"mac": "02:00:00:ab:cd:ef"}))
    (tmp_path / "pairing.json").write_text(
        json.dumps(
            {
                "access_token": "test-device-token",
                "refresh_token": "test-refresh-token",
                "access_token_saved_at": time.time(),
                "noise_host": "muse.invalid",
            }
        )
    )
    settings = SimpleNamespace(
        state_dir=tmp_path,
        sdk_token=None,
        vm_id=None,
        request_timeout_seconds=3,
        connect_timeout_seconds=1,
    )
    peer_tasks = []
    peers = []
    sessions = []
    connected = asyncio.Queue()

    async def connect(url, headers):
        assert url == "wss://muse.invalid/v1/noise?vm_id=vm-1"
        assert headers == {"Authorization": "Bearer test-vm-token"}
        websocket = MemoryWebSocket()
        peer = RoutingPeer(
            websocket,
            suffix=str(len(peers)),
            terminal_before_last_reply=terminal_before_last_reply,
        )
        peers.append(peer)
        connected.put_nowait(peer)
        peer_tasks.append(asyncio.create_task(peer.run()))
        return websocket

    def session_factory(**kwargs):
        session = LinkSession(**kwargs, connect=connect)
        sessions.append(session)
        return session

    sdk = _load_sdk()
    sdk.LinkSession = session_factory
    sdk.api = SimpleNamespace(
        fetch_vms_with_status=lambda *_: (
            [{"vm_id": "vm-1", "vm_auth_token": "test-vm-token", "is_default": True}],
            200,
        )
    )
    monkeypatch.setattr("muse_proxy.muse._load_sdk", lambda: sdk)
    client = MuseClient(settings)
    tasks = []

    async def collect(text, request_id, first_reply=None, record=None):
        messages = []
        async for message in client.stream_request(text, TEST_SIDE, request_id=request_id):
            messages.append(message)
            if record is not None:
                record.append(message)
                if message.kind == "reply":
                    first_reply.set()
        return messages

    try:
        # Reconnect to the same chat for another batch. Muse reuses task-root as
        # an agent ID, while user and assistant message IDs remain unique.
        for batch in range(2):
            received = []
            first_reply = asyncio.Event()
            request_a = f"request-A-{batch}"
            request_b = f"request-B-{batch}"
            first = asyncio.create_task(collect("札幌のラーメン屋さんを調べて", request_a))
            tasks.append(first)
            peer = await asyncio.wait_for(connected.get(), timeout=1)
            await asyncio.wait_for(peer.first_accepted.wait(), timeout=1)
            second = asyncio.create_task(
                collect("博多も調べて。札幌を先に回答して", request_b, first_reply, received)
            )
            tasks.append(second)
            await asyncio.wait_for(first_reply.wait(), timeout=1)
            older = await asyncio.wait_for(asyncio.shield(first), timeout=1)
            assert len(older) == 1
            assert older[0].kind == "cancel"
            assert older[0].superseded_by == request_b
            assert received[0].kind == "reply"
            assert received[0].text == "札幌の回答"
            assert not second.done(), "A message end or a child task end cannot close the stream"
            peer.release_second.set()
            newer = await asyncio.wait_for(asyncio.shield(second), timeout=1)
            assert [message.kind for message in newer] == ["reply", "reply"]
            assert [message.message_id for message in newer] == [
                f"reply-sapporo-{batch}",
                f"reply-hakata-{batch}",
            ]
            assert "".join(message.text for message in newer) == "札幌の回答\n\n博多の回答"
            await asyncio.wait_for(asyncio.shield(peer_tasks[-1]), timeout=1)
    finally:
        for peer in peers:
            peer.release_second.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()
        if peer_tasks:
            await asyncio.wait_for(asyncio.gather(*peer_tasks), timeout=1)
    assert len(sessions) == 2
    for peer, session in zip(peers, sessions, strict=True):
        assert peer.requests == ["/chat/subscribe", "/chat/stream", "/chat/stream"]
        assert peer.finished and peer.websocket.closed
        assert not session._requests
        assert not session._streams
