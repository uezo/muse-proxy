"""Exercise the official encrypted transport against an in-memory Muse peer."""

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
    NoiseFrameDecoder,
    NoiseXXResponder,
    ServiceFrame,
    encode_noise_frames,
)
from musegadget.noise.transport import decode_request_envelope, encode_response_envelope

from muse_proxy.errors import ProxyError
from muse_proxy.muse import MuseClient, _load_sdk

TEST_SIDE = "a2ac0148-138d-48ed-94eb-a631ab171112"


class MemoryWebSocket:
    def __init__(self):
        self.requests = asyncio.Queue()
        self.responses = asyncio.Queue()
        self.closed = False

    async def send(self, message):
        await self.requests.put(message)

    async def recv(self):
        message = await self.responses.get()
        if message is None:
            raise ConnectionError("The in-memory peer disconnected")
        return message

    async def close(self):
        self.closed = True
        await self.requests.put(None)


class MusePeer:
    """Protocol peer: real Noise handshake/envelopes, fixed text reply, no sockets."""

    def __init__(self, websocket, lose_ack):
        self.websocket = websocket
        self.lose_ack = lose_ack
        self.requests = []
        self.reset_streams = []
        self.decoder = NoiseFrameDecoder()
        self.finished = False

    async def read(self):
        while True:
            encrypted = await self.websocket.requests.get()
            if encrypted is None:
                return None
            plaintext = self.incoming.decrypt_with_ad(b"", encrypted)
            complete = self.decoder.decode(plaintext)
            if complete is not None:
                return decode_request_envelope(complete)

    async def send(self, frame):
        wire = encode_response_envelope(frame)
        for fragment in encode_noise_frames(wire):
            await self.websocket.responses.put(self.outgoing.encrypt_with_ad(b"", fragment))

    async def run(self):
        try:
            responder = NoiseXXResponder()
            responder.initialize()
            hello = await self.websocket.requests.get()
            response = responder.read_message1_and_write_message2(hello)
            await self.websocket.responses.put(response)
            responder.read_message3(await self.websocket.requests.get())
            self.outgoing, self.incoming = responder.split()

            control = await self.read()
            assert control.value.path == "/link-control"
            await self.send(ServiceFrame.response(control.stream_id, ApplicationResponse(200)))
            registration_frame = await self.read()
            registration = MessageDecoder().feed(registration_frame.value.data)[0]
            assert registration["method"] == "link.register"
            assert registration["params"]["commands_v2"] == {}
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

            chat = await self.read()
            self.requests.append(chat.value.path)
            assert chat.value.path == "/chat/stream"
            assert json.loads(chat.value.body) == {
                "message": "Hello, Muse",
                "device_id": "homelink-abcdef",
                "session_id": TEST_SIDE,
                "output_modality": "text",
            }
            events = [
                self.event("task.status", status="running", task_id="task-1"),
                self.event("delta.message_start", message_id="reply-1"),
                self.event(
                    "delta.text_append",
                    message_id="reply-1",
                    parent_message_id="reply-1",
                    text="Draft",
                ),
                self.event("delta.message_done", message_id="reply-1", content="こんにちは"),
                self.event("task.status", status="completed", task_id="task-1"),
            ]
            ndjson = b"".join(
                json.dumps(event, ensure_ascii=False).encode() + b"\n" for event in events
            )
            # Split inside a multibyte character and deliver events before the chat ACK.
            split = ndjson.index("こ".encode()) + 1
            for part in (ndjson[:split], ndjson[split:]):
                await self.send(ServiceFrame.body_chunk(subscribe.stream_id, BodyChunk(data=part)))
            if self.lose_ack:
                await self.websocket.responses.put(None)
            else:
                ack = {"message_id": "user-1", "session_id": TEST_SIDE, "is_thread": True}
                await self.send(
                    ServiceFrame.response(
                        chat.stream_id,
                        ApplicationResponse(200, body=json.dumps(ack).encode(), end_body=True),
                    )
                )
            while (frame := await self.read()) is not None:
                if frame.kind == "reset":
                    self.reset_streams.append(frame.stream_id)
            if not self.lose_ack:
                assert subscribe.stream_id in self.reset_streams
        finally:
            self.finished = True

    @staticmethod
    def event(name, **payload):
        return {
            "type": "event",
            "event": name,
            "payload": {
                "session_id": TEST_SIDE,
                **payload,
            },
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("lose_ack", [False, True], ids=["reply", "lost-ack-no-resubmit"])
async def test_real_encrypted_transport(tmp_path, monkeypatch, lose_ack):
    monkeypatch.delenv("MUSEGADGET_SDK_TOKEN", raising=False)
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
        request_timeout_seconds=2,
        connect_timeout_seconds=1,
    )
    websocket = MemoryWebSocket()
    peer = MusePeer(websocket, lose_ack)
    peer_task = None
    sessions = []
    connections = []

    async def connect(url, headers):
        nonlocal peer_task
        connections.append((url, headers))
        peer_task = asyncio.create_task(peer.run())
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
    try:
        if lose_ack:
            with pytest.raises(ProxyError):
                _ = [text async for text in client.stream_chat("Hello, Muse", TEST_SIDE)]
        else:
            assert [text async for text in client.stream_chat("Hello, Muse", TEST_SIDE)] == [
                "こんにちは"
            ]
        await asyncio.wait_for(peer_task, timeout=1)
    finally:
        await client.close()
        if peer_task is not None and not peer_task.done():
            peer_task.cancel()
            await asyncio.gather(peer_task, return_exceptions=True)
    assert connections == [
        ("wss://muse.invalid/v1/noise?vm_id=vm-1", {"Authorization": "Bearer test-vm-token"})
    ]
    assert peer.requests == ["/chat/subscribe", "/chat/stream"]
    assert peer.finished and websocket.closed
    assert not sessions[0]._requests
    assert not sessions[0]._streams
    assert not sessions[0].registered.is_set()
    assert not client._active
