"""Diagnose upstream failures using the real SDK and an in-memory Noise peer."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from musegadget.link_client import LinkSession, MessageDecoder, encode_message
from musegadget.noise import ApplicationResponse, BodyChunk, NoiseXXResponder, ServiceFrame
from test_transport import TEST_SIDE, MemoryWebSocket, MusePeer

from muse_proxy.errors import ProxyError
from muse_proxy.muse import MuseClient, _load_sdk

REQUEST_TEXT = "こんにちは。上流で拒否された理由を確認したい"
REQUEST_ID = "chatcmpl-upstream-refusal"
ACCESS_TOKEN = "test-upstream-access-token"
REFRESH_SUFFIX = "test-upstream-refresh-suffix"
REFRESH_TOKEN = "hatch_refresh:" + REFRESH_SUFFIX
VM_TOKEN = "test-upstream-vm-token"
SDK_TOKEN = "mgst_" + "a" * 42 + "A"
ROTATED_ACCESS = "test-upstream-rotated-access"
ROTATED_REFRESH_SUFFIX = "test-upstream-rotated-refresh"
ROTATED_REFRESH = "hatch_refresh:" + ROTATED_REFRESH_SUFFIX
REFUSAL = "This device cannot open the requested chat"
UNRELATED_TEXT = "original unrelated event remains observable"


class DiagnosticPeer(MusePeer):
    """Reply with an actual application status/body inside encrypted frames."""

    def __init__(self, websocket, *, reject, missing_chat, rotate):
        super().__init__(websocket, lose_ack=False)
        self.reject = reject
        self.missing_chat = missing_chat
        self.rotate = rotate
        self.chat_bodies = []
        self.upstream_ids = []

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
            await self.send(
                ServiceFrame.body_chunk(
                    control.stream_id,
                    BodyChunk(data=encode_message({"id": registration["id"], "ok": True})),
                )
            )

            subscribed = None
            while (frame := await self.read()) is not None:
                if frame.kind == "reset":
                    self.reset_streams.append(frame.stream_id)
                    continue
                path = frame.value.path
                self.requests.append(path)
                if path == "/chat/subscribe":
                    assert json.loads(frame.value.body) == {"session_id": TEST_SIDE}
                    if self.missing_chat:
                        await self.send(
                            ServiceFrame.response(
                                frame.stream_id,
                                ApplicationResponse(
                                    404,
                                    body=b'{"error":"The requested side chat does not exist"}',
                                    end_body=True,
                                ),
                            )
                        )
                    else:
                        subscribed = frame.stream_id
                        await self.send(
                            ServiceFrame.response(frame.stream_id, ApplicationResponse(200))
                        )
                elif path == "/chat/stream":
                    self.chat_bodies.append(json.loads(frame.value.body))
                    self.upstream_ids.append(
                        next(
                            header.value
                            for header in frame.value.headers
                            if header.key.lower() == "x-request-id"
                        )
                    )
                    if self.reject:
                        error = {
                            "error": {
                                "code": "device_chat_unavailable",
                                "message": REFUSAL,
                                "access_token": ACCESS_TOKEN,
                                "detail": f"Bearer {VM_TOKEN}; refresh={REFRESH_SUFFIX}",
                            }
                        }
                        if self.rotate:
                            error["error"]["detail"] += (
                                f"; access={ROTATED_ACCESS}; refresh={ROTATED_REFRESH_SUFFIX}"
                            )
                        await self.send(
                            ServiceFrame.response(
                                frame.stream_id,
                                ApplicationResponse(
                                    422, body=json.dumps(error).encode(), end_body=True
                                ),
                            )
                        )
                    else:
                        assert subscribed is not None
                        unrelated = self.event(
                            "delta.message_done", message_id="unrelated", content=UNRELATED_TEXT
                        )
                        unrelated["payload"]["session_id"] = "another-chat"
                        events = [
                            unrelated,
                            self.event("task.status", status="running", task_id="root"),
                            self.event("delta.message_start", message_id="reply-1"),
                            self.event(
                                "delta.message_done", message_id="reply-1", content="こんにちは"
                            ),
                            self.event("task.status", status="completed", task_id="root"),
                        ]
                        ndjson = b"".join(
                            json.dumps(event, ensure_ascii=False).encode() + b"\n"
                            for event in events
                        )
                        split = ndjson.index("こ".encode()) + 1
                        for chunk in (ndjson[:split], ndjson[split:]):
                            await self.send(
                                ServiceFrame.body_chunk(subscribed, BodyChunk(data=chunk))
                            )
                        ack = {
                            "message_id": "user-1",
                            "session_id": TEST_SIDE,
                            "is_thread": True,
                        }
                        await self.send(
                            ServiceFrame.response(
                                frame.stream_id,
                                ApplicationResponse(
                                    200, body=json.dumps(ack).encode(), end_body=True
                                ),
                            )
                        )
                else:
                    raise AssertionError(f"Unexpected application request {path}")
        finally:
            self.finished = True


@asynccontextmanager
async def diagnostic_client(
    tmp_path, monkeypatch, *, reject=True, missing_chat=False, rotate=False
):
    monkeypatch.delenv("MUSEGADGET_SDK_TOKEN", raising=False)
    (tmp_path / "identity.json").write_text(json.dumps({"mac": "02:00:00:ab:cd:ef"}))
    (tmp_path / "pairing.json").write_text(
        json.dumps(
            {
                "access_token": ACCESS_TOKEN,
                "refresh_token": REFRESH_TOKEN,
                "access_token_saved_at": time.time(),
                "noise_host": "muse.invalid",
            }
        )
    )
    settings = SimpleNamespace(
        state_dir=tmp_path,
        sdk_token=SDK_TOKEN,
        vm_id=None,
        request_timeout_seconds=2,
        connect_timeout_seconds=1,
    )
    websocket = MemoryWebSocket()
    peer = DiagnosticPeer(websocket, reject=reject, missing_chat=missing_chat, rotate=rotate)
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
    fetch_calls = []
    refresh_calls = []

    def fetch(token, root):
        fetch_calls.append((token, root))
        if rotate and len(fetch_calls) == 1:
            return [], 401
        return [{"vm_id": "vm-1", "vm_auth_token": VM_TOKEN, "is_default": True}], 200

    def refresh(*args):
        refresh_calls.append(args)
        return {"access_token": ROTATED_ACCESS, "refresh_token": ROTATED_REFRESH}, 200

    sdk.api = SimpleNamespace(fetch_vms_with_status=fetch, refresh_device_token=refresh)
    monkeypatch.setattr("muse_proxy.muse._load_sdk", lambda: sdk)
    client = MuseClient(settings)
    try:
        yield client, peer
        await asyncio.wait_for(peer_task, timeout=1)
    finally:
        await client.close()
        if peer_task is not None and not peer_task.done():
            peer_task.cancel()
            await asyncio.gather(peer_task, return_exceptions=True)
    assert peer.finished and websocket.closed
    assert len(connections) == 1
    assert not sessions[0]._requests
    assert not sessions[0]._streams
    assert not sessions[0].registered.is_set()
    assert not client._active
    assert len(fetch_calls) == (2 if rotate else 1)
    assert len(refresh_calls) == (1 if rotate else 0)


def upstream_messages(caplog):
    return [
        record.getMessage() for record in caplog.records if record.name == "muse_proxy.upstream"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_chat", [False, True], ids=["existing-chat", "new-chat"])
async def test_trace_records_real_noise_chat_refusal_and_original_status(
    tmp_path, monkeypatch, caplog, missing_chat
):
    caplog.set_level(5, logger="muse_proxy")
    async with diagnostic_client(tmp_path, monkeypatch, missing_chat=missing_chat) as (
        client,
        peer,
    ):
        with pytest.raises(ProxyError) as rejected:
            _ = [
                part
                async for part in client.stream_request(
                    REQUEST_TEXT, TEST_SIDE, request_id=REQUEST_ID
                )
            ]
        assert rejected.value.status_code == 502
        assert rejected.value.code == "muse_request_error"
        assert str(rejected.value) == "Muse refused the chat request."
        assert peer.chat_bodies == [
            {
                "message": REQUEST_TEXT,
                "output_modality": "text",
                "device_id": "homelink-abcdef",
                "session_id": TEST_SIDE,
            }
        ]
        upstream_id = peer.upstream_ids[0]
    messages = upstream_messages(caplog)
    text = "\n".join(messages)
    assert REQUEST_TEXT in text
    assert REFUSAL in text
    assert "device_chat_unavailable" in text
    assert "status=422" in text
    assert "/chat/stream" in text
    assert f"request_id={REQUEST_ID}" in text
    assert upstream_id in text
    assert "stream_id=" in text
    assert "router_id=router-" in text
    assert "link.register" in text
    assert "muse.control.request" in text
    assert "muse.control.response" in text
    wire_requests = [
        message
        for message in messages
        if message.startswith("muse.request.body ") and "path=/chat/stream " in message
    ]
    assert len(wire_requests) == 1
    assert json.loads(wire_requests[0].split(" body=", 1)[1]) == peer.chat_bodies[0]
    wire_responses = [
        message
        for message in messages
        if message.startswith("muse.response ") and "path=/chat/stream " in message
    ]
    assert len(wire_responses) == 1
    assert f"request_id={REQUEST_ID} " in wire_responses[0]
    assert f"upstream_request_id={upstream_id} " in wire_responses[0]
    assert "status=422 " in wire_responses[0]
    if missing_chat:
        assert "status=404" in text
        assert "The requested side chat does not exist" in text
    for secret in (ACCESS_TOKEN, REFRESH_TOKEN, REFRESH_SUFFIX, VM_TOKEN, SDK_TOKEN):
        assert secret not in text


@pytest.mark.asyncio
async def test_debug_does_not_log_upstream_chat_or_refusal_contents(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="muse_proxy")
    async with diagnostic_client(tmp_path, monkeypatch) as (client, peer):
        with pytest.raises(ProxyError):
            _ = [
                part
                async for part in client.stream_request(
                    REQUEST_TEXT, TEST_SIDE, request_id=REQUEST_ID
                )
            ]
        assert len(peer.chat_bodies) == 1
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert REQUEST_TEXT not in text
    assert REFUSAL not in text
    for secret in (ACCESS_TOKEN, REFRESH_TOKEN, REFRESH_SUFFIX, VM_TOKEN, SDK_TOKEN):
        assert secret not in text


@pytest.mark.asyncio
async def test_trace_observes_original_events_before_session_filtering(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(5, logger="muse_proxy")
    async with diagnostic_client(tmp_path, monkeypatch, reject=False) as (client, peer):
        received = [
            part
            async for part in client.stream_request(REQUEST_TEXT, TEST_SIDE, request_id=REQUEST_ID)
        ]
        assert [part.text for part in received if part.kind == "reply"] == ["こんにちは"]
        assert len(peer.chat_bodies) == 1
    text = "\n".join(upstream_messages(caplog))
    assert UNRELATED_TEXT in text
    assert "another-chat" in text
    assert "delta.message_done" in text
    assert "こんにちは" in text
    assert "status=200" in text


@pytest.mark.asyncio
async def test_trace_vm_401_refresh_logs_status_and_redacts_rotated_credentials(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(5, logger="muse_proxy")
    async with diagnostic_client(tmp_path, monkeypatch, rotate=True) as (client, peer):
        with pytest.raises(ProxyError) as rejected:
            _ = [
                part
                async for part in client.stream_request(
                    REQUEST_TEXT, TEST_SIDE, request_id=REQUEST_ID
                )
            ]
        assert rejected.value.code == "muse_request_error"
        assert len(peer.chat_bodies) == 1
    text = "\n".join(upstream_messages(caplog))
    assert "status=401" in text
    assert "status=200" in text
    assert "status=422" in text
    assert "/fetch_vms" in text
    assert "/device_token/refresh" in text
    assert "sdk_result" in text
    assert REFUSAL in text
    for secret in (
        ACCESS_TOKEN,
        REFRESH_TOKEN,
        REFRESH_SUFFIX,
        VM_TOKEN,
        SDK_TOKEN,
        ROTATED_ACCESS,
        ROTATED_REFRESH,
        ROTATED_REFRESH_SUFFIX,
    ):
        assert secret not in text


def test_trace_failure_keeps_exception_reason_and_redacts_known_runtime_secrets(caplog):
    from muse_proxy._upstream import MuseTrace, trace_context

    caplog.set_level(5, logger="muse_proxy")
    trace = MuseTrace("router-failure", secrets=(VM_TOKEN,))
    trace.credentials(
        SimpleNamespace(
            pairing={"access_token": ACCESS_TOKEN, "refresh_token": REFRESH_TOKEN},
            sdk_token=SDK_TOKEN,
        )
    )
    trace.add_secrets(ROTATED_ACCESS, ROTATED_REFRESH)
    failure = ConnectionError(
        f"server rejected scope; Authorization: Bearer {VM_TOKEN}; access={ACCESS_TOKEN}; "
        f"refresh={REFRESH_SUFFIX}; sdk={SDK_TOKEN}; rotated={ROTATED_ACCESS}"
    )
    failure.status = 403
    with trace_context(trace, request_id=REQUEST_ID):
        trace.failure("connect", failure)
    text = "\n".join(upstream_messages(caplog))
    assert "server rejected scope" in text
    assert "ConnectionError" in text
    assert "status=403" in text
    assert f"request_id={REQUEST_ID}" in text
    assert "router_id=router-failure" in text
    for secret in (
        VM_TOKEN,
        ACCESS_TOKEN,
        REFRESH_TOKEN,
        REFRESH_SUFFIX,
        SDK_TOKEN,
        ROTATED_ACCESS,
    ):
        assert secret not in text


@pytest.mark.asyncio
async def test_trace_request_context_isolated_between_concurrent_tasks(caplog):
    from muse_proxy._upstream import MuseTrace, trace_context

    caplog.set_level(5, logger="muse_proxy")
    trace = MuseTrace("router-shared")

    async def emit(request_id):
        with trace_context(trace, request_id=request_id):
            trace.emit("muse.test", operation=f"{request_id}-start")
            await asyncio.sleep(0)
            trace.emit("muse.test", operation=f"{request_id}-end")

    await asyncio.gather(emit("request-A"), emit("request-B"))
    messages = upstream_messages(caplog)
    assert len(messages) == 4
    for message in messages:
        if "request-A-" in message:
            assert "request_id=request-A " in message
        if "request-B-" in message:
            assert "request_id=request-B " in message


def test_trace_failure_records_upgrade_body_and_websocket_close_reason(caplog):
    from muse_proxy._upstream import MuseTrace

    caplog.set_level(5, logger="muse_proxy")
    trace = MuseTrace("router-upgrade", secrets=(ACCESS_TOKEN,))
    failure = ConnectionError("VM edge policy rejected connection")
    failure.response = SimpleNamespace(
        status_code=429,
        body=json.dumps(
            {"error": "connection limit", "detail": ACCESS_TOKEN, "api_key": "unknown-key"}
        ).encode(),
    )
    failure.rcvd = SimpleNamespace(code=1008, reason=f"policy denial: {ACCESS_TOKEN}")
    trace.failure("websocket_open", failure)
    text = "\n".join(upstream_messages(caplog))
    assert "status=429" in text
    assert "close_code=1008" in text
    assert "connection limit" in text
    assert "policy denial" in text
    assert ACCESS_TOKEN not in text
    assert "unknown-key" not in text


def test_trace_unprintable_exception_never_changes_failure_handling(caplog):
    from muse_proxy._upstream import MuseTrace

    class UnprintableError(Exception):
        def __str__(self):
            raise ValueError("do not let diagnostic formatting escape")

    caplog.set_level(5, logger="muse_proxy")
    MuseTrace("router-unprintable").failure("connect", UnprintableError())
    text = "\n".join(upstream_messages(caplog))
    assert "UnprintableError" in text
    assert "do not let diagnostic formatting escape" not in text
