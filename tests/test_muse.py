import asyncio
import json
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import UUID

import pytest
from musegadget.link_client import HttpStreamError
from musegadget.muse_turn import MuseTurn, ReplayScope, TaskFinished

from muse_proxy.errors import ProxyError
from muse_proxy.events import MuseMessage
from muse_proxy.muse import MuseClient, _chat_session_id, _refuse_command, _RequestTurn

SIDE_1 = "11111111-1111-4111-8111-111111111111"
SIDE_2 = "22222222-2222-4222-8222-222222222222"
SIDE_DEFAULT = "33333333-3333-4333-8333-333333333333"
SIDE_OVERRIDE = "44444444-4444-4444-8444-444444444444"


def event(name, message_id=None, parent="user-1", session="main", **fields):
    payload = {"session_id": session, **fields}
    if parent is not None:
        payload["reply_to_message_id"] = parent
    if message_id:
        payload["message_id"] = message_id
    return {"type": "event", "event": name, "payload": payload}


def reply(text="Hello", message_id="reply-1", **fields):
    return event("delta.message_done", message_id, content=text, **fields)


def terminal(**fields):
    return event("task.status", status="completed", **fields)


@dataclass
class FakeDevice:
    node_id: str
    display_name: str
    version: str
    commands: dict


class FakeLink:
    def __init__(self, harness, **kwargs):
        self.harness = harness
        self.kwargs = kwargs
        self.registered = asyncio.Event()
        self.chat_subscribed = asyncio.Event()
        self.queue = asyncio.Queue()
        self.disconnect = asyncio.Event()
        self.connected = False
        self.subscribed = False
        self.subscription_closed = False
        self.connection_closed = False
        harness.sessions.append(self)

    async def run(self, stop):
        self.connected = True
        try:
            if self.harness.register:
                self.registered.set()
            await self.disconnect.wait()
            return self.harness.outcome
        finally:
            self.connected = False
            self.connection_closed = True

    async def subscribe_chat(self, session_id=None):
        self.harness.subscriptions.append(session_id)
        try:
            if self.harness.subscription_error:
                raise ConnectionError("secret-token-in-network-error")
            status = (
                self.harness.subscription_statuses.pop(0)
                if self.harness.subscription_statuses
                else self.harness.subscription_default_status
            )
            if status != 200:
                raise HttpStreamError(status, "/chat/subscribe")
            self.subscribed = True
            self.chat_subscribed.set()
            while True:
                value = await self.queue.get()
                if value is None:
                    return
                yield value
                if self.queue.empty():
                    self.harness.events_consumed.set()
        finally:
            self.chat_subscribed.clear()
            self.subscription_closed = True

    async def send_chat(self, text, session_id=None, *, output_modality="text"):
        assert self.harness.subscriptions, "must attempt subscription before sending"
        self.harness.sent.append((text, session_id, output_modality))
        self.harness.submitted.set()
        try:
            for raw in self.harness.events:
                await self.queue.put(raw)
            # These events are consumed before the send ACK, as on a fast Muse.
            await asyncio.sleep(0)
            if self.harness.drop:
                self.disconnect.set()
            if self.harness.block_send:
                await asyncio.Event().wait()
            if self.harness.release_ack is not None:
                await self.harness.release_ack.wait()
            return self.harness.ack
        except asyncio.CancelledError:
            self.harness.send_cancelled = True
            raise
        finally:
            self.harness.send_finished.set()


class Harness:
    def __init__(self):
        self.sessions = []
        self.sent = []
        self.submitted = asyncio.Event()
        self.events_consumed = asyncio.Event()
        self.send_finished = asyncio.Event()
        self.send_cancelled = False
        self.release_ack = None
        self.subscriptions = []
        self.subscription_statuses = []
        self.subscription_default_status = 200
        self.events = [reply(), terminal()]
        self.ack = {
            "ok": True,
            "status": 200,
            "response": {"result": {"message_id": "user-1", "session_id": "main"}},
        }
        self.register = True
        self.drop = False
        self.outcome = "closed"
        self.subscription_error = False
        self.block_send = False
        self.fetch_statuses = [200]
        self.fetch_calls = []
        self.refresh_calls = []
        self.refresh_status = 200
        self.vms = [{"vm_id": "vm-1", "vm_auth_token": "test-vm-token", "is_default": True}]

    def fetch(self, token, root):
        self.fetch_calls.append((token, root))
        status = self.fetch_statuses.pop(0) if self.fetch_statuses else 200
        return self.vms if status == 200 else [], status

    def refresh(self, *args):
        self.refresh_calls.append(args)
        if self.refresh_status != 200:
            return None, self.refresh_status
        return {"access_token": "new-access", "refresh_token": "new-refresh"}, 200

    def sdk(self):
        return SimpleNamespace(
            version="test",
            DeviceDescription=FakeDevice,
            MuseTurn=MuseTurn,
            ReplayScope=ReplayScope,
            TaskFinished=TaskFinished,
            HttpStreamError=HttpStreamError,
            LinkSession=lambda **kwargs: FakeLink(self, **kwargs),
            api=SimpleNamespace(
                fetch_vms_with_status=self.fetch, refresh_device_token=self.refresh
            ),
        )


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSEGADGET_SDK_TOKEN", raising=False)
    (tmp_path / "identity.json").write_text(json.dumps({"mac": "02:00:00:ab:cd:ef"}))
    (tmp_path / "pairing.json").write_text(
        json.dumps(
            {
                "access_token": "test-access",
                "refresh_token": "test-refresh",
                "access_token_saved_at": time.time(),
            }
        )
    )
    settings = SimpleNamespace(
        state_dir=tmp_path,
        sdk_token=None,
        vm_id=None,
        request_timeout_seconds=0.3,
        connect_timeout_seconds=0.1,
    )
    harness = Harness()
    monkeypatch.setattr("muse_proxy.muse._load_sdk", harness.sdk)
    return settings, harness


async def collect(client, text="Hi", session_id=None):
    return [chunk async for chunk in client.stream_chat(text, session_id)]


def assert_closed(harness):
    assert harness.sessions
    assert all(session.connection_closed for session in harness.sessions)
    assert all(session.subscription_closed for session in harness.sessions if session.subscribed)


@pytest.mark.asyncio
async def test_real_sdk_correlates_events_before_ack_and_finalizes_text(configured):
    settings, harness = configured
    harness.events = [
        event("delta.text_append", "reply-1", text="draft"),
        reply("Final."),
        reply("Final."),
        event("delta.text_append", "reply-1", text="duplicate draft ignored"),
        reply("Second", message_id="reply-2", parent="reply-1"),
        terminal(),
    ]
    client = MuseClient(settings)
    assert await collect(client) == ["Final.", "\n\nSecond"]
    assert harness.sent == [("Hi", None, "text")]
    assert harness.sessions[0].kwargs["device"].commands == {}
    assert_closed(harness)


@pytest.mark.asyncio
async def test_other_chats_requests_and_parentless_messages_are_ignored(configured):
    settings, harness = configured
    harness.events = [
        reply("Other chat secret", session="other"),
        reply("Other request secret", message_id="other", parent="another-user"),
        reply("Parentless secret", message_id="unbound", parent=None),
        event("task.status", parent="another-user", status="failed"),
        reply("Mine"),
        terminal(),
    ]
    assert await collect(MuseClient(settings)) == ["Mine"]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_explicit_session_and_nondefault_vm(configured):
    settings, harness = configured
    settings.vm_id = "requested-vm"
    harness.vms.append({"vm_id": "requested-vm", "vm_auth_token": "test-vm-token"})
    harness.events = [reply(session=SIDE_1), terminal(session=SIDE_1)]
    harness.ack["response"]["result"].update(session_id=SIDE_1, is_thread=True)
    assert await collect(MuseClient(settings), session_id=SIDE_1) == ["Hello"]
    assert harness.sent[0][1] == SIDE_1
    assert harness.sessions[0].kwargs["vm_id"] == "requested-vm"


@pytest.mark.asyncio
async def test_vm_401_refreshes_once_before_submission(configured):
    settings, harness = configured
    harness.fetch_statuses = [401, 200]
    assert await collect(MuseClient(settings)) == ["Hello"]
    assert [token for token, _ in harness.fetch_calls] == ["test-access", "new-access"]
    assert len(harness.refresh_calls) == 1
    assert len(harness.sent) == 1


@pytest.mark.asyncio
async def test_chat_401_rotates_for_next_request_without_resubmitting(configured):
    settings, harness = configured
    harness.ack = {"ok": False, "status": 401, "response": {"error": "secret"}}
    with pytest.raises(ProxyError, match="refused"):
        await collect(MuseClient(settings))
    assert len(harness.sent) == 1
    assert len(harness.refresh_calls) == 1
    assert (
        json.loads((settings.state_dir / "pairing.json").read_text())["refresh_token"]
        == "new-refresh"
    )
    assert_closed(harness)


@pytest.mark.asyncio
async def test_refresh_rejection_never_submits_or_deletes_credentials(configured):
    settings, harness = configured
    harness.fetch_statuses = [401]
    harness.refresh_status = 401
    original = (settings.state_dir / "pairing.json").read_bytes()
    with pytest.raises(ProxyError, match="refresh token"):
        await collect(MuseClient(settings))
    assert not harness.sent
    assert (settings.state_dir / "pairing.json").read_bytes() == original


@pytest.mark.asyncio
async def test_timeout_cleans_up_both_streams(configured):
    settings, harness = configured
    harness.events = []
    with pytest.raises(ProxyError) as caught:
        await collect(MuseClient(settings))
    assert caught.value.status_code == 504
    assert len(harness.sent) == 1
    assert_closed(harness)


@pytest.mark.asyncio
async def test_disconnect_does_not_retry_chat(configured):
    settings, harness = configured
    harness.events = []
    harness.drop = True
    with pytest.raises(ProxyError, match="disconnected"):
        await collect(MuseClient(settings))
    assert len(harness.sent) == 1
    assert_closed(harness)


@pytest.mark.asyncio
async def test_connect_timeout_cleans_connection_before_submission(configured):
    settings, harness = configured
    harness.register = False
    with pytest.raises(ProxyError, match="connecting"):
        await collect(MuseClient(settings))
    assert not harness.sent
    assert_closed(harness)


@pytest.mark.asyncio
async def test_subscription_failure_is_sanitized_before_submission(configured):
    settings, harness = configured
    harness.subscription_error = True
    with pytest.raises(ProxyError) as caught:
        await collect(MuseClient(settings))
    assert "secret-token" not in str(caught.value)
    assert not harness.sent
    assert_closed(harness)


@pytest.mark.asyncio
async def test_closing_generator_closes_subscription_and_connection(configured):
    settings, harness = configured
    harness.events = [reply(), terminal()]
    client = MuseClient(settings)
    stream = client.stream_chat("Hi")
    assert await anext(stream) == "Hello"
    await stream.aclose()
    assert_closed(harness)


@pytest.mark.asyncio
async def test_later_authoritative_revision_replaces_completed_delta(configured):
    settings, harness = configured
    harness.events = [reply("Draft from message_done")]
    client = MuseClient(settings)
    answer = asyncio.create_task(collect(client))
    await harness.submitted.wait()
    await asyncio.sleep(0.02)
    await harness.sessions[0].queue.put(
        event("message.assistant", "reply-1", display_text="Authoritative final text")
    )
    await harness.sessions[0].queue.put(terminal())
    assert await answer == ["Authoritative final text"]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_parentless_draft_is_replayed_when_final_event_binds_request(configured):
    settings, harness = configured
    harness.events = [
        event("delta.text_append", "reply-1", parent=None, text="Hello"),
        event("delta.text_append", "reply-1", parent=None, text=" from Muse"),
        event("delta.message_done", "reply-1"),
        terminal(),
    ]
    assert await collect(MuseClient(settings)) == ["Hello from Muse"]
    assert_closed(harness)


def test_unbound_drafts_are_bounded_and_do_not_hold_valid_turn_open():
    turn = _RequestTurn(MuseTurn("main"))
    turn.acknowledge({"result": {"message_id": "user-1", "session_id": "main"}})
    for index in range(1000):
        turn.feed(event("delta.text_append", f"unbound-{index}", parent=None, text="x" * 1024))
    assert len(turn.unbound) <= 32
    assert turn.unbound_events <= 64
    assert turn.unbound_bytes <= 256 * 1024
    turn.feed(reply("Correct request"))
    turn.feed(terminal())
    assert turn.complete()
    assert turn.output() == ["Correct request"]


@pytest.mark.asyncio
async def test_paused_consumer_does_not_get_cancelled_outside_iteration(configured):
    settings, harness = configured
    settings.request_timeout_seconds = 0.2
    client = MuseClient(settings)
    stream = client.stream_chat("Hi")
    assert await anext(stream) == "Hello"
    assert_closed(harness)
    # No generator-owned timer may cancel this unrelated caller wait.
    await asyncio.sleep(0.25)
    with pytest.raises(ProxyError) as caught:
        await anext(stream)
    assert caught.value.code == "muse_timeout"
    assert asyncio.current_task().cancelling() == 0
    await stream.aclose()


@pytest.mark.asyncio
async def test_repeated_request_cancellation_keeps_lock_until_refresh_finishes(configured):
    settings, harness = configured
    settings.request_timeout_seconds = 2
    path = settings.state_dir / "pairing.json"
    original = json.loads(path.read_text())
    path.write_text(json.dumps({**original, "access_token_saved_at": 0}))
    started = threading.Event()
    release = threading.Event()

    def delayed_refresh(*args):
        started.set()
        release.wait(timeout=2)
        return {"access_token": "rotated-access", "refresh_token": "rotated-refresh"}, 200

    harness.refresh = delayed_refresh
    client = MuseClient(settings)
    task = asyncio.create_task(collect(client))
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    try:
        assert not task.done()
        assert client._store._refresh_lock.locked()
        assert not harness.sent
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not client._session_gates
    assert json.loads(path.read_text())["refresh_token"] == "rotated-refresh"
    await client.close()


@pytest.mark.asyncio
async def test_cancellation_and_shutdown_join_tasks(configured):
    settings, harness = configured
    settings.request_timeout_seconds = 2
    harness.events = []
    harness.block_send = True
    client = MuseClient(settings)
    task = asyncio.create_task(collect(client))
    await harness.submitted.wait()
    await client.close()
    assert task.cancelled()
    assert_closed(harness)
    with pytest.raises(ProxyError, match="shutting down"):
        await collect(client)


@pytest.mark.asyncio
async def test_upstream_requests_are_serialized(configured):
    settings, harness = configured
    client = MuseClient(settings)
    first, second = await asyncio.gather(collect(client, "First"), collect(client, "Second"))
    assert first == second == ["Hello"]
    assert len(harness.sessions) == 2
    assert_closed(harness)


def test_device_callback_refuses_every_command():
    result = _refuse_command("exec", {"command": "do-not-run"}, 1000)
    assert result["ok"] is False
    assert result["error"]["code"] == "unsupported_command"


def owned_ack(session=SIDE_1, message_id="user-1", **fields):
    return {
        "ok": True,
        "status": 200,
        "response": {"message_id": message_id, "session_id": session, "is_thread": True, **fields},
    }


def owned_events(text="Side reply", message_id="side-reply", session=SIDE_1, task_id="side-task"):
    return [
        event("task.status", parent=None, session=session, status="running", task_id=task_id),
        event("delta.message_start", message_id, parent=None, session=session),
        event(
            "delta.text_append",
            message_id,
            parent=None,
            session=session,
            parent_message_id=message_id,
            text=text,
        ),
        event("delta.message_done", message_id, parent=None, session=session),
        event("task.status", parent=None, session=session, status="completed", task_id=task_id),
    ]


@pytest.mark.asyncio
async def test_owned_side_chat_accepts_parentless_and_self_parent_events_before_flat_ack(
    configured,
):
    settings, harness = configured
    harness.ack = owned_ack()
    harness.events = owned_events("Hello from side chat")
    assert await collect(MuseClient(settings), session_id=SIDE_1) == ["Hello from side chat"]
    assert harness.sent == [("Hi", SIDE_1, "text")]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_owned_side_chat_rejects_wrong_and_missing_event_sessions(configured):
    settings, harness = configured
    harness.ack = owned_ack()
    harness.events = [
        *owned_events("Other session secret", session="other"),
        *owned_events("Missing session secret", session=None),
        event("task.status", parent=None, session="other", status="failed", task_id="wrong"),
        *owned_events("Mine"),
    ]
    assert await collect(MuseClient(settings), session_id=SIDE_1) == ["Mine"]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_owned_side_chat_rejects_replayed_ids_from_previous_request(configured):
    settings, harness = configured
    harness.ack = owned_ack()
    harness.events = owned_events("First", message_id="first-reply", task_id="first-task")
    client = MuseClient(settings)
    assert await collect(client, session_id=SIDE_1) == ["First"]

    harness.ack = owned_ack(message_id="user-2")
    harness.events = [
        *owned_events("Replayed secret", message_id="first-reply", task_id="first-task"),
        reply(
            "Retired parent secret", message_id="late-reply", parent="first-reply", session=SIDE_1
        ),
        event("task.status", parent=None, session=SIDE_1, status="failed", task_id="first-task"),
        *owned_events("Second", message_id="second-reply", task_id="second-task"),
    ]
    assert await collect(client, session_id=SIDE_1) == ["Second"]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_owned_side_chat_replay_scopes_are_independent_per_session(configured):
    settings, harness = configured
    harness.ack = owned_ack()
    harness.events = owned_events("First")
    client = MuseClient(settings)
    assert await collect(client, session_id=SIDE_1) == ["First"]

    harness.ack = owned_ack(session=SIDE_2, message_id="user-2")
    harness.events = owned_events("Second", session=SIDE_2)
    assert await collect(client, session_id=SIDE_2) == ["Second"]
    assert_closed(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "acknowledgment",
    [
        {"message_id": "user-1", "session_id": "other", "is_thread": True},
        {"message_id": "user-1", "session_id": SIDE_1, "is_thread": False},
        {"message_id": "user-1", "session_id": SIDE_1},
    ],
)
async def test_owned_side_chat_requires_acknowledged_thread(configured, acknowledgment):
    settings, harness = configured
    harness.ack = {"ok": True, "status": 200, "response": acknowledgment}
    harness.events = owned_events("Must not leak")
    with pytest.raises(ProxyError) as caught:
        await collect(MuseClient(settings), session_id=SIDE_1)
    assert caught.value.code == "muse_protocol_error"
    assert_closed(harness)


@pytest.mark.asyncio
async def test_owned_side_chat_preserves_sdk_accepted_terminal_failure(configured):
    settings, harness = configured
    harness.ack = owned_ack()
    harness.events = owned_events("Partial response")
    harness.events[-1]["payload"]["status"] = "failed"
    with pytest.raises(ProxyError) as caught:
        await collect(MuseClient(settings), session_id=SIDE_1)
    assert caught.value.code == "muse_task_error"
    assert_closed(harness)


@pytest.mark.asyncio
async def test_shared_main_chat_still_rejects_parentless_and_self_parent_replies(configured):
    settings, harness = configured
    harness.events = [
        *owned_events("Unrelated main chat secret", session="main"),
        reply("Mine", message_id="bound-reply"),
        terminal(),
    ]
    assert await collect(MuseClient(settings)) == ["Mine"]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_owned_side_chat_retires_observed_ids_after_timeout(configured):
    settings, harness = configured
    harness.ack = owned_ack()
    harness.events = [event("delta.message_start", "unfinished-reply", parent=None, session=SIDE_1)]
    client = MuseClient(settings)
    with pytest.raises(ProxyError) as caught:
        await collect(client, session_id=SIDE_1)
    assert caught.value.code == "muse_timeout"

    harness.ack = owned_ack(message_id="user-2")
    harness.events = [
        reply(
            "Timed-out replay secret", message_id="unfinished-reply", parent=None, session=SIDE_1
        ),
        *owned_events("Current reply", message_id="new-reply", task_id="new-task"),
    ]
    assert await collect(client, session_id=SIDE_1) == ["Current reply"]
    assert_closed(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_session", [None, SIDE_OVERRIDE])
async def test_configured_default_side_chat_and_explicit_override(configured, requested_session):
    settings, harness = configured
    settings.default_session_id = SIDE_DEFAULT
    expected_session = requested_session or settings.default_session_id
    harness.ack = owned_ack(session=expected_session)
    harness.events = owned_events("Configured side chat", session=expected_session)
    assert await collect(MuseClient(settings), session_id=requested_session) == [
        "Configured side chat"
    ]
    assert harness.sent == [("Hi", expected_session, "text")]
    assert_closed(harness)


def test_chat_session_labels_map_to_stable_node_specific_uuids():
    first = _chat_session_id("node-one", "muse-proxy")
    assert str(UUID(first)) == first
    assert first == _chat_session_id("node-one", "muse-proxy")
    assert first != _chat_session_id("node-two", "muse-proxy")
    assert first != _chat_session_id("node-one", "another-chat")


@pytest.mark.parametrize(
    "session", [SIDE_1, SIDE_1.upper(), "{B41FDF09-824A-4BD4-B619-F0BB583144B4}"]
)
def test_chat_session_valid_uuid_is_preserved_canonically(session):
    assert _chat_session_id("node-one", session) == str(UUID(session))


@pytest.mark.asyncio
async def test_named_session_is_mapped_before_subscription_and_send(configured):
    settings, harness = configured
    session = _chat_session_id("homelink-abcdef", "friendly-label")
    harness.ack = owned_ack(session=session)
    harness.events = owned_events("Named chat", session=session)
    assert await collect(MuseClient(settings), session_id="friendly-label") == ["Named chat"]
    assert harness.subscriptions == [session]
    assert harness.sent == [("Hi", session, "text")]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_cold_side_chat_404_submits_once_and_consumes_reply_before_ack(configured):
    settings, harness = configured
    settings.request_timeout_seconds = 1
    harness.subscription_statuses = [404, 200]
    harness.ack = owned_ack()
    harness.events = owned_events("Fast cold-chat reply")
    harness.release_ack = asyncio.Event()
    client = MuseClient(settings)
    task = asyncio.create_task(collect(client, session_id=SIDE_1))
    try:
        await asyncio.wait_for(harness.submitted.wait(), timeout=0.5)
        await asyncio.wait_for(harness.events_consumed.wait(), timeout=0.5)
        assert not task.done()
        harness.release_ack.set()
        assert await task == ["Fast cold-chat reply"]
    finally:
        await client.close()
    assert harness.subscriptions == [SIDE_1, SIDE_1]
    assert harness.sent == [("Hi", SIDE_1, "text")]
    assert harness.send_finished.is_set()
    assert not harness.send_cancelled
    assert_closed(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 500])
async def test_cold_side_chat_only_404_allows_submission(configured, status):
    settings, harness = configured
    harness.subscription_statuses = [status]
    harness.ack = owned_ack()
    harness.events = owned_events("Should not be sent")
    with pytest.raises(ProxyError):
        await collect(MuseClient(settings), session_id=SIDE_1)
    assert harness.subscriptions == [SIDE_1]
    assert not harness.sent
    assert_closed(harness)


@pytest.mark.asyncio
async def test_main_subscription_404_never_submits(configured):
    settings, harness = configured
    harness.subscription_statuses = [404]
    with pytest.raises(ProxyError):
        await collect(MuseClient(settings))
    assert harness.subscriptions == [None]
    assert not harness.sent
    assert_closed(harness)


@pytest.mark.asyncio
async def test_cold_side_chat_repeated_404_times_out_and_cancels_single_submission(configured):
    settings, harness = configured
    settings.request_timeout_seconds = 0.2
    harness.subscription_default_status = 404
    harness.block_send = True
    harness.ack = owned_ack()
    harness.events = owned_events("Unreceived reply")
    client = MuseClient(settings)
    with pytest.raises(ProxyError) as caught:
        await asyncio.wait_for(collect(client, session_id=SIDE_1), timeout=1)
    assert caught.value.code == "muse_timeout"
    assert harness.sent == [("Hi", SIDE_1, "text")]
    assert harness.send_finished.is_set()
    assert harness.send_cancelled
    assert not client._session_gates
    assert_closed(harness)


@pytest.mark.asyncio
async def test_cold_side_chat_refused_request_is_not_resubmitted(configured):
    settings, harness = configured
    harness.subscription_statuses = [404, 200]
    harness.ack = {"ok": False, "status": 500, "response": {"error": "secret"}}
    harness.events = []
    with pytest.raises(ProxyError, match="refused"):
        await collect(MuseClient(settings), session_id=SIDE_1)
    assert harness.sent == [("Hi", SIDE_1, "text")]
    assert harness.send_finished.is_set()
    assert_closed(harness)


@pytest.mark.asyncio
async def test_progress_is_available_while_waiting_and_final_reply_closes_upstream(configured):
    settings, harness = configured
    settings.request_timeout_seconds = 1
    harness.events = [event("agent.status", activity_code="thinking", activity_text="Considering")]
    client = MuseClient(settings)
    stream = client.stream_events("Hi")
    try:
        progress = await asyncio.wait_for(anext(stream), timeout=0.5)
        assert progress == MuseMessage("progress", "Considering", activity_code="thinking")
        assert not harness.sessions[0].connection_closed
        await harness.sessions[0].queue.put(reply("Ready"))
        await harness.sessions[0].queue.put(terminal())
        assert await asyncio.wait_for(anext(stream), timeout=0.5) == MuseMessage(
            "reply", "Ready", message_id="reply-1"
        )
        assert_closed(harness)
    finally:
        await stream.aclose()
    assert not client._session_gates


@pytest.mark.asyncio
async def test_pre_ack_progress_is_ordered_deduplicated_and_public_only(configured):
    settings, harness = configured
    settings.request_timeout_seconds = 1
    harness.release_ack = asyncio.Event()
    harness.events = [
        event("agent.status", activity_code="thinking", activity_text="First\x1b"),
        event("agent.status", activity_code="thinking", activity_text="First"),
        event("agent.status", activity_code="idle", activity_text="Do not publish idle"),
        event("agent.status", activity_code="online", activity_text="Do not publish online"),
        event("agent.status", activity_text="  "),
        event("agent.status", activity_code="researching"),
        event("agent.status", activity_text="Public progress"),
        event(
            "agent.status", activity_code="working", activity_text="Wrong request", parent="other"
        ),
        reply("Done"),
        terminal(),
    ]
    client = MuseClient(settings)
    task = asyncio.create_task(_collect_events(client))
    try:
        await asyncio.wait_for(harness.events_consumed.wait(), timeout=0.5)
        assert not task.done()
        harness.release_ack.set()
        assert await task == [
            MuseMessage("progress", "First", activity_code="thinking"),
            MuseMessage("progress", "researching", activity_code="researching"),
            MuseMessage("progress", "Public progress"),
            MuseMessage("reply", "Done", message_id="reply-1"),
        ]
    finally:
        await client.close()
    assert_closed(harness)


async def _collect_events(client, session_id=None):
    return [message async for message in client.stream_events("Hi", session_id)]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["cancelled", "canceled"])
async def test_native_task_cancel_is_one_terminal_event_and_drops_partial_reply(configured, status):
    settings, harness = configured
    harness.events = [
        event("agent.status", activity_code="working"),
        reply("Partial reply must not be exposed"),
        event("task.status", status=status, superseded_by="untrusted-remote-id"),
    ]
    client = MuseClient(settings)
    assert await _collect_events(client) == [
        MuseMessage("progress", "working", activity_code="working"),
        MuseMessage("cancel"),
    ]
    assert_closed(harness)
    assert not client._session_gates


@pytest.mark.asyncio
async def test_legacy_text_stream_preserves_task_error_on_native_cancel(configured):
    settings, harness = configured
    harness.events = [event("task.status", status="cancelled")]
    with pytest.raises(ProxyError) as caught:
        await collect(MuseClient(settings))
    assert caught.value.code == "muse_task_error"
    assert_closed(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "errored"])
async def test_stream_events_task_failure_remains_an_error(configured, status):
    settings, harness = configured
    harness.events = [reply("Partial reply"), event("task.status", status=status)]
    with pytest.raises(ProxyError) as caught:
        await _collect_events(MuseClient(settings))
    assert caught.value.code == "muse_task_error"
    assert_closed(harness)


@pytest.mark.asyncio
async def test_main_chat_unrelated_progress_cancel_and_invalid_event_type_are_ignored(configured):
    settings, harness = configured
    invalid_cancel = event("task.status", status="cancelled", task_id="invalid-task")
    invalid_cancel["type"] = "response"
    harness.events = [
        event("agent.status", activity_code="thinking", activity_text="Private", parent="other"),
        event("task.status", status="cancelled", parent="other", task_id="other-task"),
        event("task.status", status="failed", session="another-chat"),
        invalid_cancel,
        event("task.status", parent=None, task_id="invalid-task", status="cancelled"),
        reply("Mine"),
        terminal(),
    ]
    assert await _collect_events(MuseClient(settings)) == [
        MuseMessage("reply", "Mine", message_id="reply-1")
    ]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_owned_chat_replayed_and_wrong_session_progress_cancel_failures_are_ignored(
    configured,
):
    settings, harness = configured
    harness.ack = owned_ack()
    harness.events = owned_events("First", message_id="first-reply", task_id="first-task")
    client = MuseClient(settings)
    assert await _collect_events(client, SIDE_1) == [
        MuseMessage("reply", "First", message_id="first-reply")
    ]
    harness.ack = owned_ack(message_id="user-2")
    harness.events = [
        event(
            "agent.status",
            parent=None,
            session=SIDE_1,
            task_id="first-task",
            activity_code="thinking",
            activity_text="Replayed secret",
        ),
        event("task.status", parent=None, session=SIDE_1, task_id="first-task", status="cancelled"),
        event("task.status", parent=None, session=SIDE_1, task_id="first-task", status="failed"),
        event(
            "agent.status",
            parent=None,
            session=SIDE_2,
            activity_code="thinking",
            activity_text="Wrong chat secret",
        ),
        event("task.status", parent=None, session=SIDE_2, task_id="other-task", status="cancelled"),
        *owned_events("Second", message_id="second-reply", task_id="second-task"),
    ]
    assert await _collect_events(client, SIDE_1) == [
        MuseMessage("reply", "Second", message_id="second-reply")
    ]
    assert_closed(harness)


def test_pending_public_progress_is_bounded_and_blank_statuses_are_skipped():
    turn = _RequestTurn(MuseTurn("main"))
    turn.acknowledge({"result": {"message_id": "user-1", "session_id": "main"}})
    for index in range(200):
        turn.feed(event("agent.status", activity_code="working", activity_text=f"Step {index}"))
    assert len(turn.progress) == 64
    assert turn.progress[0].text == "Step 136"
    assert turn.progress[-1].text == "Step 199"
    turn.feed(event("agent.status", activity_code=" ", activity_text=" "))
    assert len(turn.progress) == 64
    assert turn.progress[-1].text == "Step 199"


@pytest.mark.asyncio
async def test_closing_progress_stream_cleans_up_and_does_not_send_remote_cancel(configured):
    settings, harness = configured
    harness.events = [event("agent.status", activity_code="working")]
    client = MuseClient(settings)
    stream = client.stream_events("Hi")
    assert (await anext(stream)).kind == "progress"
    await stream.aclose()
    assert_closed(harness)
    assert not client._session_gates
    assert harness.sent == [("Hi", None, "text")]


@pytest.mark.asyncio
async def test_timeout_after_progress_cleans_up_and_does_not_report_cancel(configured):
    settings, harness = configured
    harness.events = [event("agent.status", activity_code="working")]
    client = MuseClient(settings)
    stream = client.stream_events("Hi")
    assert (await anext(stream)).kind == "progress"
    with pytest.raises(ProxyError) as caught:
        await anext(stream)
    assert caught.value.code == "muse_timeout"
    assert_closed(harness)
    assert not client._session_gates


@pytest.mark.asyncio
async def test_new_request_waits_for_progress_stream_without_auto_cancelling(configured):
    settings, harness = configured
    settings.request_timeout_seconds = 1
    harness.events = [event("agent.status", activity_code="working")]
    client = MuseClient(settings)
    first = client.stream_events("Hi")
    assert (await anext(first)).kind == "progress"
    second = asyncio.create_task(_collect_events(client))
    try:
        await asyncio.sleep(0.02)
        assert not second.done()
        assert len(harness.sent) == 1
        harness.events = [reply("Second"), terminal()]
        await harness.sessions[0].queue.put(reply("First"))
        await harness.sessions[0].queue.put(terminal())
        assert await anext(first) == MuseMessage("reply", "First", message_id="reply-1")
        await first.aclose()
        assert await second == [MuseMessage("reply", "Second", message_id="reply-1")]
        assert len(harness.sent) == 2
    finally:
        await first.aclose()
        await client.close()
    assert_closed(harness)


class RoutingLink(FakeLink):
    async def send_chat(self, text, session_id=None, *, output_modality="text"):
        self.harness.sent.append((text, session_id, output_modality))
        number = len(self.harness.sent)
        gate = self.harness.ack_gates.get(number)
        if gate is not None:
            await gate.wait()
        if number in self.harness.rejections:
            return {"ok": False, "status": self.harness.rejections[number]}
        return self.harness.ack_overrides.get(
            number,
            {
                "ok": True,
                "response": {
                    "result": {
                        "message_id": f"user-{number}",
                        "session_id": session_id or "main",
                        "is_thread": session_id is not None,
                    }
                },
            },
        )


@pytest.fixture
def routed(configured, monkeypatch):
    settings, harness = configured
    settings.request_timeout_seconds = 1
    harness.events = []
    harness.ack_gates = {}
    harness.rejections = {}
    harness.ack_overrides = {}
    sdk = harness.sdk()
    sdk.LinkSession = lambda **kwargs: RoutingLink(harness, **kwargs)
    monkeypatch.setattr("muse_proxy.muse._load_sdk", lambda: sdk)
    return settings, harness


async def wait_until(predicate):
    async with asyncio.timeout(1):
        while not predicate():  # noqa: ASYNC110 - inspect asynchronous fake transport state
            await asyncio.sleep(0.001)


async def route_collect(client, text, request_id, session_id=SIDE_1):
    return [item async for item in client.stream_request(text, session_id, request_id=request_id)]


async def push(harness, *events):
    for item in events:
        await harness.sessions[-1].queue.put(item)


@pytest.mark.asyncio
async def test_routed_parentless_answers_supersede_old_waiter_but_wait_for_root(routed):
    settings, harness = routed
    client = MuseClient(settings)
    received = []
    first = asyncio.create_task(route_collect(client, "Sapporo", "request-a"))
    await wait_until(lambda: harness.sent)

    async def collect_latest():
        async for item in client.stream_request("Hakata too", SIDE_1, request_id="request-b"):
            received.append(item)

    second = asyncio.create_task(collect_latest())
    await wait_until(lambda: len(harness.sent) == 2)
    await wait_until(lambda: next(iter(client._routers.values())).latest.request_id == "request-b")
    await push(
        harness,
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
        event("task.status", session=SIDE_1, parent=None, task_id="child", status="running"),
        reply("Sapporo", session=SIDE_1, parent=None),
        terminal(session=SIDE_1, parent=None, task_id="child"),
    )
    assert await first == [MuseMessage("cancel", superseded_by="request-b")]
    await wait_until(lambda: any(item.kind == "reply" for item in received))
    assert not second.done()
    assert not harness.sessions[0].connection_closed
    await push(
        harness,
        reply("Hakata", message_id="reply-2", session=SIDE_1, parent="reply-2"),
        terminal(session=SIDE_1, parent=None, task_id="root"),
    )
    await second
    assert received == [
        MuseMessage("reply", "Sapporo", message_id="reply-1"),
        MuseMessage("reply", "\n\nHakata", message_id="reply-2"),
    ]
    assert_closed(harness)
    assert not client._routers


@pytest.mark.asyncio
async def test_routed_linked_replies_keep_independent_request_owners(routed):
    settings, harness = routed
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a"))
    await wait_until(lambda: len(harness.sent) == 1)
    second = asyncio.create_task(route_collect(client, "B", "request-b"))
    await wait_until(lambda: len(harness.sent) == 2)
    await push(
        harness,
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
        event("delta.message_start", "reply-a", parent="user-1", session=SIDE_1),
        reply("A", message_id="reply-a", parent=None, session=SIDE_1),
        reply("A detail", message_id="detail-a", parent="reply-a", session=SIDE_1),
        reply("B", message_id="reply-b", parent="user-2", session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    )
    a, b = await asyncio.gather(first, second)
    assert [item.text for item in a] == ["A", "\n\nA detail"]
    assert [item.text for item in b] == ["B"]
    assert all(item.kind == "reply" for item in a + b)
    assert_closed(harness)


@pytest.mark.asyncio
async def test_routed_failed_new_submission_does_not_cancel_existing_request(routed):
    settings, harness = routed
    harness.rejections[2] = 500
    harness.ack_gates[2] = asyncio.Event()
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a"))
    await wait_until(lambda: len(harness.sent) == 1)
    second = asyncio.create_task(route_collect(client, "B", "request-b"))
    await wait_until(lambda: len(harness.sent) == 2)
    await push(
        harness,
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
        reply("A remains active", parent=None, session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    )
    await wait_until(lambda: len(next(iter(client._routers.values())).buffered) == 3)
    assert not first.done()
    harness.ack_gates[2].set()
    with pytest.raises(ProxyError) as refused:
        await second
    assert refused.value.code == "muse_request_error"
    assert [item.text for item in await first] == ["A remains active"]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_routed_events_before_new_ack_are_assigned_after_ack(routed):
    settings, harness = routed
    harness.ack_gates[2] = asyncio.Event()
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a"))
    await wait_until(lambda: len(harness.sent) == 1)
    second = asyncio.create_task(route_collect(client, "B", "request-b"))
    await wait_until(lambda: len(harness.sent) == 2)
    await push(
        harness,
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
        reply("AB combined", parent=None, session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    )
    await wait_until(lambda: len(next(iter(client._routers.values())).buffered) == 3)
    assert not first.done() and not second.done()
    harness.ack_gates[2].set()
    assert (await first)[0].superseded_by == "request-b"
    assert [item.text for item in await second] == ["AB combined"]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_routed_known_link_to_superseded_request_is_not_reassigned(routed):
    settings, harness = routed
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a"))
    await wait_until(lambda: len(harness.sent) == 1)
    second = asyncio.create_task(route_collect(client, "B", "request-b"))
    await wait_until(lambda: len(harness.sent) == 2)
    await push(
        harness,
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
        reply("B owns this", parent=None, session=SIDE_1),
        reply("Old A answer", message_id="old-a", parent="user-1", session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    )
    assert (await first)[0].kind == "cancel"
    assert [item.text for item in await second] == ["B owns this"]


@pytest.mark.asyncio
async def test_routed_message_done_alone_never_closes_stream(routed):
    settings, harness = routed
    settings.request_timeout_seconds = 0.12
    client = MuseClient(settings)
    request = asyncio.create_task(route_collect(client, "A", "request-a"))
    await wait_until(lambda: len(harness.sent) == 1)
    await push(harness, reply("A", parent=None, session=SIDE_1))
    with pytest.raises(ProxyError) as timeout:
        await request
    assert timeout.value.code == "muse_timeout"
    assert_closed(harness)


@pytest.mark.asyncio
async def test_routed_root_task_id_can_be_reused_and_old_messages_are_filtered(routed):
    settings, harness = routed
    client = MuseClient(settings)
    for number in (1, 2):
        request = asyncio.create_task(route_collect(client, f"A{number}", f"request-{number}"))
        await wait_until(lambda number=number: len(harness.sent) == number)
        events = [
            event(
                "task.status", session=SIDE_1, parent=None, task_id="stable-agent", status="running"
            )
        ]
        if number == 2:
            events.append(reply("REPLAY", message_id="reply-1", parent=None, session=SIDE_1))
        events.extend(
            [
                reply(str(number), message_id=f"reply-{number}", parent=None, session=SIDE_1),
                terminal(parent=None, session=SIDE_1, task_id="stable-agent"),
            ]
        )
        await push(harness, *events)
        assert [item.text for item in await request] == [str(number)]
    assert len(harness.sessions) == 2
    assert_closed(harness)


@pytest.mark.asyncio
async def test_routed_sessions_use_independent_connections_before_either_completes(routed):
    settings, harness = routed
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a", SIDE_1))
    await wait_until(lambda: len(harness.sent) == 1)
    first_link = harness.sessions[0]
    second = asyncio.create_task(route_collect(client, "B", "request-b", SIDE_2))
    await wait_until(lambda: len(harness.sent) == 2)
    assert first_link.connected and harness.sessions[1].connected
    await push(
        harness,
        event("task.status", session=SIDE_2, parent=None, task_id="root", status="running"),
        reply("B", parent=None, session=SIDE_2),
        terminal(parent=None, session=SIDE_2, task_id="root"),
    )
    assert [item.text for item in await second] == ["B"]
    assert not first.done()
    for item in (
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
        reply("A", parent=None, session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    ):
        first_link.queue.put_nowait(item)
    assert [item.text for item in await first] == ["A"]
    assert_closed(harness)
    assert not client._session_gates


@pytest.mark.asyncio
async def test_routed_invalid_follow_up_ack_does_not_poison_old_turn(routed):
    settings, harness = routed
    harness.ack_overrides[2] = {
        "ok": True,
        "response": {
            "result": {"message_id": "poison-id", "session_id": SIDE_2, "is_thread": True}
        },
    }
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a"))
    await wait_until(lambda: len(harness.sent) == 1)
    second = asyncio.create_task(route_collect(client, "B", "request-b"))
    with pytest.raises(ProxyError) as rejected:
        await second
    assert rejected.value.code == "muse_protocol_error"
    router = next(iter(client._routers.values()))
    assert "poison-id" not in router.turn._user_ids
    assert router.latest.request_id == "request-a"
    await push(
        harness,
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
        reply("A", parent=None, session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    )
    assert [item.text for item in await first] == ["A"]


@pytest.mark.asyncio
async def test_routed_root_agent_id_handles_child_running_before_root(routed):
    settings, harness = routed
    client = MuseClient(settings)
    received = []

    async def collect_request():
        async for item in client.stream_request("A", SIDE_1, request_id="request-a"):
            received.append(item)

    request = asyncio.create_task(collect_request())
    await wait_until(lambda: len(harness.sent) == 1)
    await push(
        harness,
        event(
            "task.status",
            session=SIDE_1,
            parent=None,
            task_id="child",
            root_agent_id="root",
            status="running",
        ),
        reply("First", parent=None, session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="child", root_agent_id="root"),
    )
    await wait_until(lambda: received)
    await asyncio.sleep(0.07)
    assert not request.done()
    await push(
        harness,
        event(
            "task.status",
            session=SIDE_1,
            parent=None,
            task_id="root",
            root_agent_id="root",
            status="running",
        ),
        reply("Second", message_id="reply-2", parent=None, session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="root", root_agent_id="root"),
    )
    await request
    assert [item.text for item in received] == ["First", "\n\nSecond"]


@pytest.mark.asyncio
async def test_routed_main_chat_applies_parentless_policy_with_exact_session_filter(routed):
    settings, harness = routed
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a", None))
    await wait_until(lambda: len(harness.sent) == 1)
    second = asyncio.create_task(route_collect(client, "B", "request-b", None))
    await wait_until(lambda: len(harness.sent) == 2)
    await push(
        harness,
        reply("WRONG", parent=None, session=SIDE_1),
        reply("MISSING", parent=None, session=None),
        event("task.status", parent=None, task_id="root", status="running"),
        reply("Main reply", parent=None),
        terminal(parent=None, task_id="root"),
    )
    assert (await first)[0].superseded_by == "request-b"
    assert [item.text for item in await second] == ["Main reply"]
    assert_closed(harness)


@pytest.mark.asyncio
async def test_routed_unknown_nonempty_parent_does_not_supersede_pending_requests(routed):
    settings, harness = routed
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a"))
    await wait_until(lambda: len(harness.sent) == 1)
    second = asyncio.create_task(route_collect(client, "B", "request-b"))
    await wait_until(lambda: len(harness.sent) == 2)
    await push(
        harness,
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
        reply("Unrelated", parent="unknown-parent", message_id="unknown", session=SIDE_1),
    )
    await asyncio.sleep(0.03)
    assert not first.done() and not second.done()
    await push(
        harness,
        reply("A", parent="user-1", message_id="reply-a", session=SIDE_1),
        reply("B", parent="user-2", message_id="reply-b", session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    )
    a, b = await asyncio.gather(first, second)
    assert [item.text for item in a] == ["A"]
    assert [item.text for item in b] == ["B"]


@pytest.mark.asyncio
async def test_routed_authoritative_revision_after_emission_is_safe_error(routed):
    settings, harness = routed
    client = MuseClient(settings)
    stream = client.stream_request("A", SIDE_1, request_id="request-a")
    next_reply = asyncio.create_task(anext(stream))
    await wait_until(lambda: len(harness.sent) == 1)
    await push(
        harness,
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
        reply("First revision", parent=None, session=SIDE_1),
    )
    assert (await next_reply).text == "First revision"
    await push(
        harness,
        event(
            "message.assistant",
            "reply-1",
            parent=None,
            session=SIDE_1,
            display_text="Changed final",
        ),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    )
    with pytest.raises(ProxyError) as error:
        await anext(stream)
    assert error.value.code == "muse_protocol_error"
    assert_closed(harness)


@pytest.mark.asyncio
async def test_routed_terminal_before_reordered_message_waits_for_late_text(routed):
    settings, harness = routed
    client = MuseClient(settings)
    request = asyncio.create_task(route_collect(client, "A", "request-a"))
    await wait_until(lambda: len(harness.sent) == 1)
    await push(
        harness,
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    )
    await asyncio.sleep(0.08)
    assert not request.done()
    await push(harness, reply("Late text", parent=None, session=SIDE_1))
    assert [item.text for item in await request] == ["Late text"]


@pytest.mark.asyncio
async def test_routed_old_completion_cannot_finish_new_request_without_new_work(routed):
    settings, harness = routed
    settings.request_timeout_seconds = 0.25
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a"))
    await wait_until(lambda: len(harness.sent) == 1)
    await push(
        harness,
        event("task.status", session=SIDE_1, parent=None, task_id="root", status="running"),
    )
    await wait_until(lambda: next(iter(client._routers.values())).root_task == "root")
    second = asyncio.create_task(route_collect(client, "B", "request-b"))
    await wait_until(lambda: len(harness.sent) == 2)
    await push(
        harness,
        reply("A only", parent="user-1", session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    )
    await asyncio.sleep(0.07)
    assert not second.done()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(
        isinstance(result, ProxyError) and result.code == "muse_timeout" for result in results
    )
    assert_closed(harness)


@pytest.mark.asyncio
async def test_routed_newest_disconnect_leaves_older_pending_request_live(routed):
    settings, harness = routed
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a"))
    await wait_until(lambda: len(harness.sent) == 1)
    await push(
        harness, event("task.status", session=SIDE_1, parent=None, task_id="root", status="running")
    )
    await wait_until(lambda: next(iter(client._routers.values())).root_task == "root")
    second = client.stream_request("B", SIDE_1, request_id="request-b")
    progress = asyncio.create_task(anext(second))
    await wait_until(lambda: len(harness.sent) == 2)
    await push(
        harness,
        event(
            "agent.status",
            session=SIDE_1,
            parent=None,
            activity_code="working",
            activity_text="Working",
        ),
    )
    assert (await progress).kind == "progress"
    await second.aclose()
    assert not harness.sessions[0].connection_closed
    await push(
        harness,
        reply("A still receives this", parent=None, session=SIDE_1),
        terminal(parent=None, session=SIDE_1, task_id="root"),
    )
    assert [item.text for item in await first] == ["A still receives this"]
    assert_closed(harness)
