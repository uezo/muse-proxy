import asyncio

import pytest
from musegadget.muse_turn import MuseTurn
from test_muse import (
    SIDE_1,
    SIDE_2,
    RoutingLink,
    assert_closed,
    configured,  # noqa: F401 - required by the imported routed fixture
    event,
    reply,
    route_collect,
    routed,  # noqa: F401 - reused pytest fixture
    terminal,
)

from muse_proxy.errors import ProxyError
from muse_proxy.muse import MuseClient


class ControlledLink(RoutingLink):
    def __init__(self, harness, **kwargs):
        super().__init__(harness, **kwargs)
        self.teardown_started = asyncio.Event()
        self.teardown_release = asyncio.Event()
        self.teardown_complete = asyncio.Event()
        self.number = len(harness.sessions)
        if self.number not in harness.held_closes:
            self.teardown_release.set()

    async def run(self, stop):
        self.connected = True
        self.registered.set()
        try:
            await self.disconnect.wait()
        finally:
            self.teardown_started.set()
            await self.teardown_release.wait()
            self.connected = False
            self.connection_closed = True
            self.teardown_complete.set()
            if self.number in self.harness.close_errors:
                raise ConnectionError("transport teardown failed")

    async def send_chat(self, text, session_id=None, *, output_modality="text"):
        response = await super().send_chat(text, session_id, output_modality=output_modality)
        self.harness.transmissions.put_nowait((self, text, session_id))
        return response


@pytest.fixture
def controlled(request, monkeypatch):
    settings, harness = request.getfixturevalue("routed")
    settings.request_timeout_seconds = 2
    harness.held_closes = set()
    harness.close_errors = set()
    harness.transmissions = asyncio.Queue()
    harness.retired_turns = []
    sdk = harness.sdk()
    sdk.LinkSession = lambda **kwargs: ControlledLink(harness, **kwargs)

    def tracked_turn(*args, **kwargs):
        turn = MuseTurn(*args, **kwargs)
        retire = turn.retire

        def tracked_retire():
            retire()
            harness.retired_turns.append(turn)

        turn.retire = tracked_retire
        return turn

    sdk.MuseTurn = tracked_turn
    monkeypatch.setattr("muse_proxy.muse._load_sdk", lambda: sdk)
    return settings, harness


async def sent(harness):
    return await asyncio.wait_for(harness.transmissions.get(), timeout=1)


async def finish(link, session_id, text, message_id="reply-1"):
    for item in (
        event("task.status", parent=None, session=session_id, task_id="root", status="running"),
        reply(text, message_id=message_id, parent=None, session=session_id),
        terminal(parent=None, session=session_id, task_id="root"),
    ):
        link.queue.put_nowait(item)


async def close_all(client, harness, *requests):
    for link in harness.sessions:
        link.teardown_release.set()
    for request in requests:
        if not request.done():
            request.cancel()
    await asyncio.gather(*requests, return_exceptions=True)
    await client.close()


@pytest.mark.asyncio
async def test_different_sessions_send_before_either_reply_and_keep_replies_isolated(controlled):
    settings, harness = controlled
    client = MuseClient(settings)
    first = asyncio.create_task(route_collect(client, "A", "request-a", SIDE_1))
    second = None
    try:
        a, _, _ = await sent(harness)
        second = asyncio.create_task(route_collect(client, "B", "request-b", SIDE_2))
        b, _, _ = await sent(harness)
        assert a is not b
        assert a.connected and b.connected
        assert not first.done() and not second.done()
        # The same upstream message and task IDs may occur independently in each chat.
        a.queue.put_nowait(reply("Wrong chat", parent=None, session=SIDE_2))
        b.queue.put_nowait(reply("Wrong chat", parent=None, session=SIDE_1))
        await finish(b, SIDE_2, "B")
        assert [item.text for item in await second] == ["B"]
        assert not first.done()
        await finish(a, SIDE_1, "A")
        assert [item.text for item in await first] == ["A"]
        assert_closed(harness)
        assert not client._routers
    finally:
        await close_all(client, harness, *(task for task in (first, second) if task is not None))


@pytest.mark.asyncio
async def test_cold_chat_send_404_cleans_up_and_later_requests_recover(controlled):
    settings, harness = controlled
    harness.subscription_statuses = [404, 404]
    harness.rejections[1] = 404
    client = MuseClient(settings)
    requests = []
    try:
        failed = asyncio.create_task(route_collect(client, "Deleted", "request-failed", SIDE_1))
        requests.append(failed)
        rejected_link, _, _ = await sent(harness)
        failed_router = client._routers[SIDE_1]
        with pytest.raises(ProxyError) as caught:
            await failed
        assert caught.value.code == "muse_request_error"
        assert failed_router.closed.is_set()
        assert failed_router.task.done()
        assert rejected_link.teardown_complete.is_set()
        assert not client._routers
        assert len(harness.sent) == 1

        for index, session_id in enumerate((SIDE_1, SIDE_2), start=1):
            recovered = asyncio.create_task(
                route_collect(client, "Retry", f"request-retry-{index}", session_id)
            )
            requests.append(recovered)
            link, _, actual_session = await sent(harness)
            assert actual_session == session_id
            await finish(link, session_id, f"Recovered {index}", f"reply-{index}")
            assert [item.text for item in await recovered] == [f"Recovered {index}"]
        assert_closed(harness)
        assert not client._routers
    finally:
        await close_all(client, harness, *requests)


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_transport_teardown_and_retires_turn(controlled):
    settings, harness = controlled
    harness.held_closes.add(1)
    client = MuseClient(settings)
    request = asyncio.create_task(route_collect(client, "A", "request-a", SIDE_1))
    try:
        link, _, _ = await sent(harness)
        router = client._routers[SIDE_1]
        request.cancel()
        await asyncio.wait_for(link.teardown_started.wait(), timeout=1)
        request.cancel()
        # Direct router cancellation also occurs when shutdown overlaps request cleanup.
        router.task.cancel()
        await asyncio.sleep(0)
        request.cancel()
        router.task.cancel()
        await asyncio.sleep(0)
        assert not router.closed.is_set()
        assert not request.done()
        assert not link.teardown_complete.is_set()
        link.teardown_release.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert router.closed.is_set()
        assert router.task.done()
        assert link.teardown_complete.is_set()
        assert router.turn in harness.retired_turns
        assert not client._routers
        assert_closed(harness)
    finally:
        await close_all(client, harness, request)


@pytest.mark.asyncio
async def test_same_session_replacement_waits_for_old_teardown_and_replay_retirement(controlled):
    settings, harness = controlled
    harness.held_closes.add(1)
    client = MuseClient(settings)
    requests = []
    try:
        old = asyncio.create_task(route_collect(client, "Old", "request-old", SIDE_1))
        requests.append(old)
        old_link, _, _ = await sent(harness)
        old_router = client._routers[SIDE_1]
        await finish(old_link, SIDE_1, "Old", "old-reply")
        await asyncio.wait_for(old_link.teardown_started.wait(), timeout=1)
        assert old_router.closing
        replacement = asyncio.create_task(route_collect(client, "New", "request-new", SIDE_1))
        other = asyncio.create_task(route_collect(client, "Other", "request-other", SIDE_2))
        requests.extend((replacement, other))
        other_link, text, session_id = await sent(harness)
        assert (text, session_id) == ("Other", SIDE_2)
        await finish(other_link, SIDE_2, "Other")
        assert [item.text for item in await other] == ["Other"]
        assert not replacement.done()
        assert [text for text, _, _ in harness.sent] == ["Old", "Other"]
        assert old_router.turn not in harness.retired_turns

        old_link.teardown_release.set()
        assert [item.text for item in await old] == ["Old"]
        replacement_link, text, session_id = await sent(harness)
        assert (text, session_id) == ("New", SIDE_1)
        assert old_router.closed.is_set()
        assert old_router.turn in harness.retired_turns
        replacement_link.queue.put_nowait(
            reply("Old replay", message_id="old-reply", parent=None, session=SIDE_1)
        )
        await finish(replacement_link, SIDE_1, "New", "new-reply")
        assert [item.text for item in await replacement] == ["New"]
        assert_closed(harness)
        assert not client._routers
    finally:
        await close_all(client, harness, *requests)


@pytest.mark.asyncio
async def test_transport_teardown_exception_still_finishes_router_bookkeeping(controlled):
    settings, harness = controlled
    harness.close_errors.add(1)
    client = MuseClient(settings)
    request = asyncio.create_task(route_collect(client, "A", "request-a", SIDE_1))
    try:
        link, _, _ = await sent(harness)
        router = client._routers[SIDE_1]
        await finish(link, SIDE_1, "A")
        assert [item.text for item in await request] == ["A"]
        assert router.closed.is_set()
        assert router.task.done()
        assert router.turn in harness.retired_turns
        assert not client._routers
        assert_closed(harness)
    finally:
        await close_all(client, harness, request)


@pytest.mark.asyncio
async def test_retirement_exception_releases_session_and_finishes_router_bookkeeping(controlled):
    settings, harness = controlled
    client = MuseClient(settings)
    requests = []
    try:
        request = asyncio.create_task(route_collect(client, "A", "request-a", SIDE_1))
        requests.append(request)
        link, _, _ = await sent(harness)
        router = client._routers[SIDE_1]
        retirement_attempted = asyncio.Event()

        def broken_retire():
            retirement_attempted.set()
            raise RuntimeError("SDK retirement failed")

        router.turn.retire = broken_retire
        await finish(link, SIDE_1, "A")
        try:
            messages = await request
        except ProxyError as error:
            assert error.code == "muse_connection_error"
        else:
            assert [item.text for item in messages] == ["A"]
        assert retirement_attempted.is_set()
        assert router.closed.is_set()
        assert router.task.done()
        assert not client._routers
        assert_closed(harness)

        retry = asyncio.create_task(route_collect(client, "Retry", "request-retry", SIDE_1))
        requests.append(retry)
        retry_link, _, session_id = await sent(harness)
        assert session_id == SIDE_1
        await finish(retry_link, SIDE_1, "Recovered", "retry-reply")
        assert [item.text for item in await retry] == ["Recovered"]
        assert not client._routers
        assert_closed(harness)
    finally:
        await close_all(client, harness, *requests)


@pytest.mark.asyncio
async def test_active_replay_scope_survives_cache_pressure_until_same_session_handoff(controlled):
    settings, harness = controlled
    settings.request_timeout_seconds = 5
    harness.held_closes.add(1)
    client = MuseClient(settings)
    requests = []
    try:
        old = asyncio.create_task(route_collect(client, "Old", "request-old", SIDE_1))
        requests.append(old)
        old_link, _, _ = await sent(harness)
        await finish(old_link, SIDE_1, "Old", "old-reply")
        await asyncio.wait_for(old_link.teardown_started.wait(), timeout=1)

        replacement = asyncio.create_task(route_collect(client, "New", "request-new", SIDE_1))
        requests.append(replacement)
        others = {}
        for number in range(32):
            session_id = f"00000000-0000-4000-8000-{number:012d}"
            request = asyncio.create_task(
                route_collect(client, f"Other {number}", f"request-other-{number}", session_id)
            )
            requests.append(request)
            others[session_id] = request
        links = {}
        for _ in others:
            link, _, session_id = await sent(harness)
            assert session_id in others
            links[session_id] = link
        assert not replacement.done()
        assert len(links) == 32

        # The oldest scope is still active and its replacement is queued. Evicting
        # it to enforce the idle-cache limit would lose the retired message IDs.
        old_link.teardown_release.set()
        assert [item.text for item in await old] == ["Old"]
        replacement_link, text, session_id = await sent(harness)
        assert (text, session_id) == ("New", SIDE_1)
        replacement_link.queue.put_nowait(
            reply("Old replay", message_id="old-reply", parent=None, session=SIDE_1)
        )
        await finish(replacement_link, SIDE_1, "New", "new-reply")
        assert [item.text for item in await replacement] == ["New"]

        for session_id, link in links.items():
            await finish(link, session_id, "Other")
        responses = await asyncio.gather(*others.values())
        assert all([item.text for item in response] == ["Other"] for response in responses)
        assert_closed(harness)
        assert not client._routers
        assert not client._session_gates
        assert len(client._replay_scopes) <= 32
    finally:
        await close_all(client, harness, *requests)
