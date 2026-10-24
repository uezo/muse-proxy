"""One chat subscription shared by overlapping OpenAI-compatible requests."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from contextlib import suppress
from typing import Any
from uuid import uuid4

from ._upstream import MuseTrace, instrument_session, trace_context
from .errors import ProxyError
from .events import MuseMessage
from .logging import TRACE_LEVEL

logger = logging.getLogger(__name__)

_END = object()
_TERMINAL = {"completed", "failed", "errored", "cancelled", "canceled"}
_TERMINAL_GRACE = 0.05
_EMPTY_REPLY_GRACE = 1.0


class _Waiter:
    def __init__(self, request_id: str, text: str):
        self.request_id = request_id
        self.text = text
        self.messages: deque[Any] = deque()
        self.ready = asyncio.Event()
        self.bytes = 0
        self.acknowledged = False
        self.finished = False
        self.detached = False
        self.has_reply = False
        self.user_ids: set[str] = set()
        self.send_after = 0
        self.observed_work = False

    @property
    def live(self) -> bool:
        return not self.finished and not self.detached

    def put(self, item: Any) -> None:
        if not self.live:
            return
        size = len(item.text.encode("utf-8")) if isinstance(item, MuseMessage) else 0
        if isinstance(item, MuseMessage) and item.kind == "progress" and len(self.messages) >= 32:
            return
        if len(self.messages) >= 64 or self.bytes + size > 256 * 1024:
            self.finish(
                ProxyError("Muse reply buffer exceeded its limit.", code="muse_protocol_error")
            )
            return
        self.messages.append(item)
        self.bytes += size
        self.ready.set()

    def finish(self, item: Any = _END) -> None:
        if not self.live:
            return
        self.finished = True
        self.messages.append(item)
        if item is not _END:
            self.messages.append(_END)
        self.ready.set()

    async def get(self) -> Any:
        while not self.messages:
            self.ready.clear()
            await self.ready.wait()
        item = self.messages.popleft()
        if isinstance(item, MuseMessage):
            self.bytes -= len(item.text.encode("utf-8"))
        return item


class _ChatRouter:
    """Share a connection within one chat and isolate its lifecycle from other chats."""

    def __init__(self, client: Any, session_id: str | None):
        self.client = client
        self.trace_id = f"router-{uuid4().hex}"
        settings = getattr(client, "settings", None)
        self.trace = (
            MuseTrace(
                self.trace_id,
                secrets=(getattr(settings, "api_key", None), getattr(settings, "sdk_token", None)),
            )
            if logger.isEnabledFor(TRACE_LEVEL)
            else None
        )
        self.phase = "created"
        self.session_id = session_id
        self.waiters: list[_Waiter] = []
        self.inbox: asyncio.Queue = asyncio.Queue(maxsize=128)
        self.task: asyncio.Task | None = None
        self.closed = asyncio.Event()
        self.closing = False
        self.turn: Any = None
        self.guard: Any = None
        self.session: Any = None
        self.connection: asyncio.Task | None = None
        self.subscription: asyncio.Task | None = None
        self.submission: asyncio.Task | None = None
        self.iterator: Any = None
        self.sending: _Waiter | None = None
        self.latest: _Waiter | None = None
        self.users: dict[str, _Waiter] = {}
        self.parents: dict[str, str | None] = {}
        self.message_owners: dict[str, _Waiter | None] = {}
        self.received = 0
        self.current_sequence = 0
        self.terminal_sequence = 0
        self.emitted: set[str] = set()
        self.emitted_text: dict[str, str] = {}
        self.buffered: list[tuple[int, dict]] = []
        self.buffered_bytes = 0
        self.root_task: str | None = None
        self.running_tasks: set[str] = set()
        self.root_status: str | None = None
        self.terminal_at: float | None = None
        self.stop = asyncio.Event()
        self.scope: Any = None
        self.credentials: Any = None
        self.sdk: Any = None

    def _phase(self, phase: str) -> None:
        self.phase = phase
        logger.debug("Muse connection phase router_id=%s phase=%s", self.trace_id, phase)

    def add(self, waiter: _Waiter) -> None:
        if len(self.waiters) >= 64:
            raise ProxyError("Too many pending Muse requests.", status_code=429, code="muse_busy")
        self.waiters.append(waiter)
        if self.trace is not None:
            self.trace.emit("muse.route", request_id=waiter.request_id, session_id=self.session_id)
        self._wake()

    def _wake(self) -> None:
        with suppress(asyncio.QueueFull):
            self.inbox.put_nowait(None)

    def detach(self, waiter: _Waiter) -> None:
        waiter.detached = True
        waiter.messages.clear()
        waiter.bytes = 0
        self._wake()
        if (
            not self.closing
            and not any(item.live for item in self.waiters)
            and self.task is not None
        ):
            self.task.cancel()

    async def _consume(self, events: Any) -> None:
        async for event in events:
            if self.trace is not None:
                self.trace.emit("muse.event", request_id="-", body=event)
            # A broken or very fast upstream must not grow memory without bound.
            try:
                self.received += 1
                self.inbox.put_nowait((self.received, event))
            except asyncio.QueueFull:
                raise ProxyError(
                    "Muse event buffer exceeded its limit.", code="muse_protocol_error"
                ) from None

    async def _open_subscription(self) -> None:
        self._phase("subscribe")
        self.iterator = self.session.subscribe_chat(self.session_id)
        self.subscription = asyncio.create_task(self._consume(self.iterator))
        try:
            await self.client._ready(
                self.session.chat_subscribed, self.connection, self.subscription
            )
        except BaseException:
            self.subscription.cancel()
            await asyncio.gather(self.subscription, return_exceptions=True)
            with suppress(Exception):
                await self.iterator.aclose()
            self.subscription = None
            self.iterator = None
            raise

    def _start_send(self) -> None:
        if self.submission is not None:
            return
        waiter = next((item for item in self.waiters if item.live and not item.acknowledged), None)
        if waiter is None:
            return
        self._phase("send")
        self.sending = waiter
        waiter.send_after = self.received

        async def submit():
            with trace_context(self.trace, waiter.request_id):
                if self.trace is not None:
                    self.trace.emit(
                        "muse.chat.submit",
                        body={
                            "message": waiter.text,
                            "session_id": self.session_id,
                            "output_modality": "text",
                        },
                    )
                return await self.session.send_chat(
                    waiter.text, self.session_id, output_modality="text"
                )

        self.submission = asyncio.create_task(submit())

    def _event_size(self, event: dict) -> int:
        return len(json.dumps(event, ensure_ascii=False).encode("utf-8"))

    def _hold(self, item: tuple[int, dict]) -> None:
        size = self._event_size(item[1])
        if len(self.buffered) >= 64 or self.buffered_bytes + size > 256 * 1024:
            raise ProxyError(
                "Too many Muse events before acknowledgement.", code="muse_protocol_error"
            )
        self.buffered.append(item)
        self.buffered_bytes += size

    async def _acknowledge(self) -> None:
        submission, waiter = self.submission, self.sending
        if submission is None or waiter is None or not submission.done():
            return
        self.submission = self.sending = None
        self._phase("acknowledge")
        try:
            response = submission.result()
            if self.trace is not None:
                self.trace.emit(
                    "muse.chat.acknowledgement", request_id=waiter.request_id, body=response
                )
            if not isinstance(response, dict) or response.get("ok") is not True:
                status = response.get("status") if isinstance(response, dict) else None
                logger.warning(
                    "Muse refused chat router_id=%s upstream_status=%s",
                    self.trace_id,
                    status if type(status) is int else "-",
                )
                if status == 401:
                    await self.client._store.refresh(self.credentials, self.sdk.api, force=True)
                code = "muse_auth_error" if status in (401, 403) else "muse_request_error"
                raise ProxyError("Muse refused the chat request.", code=code)
            ack = response.get("response")
            if not isinstance(ack, dict):
                raise ValueError
            result = ack.get("result", ack)
            if not isinstance(result, dict):
                raise ValueError
            ids = {
                result[key]
                for key in ("message_id", "reply_to_message_id")
                if isinstance(result.get(key), str) and result[key]
            }
            if (
                not ids
                or any(len(identifier) > 512 for identifier in ids)
                or any(identifier in self.users for identifier in ids)
                or (
                    self.session_id
                    and (
                        result.get("session_id") != self.session_id
                        or result.get("is_thread") is not True
                    )
                )
            ):
                raise ValueError
            # Validate before SDK acknowledge: its follow-up mode mutates its ID set.
            # A failed acknowledgement never changes the current reply owner.
            if self.turn.acknowledged:
                self.guard._ingest_effects(self.turn.acknowledge(ack, follow_up=True))
                self.guard.user_ids.update(ids)
            else:
                self.guard.acknowledge(ack)
            waiter.user_ids = ids
            waiter.acknowledged = True
            self.users.update((identifier, waiter) for identifier in ids)
            if waiter.live:
                self.latest = waiter
            self.root_status = None
            self.terminal_at = None
            self._phase("waiting_replies")
        except ProxyError as error:
            waiter.finish(error)
        except ValueError:
            waiter.finish(
                ProxyError(
                    "Muse returned an invalid chat acknowledgement.", code="muse_protocol_error"
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.trace is not None:
                self.trace.failure("chat.acknowledge", exc, request_id=waiter.request_id)
            waiter.finish(ProxyError("Muse communication failed.", code="muse_connection_error"))
        held, self.buffered = self.buffered, []
        self.buffered_bytes = 0
        for sequence, event in held:
            self._feed(event, sequence)

    def _linked_owner(self, message_id: str) -> tuple[bool, _Waiter | None]:
        seen: set[str] = set()
        current = message_id
        while current not in seen:
            seen.add(current)
            if current in self.users:
                return True, self.users[current]
            if current in self.message_owners:
                return True, self.message_owners[current]
            if current not in self.parents:
                return True, None
            parent = self.parents.get(current)
            if not parent or parent == current:
                return False, None
            current = parent
        return True, None

    def _latest_live(self) -> _Waiter | None:
        if self.latest is None or not self.latest.live:
            self.latest = next(
                (item for item in reversed(self.waiters) if item.live and item.acknowledged), None
            )
        return self.latest

    def _emit_reply(self, reply: Any) -> None:
        if not reply.done or not reply.bound or not reply.text:
            return
        if reply.message_id in self.emitted:
            if self.emitted_text.get(reply.message_id) != reply.text:
                raise ProxyError(
                    "Muse revised a reply after it was streamed.", code="muse_protocol_error"
                )
            return
        linked, owner = self._linked_owner(reply.message_id)
        if not linked:
            owner = self._latest_live()
            if owner is not None and owner.live:
                for old in self.waiters:
                    if old is not owner and old.live and old.acknowledged:
                        old.finish(MuseMessage("cancel", superseded_by=owner.request_id))
        self.emitted.add(reply.message_id)
        self.emitted_text[reply.message_id] = reply.text
        self.message_owners[reply.message_id] = owner
        if owner is not None and self.current_sequence > owner.send_after:
            owner.observed_work = True
        if owner is None or not owner.live:
            return
        text = ("\n\n" if owner.has_reply else "") + reply.text
        owner.has_reply = True
        owner.put(MuseMessage("reply", text, message_id=reply.message_id))

    def _feed_parser(self, event: dict) -> None:
        if self.session_id:
            self.guard.feed(event)
            return
        payload = event["payload"]
        event_session = payload.get("session_id") or event.get("session_id")
        if (
            not self.turn.acknowledged
            or not self.turn.session_id
            or event_session != self.turn.session_id
        ):
            return
        name = event.get("event")
        parent = payload.get("reply_to_message_id") or payload.get("parent_message_id")
        message_id = payload.get("message_id") or event.get("message_id") or payload.get("id")
        status = payload.get("status")
        if name == "task.status" and status in _TERMINAL:
            task_id = payload.get("task_id")
            if task_id not in self.running_tasks and parent not in self.users:
                return
        if not parent or parent == message_id:
            latest = self._latest_live()
            if latest is None or not latest.user_ids:
                return
            # The main-chat SDK parser requires a parent to bind replies. Routing
            # still uses the original event below, so this is not a native link.
            payload = {**payload, "reply_to_message_id": next(iter(latest.user_ids))}
            event_for_parser = {**event, "payload": payload}
        else:
            event_for_parser = event
        self.guard._ingest_effects(self.turn.feed(event_for_parser), event)

    def _feed(self, event: dict, sequence: int) -> None:
        self.current_sequence = sequence
        if not isinstance(event, dict) or event.get("type") != "event":
            return
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        name = event.get("event")
        if self.session_id is None:
            self.sdk.ReplayScope.event_ids(event)
        message_id = payload.get("message_id") or event.get("message_id") or payload.get("id")
        before = self.turn.last_activity
        previous = self.turn.replies.get(message_id) if isinstance(message_id, str) else None
        self._feed_parser(event)
        changed = self.turn.last_activity != before
        if changed and name == "task.status":
            task_id = payload.get("task_id")
            task_id = task_id if isinstance(task_id, str) and task_id else None
            status = payload.get("status")
            root_id = payload.get("root_agent_id")
            if isinstance(root_id, str) and root_id and len(root_id) <= 512:
                self.root_task = root_id
            elif status == "running" and self.root_task is None:
                self.root_task = task_id
            if status == "running":
                if task_id:
                    self.running_tasks.add(task_id)
                self.terminal_at = None
                if task_id == self.root_task:
                    self.root_status = None
                    latest = self._latest_live()
                    if latest is not None and sequence > latest.send_after:
                        latest.observed_work = True
            elif status in _TERMINAL:
                if task_id:
                    self.running_tasks.discard(task_id)
                if self.root_task is None or task_id == self.root_task:
                    self.root_status = status
                    self.terminal_at = time.monotonic()
                    self.terminal_sequence = sequence
        while self.guard.progress:
            message = self.guard.progress.popleft()
            latest = self._latest_live()
            if latest is not None:
                latest.put(message)
        current = self.turn.replies.get(message_id) if isinstance(message_id, str) else None
        if current is not None and current is not previous:
            parent = payload.get("reply_to_message_id") or payload.get("parent_message_id")
            if isinstance(parent, str) and parent and parent != message_id:
                self.parents[message_id] = parent
            else:
                self.parents.setdefault(message_id, None)
            self._emit_reply(current)

    def _complete(self) -> bool:
        return bool(
            self.root_status in _TERMINAL
            and self.turn.task_finished
            and not self.turn.busy
            and not self.running_tasks
            and self.inbox.empty()
            and (
                self.root_status != "completed"
                or all(reply.done for reply in self.turn.replies.values())
            )
            and self.terminal_at is not None
            and time.monotonic() - max(self.terminal_at, self.turn.last_activity)
            >= (
                _EMPTY_REPLY_GRACE
                if self.root_status == "completed"
                and not any(reply.text for reply in self.turn.replies.values())
                else _TERMINAL_GRACE
            )
            and self.submission is None
            and not any(item.live and not item.acknowledged for item in self.waiters)
            and all(
                (self.root_status != "completed" or item.observed_work)
                and self.terminal_sequence > item.send_after
                for item in self.waiters
                if item.live and item.acknowledged
            )
        )

    async def run(self) -> None:
        with trace_context(self.trace):
            await self._run()

    async def _run(self) -> None:
        from .muse import _join_task

        error: ProxyError | None = None
        try:
            self._phase("waiting_session_lock")
            async with self.client._session_scope(self.session_id):
                try:
                    await self._serve()
                finally:
                    self.closing = True
                    # Own cleanup independently so request/shutdown cancellation cannot
                    # abandon the transport or release this session before retirement.
                    await _join_task(asyncio.create_task(self._cleanup()))
        except asyncio.CancelledError:
            raise
        except ProxyError as exc:
            if self.trace is not None:
                self.trace.failure(self.phase, exc)
            logger.warning(
                "Muse connection failed router_id=%s phase=%s", self.trace_id, self.phase
            )
            error = exc
        except Exception as exc:
            if self.trace is not None:
                self.trace.failure(self.phase, exc)
            logger.warning(
                "Muse connection failed router_id=%s phase=%s", self.trace_id, self.phase
            )
            error = ProxyError("Muse communication failed.", code="muse_connection_error")
        finally:
            self.closing = True
            if self.client._routers.get(self.session_id) is self:
                self.client._routers.pop(self.session_id, None)
            for waiter in self.waiters:
                if not waiter.live:
                    continue
                if error is not None:
                    waiter.finish(error)
                elif self.root_status in ("cancelled", "canceled"):
                    waiter.finish(MuseMessage("cancel"))
                elif self.root_status in ("failed", "errored"):
                    waiter.finish(
                        ProxyError(
                            "Muse did not complete the requested task.", code="muse_task_error"
                        )
                    )
                elif waiter.has_reply:
                    waiter.finish()
                else:
                    waiter.finish(
                        ProxyError(
                            "Muse completed the task without a text reply.", code="muse_empty_reply"
                        )
                    )
            self.closed.set()
            self._phase("closed")

    async def _serve(self) -> None:
        from .muse import _load_sdk, _refuse_command, _RequestTurn

        if self.client._closed:
            raise ProxyError(
                "Muse proxy is shutting down.", status_code=503, code="muse_unavailable"
            )
        self._phase("credentials_load")
        self.credentials = await self.client._store.load()
        self.sdk = _load_sdk()
        self._phase("vm_lookup")
        vm, self.credentials = await self.client._vm(self.sdk, self.credentials)
        device = self.sdk.DeviceDescription(
            node_id=self.credentials.node_id,
            display_name="Muse OpenAI Proxy",
            version=self.sdk.version,
            commands={},
        )
        self.session = self.sdk.LinkSession(
            noise_host=self.credentials.noise_host,
            vm_id=vm.get("vm_id") or vm["vm_name"],
            vm_auth_token=vm["vm_auth_token"],
            device=device,
            run_command=_refuse_command,
        )
        if self.trace is not None:
            self.trace.credentials(self.credentials)
            self.trace.add_secrets(vm.get("vm_auth_token"))
            instrument_session(self.session, self.trace)
        if self.session_id:
            self.scope = self.client._replay_scopes.get(self.session_id)
            if self.scope is None:
                self.scope = self.sdk.ReplayScope()
                self.client._replay_scopes[self.session_id] = self.scope
            # Muse reuses the root agent's task ID across otherwise separate turns.
            self.scope.ids = {item for item in self.scope.ids if item[0] == "message"}
            self.scope.order = deque(item for item in self.scope.order if item[0] == "message")
            self.client._replay_scopes.move_to_end(self.session_id)
            self.client._trim_replay_scopes()
            self.turn = self.sdk.MuseTurn(self.session_id, owns_chat=True, replay_scope=self.scope)
        else:
            self.turn = self.sdk.MuseTurn(None)
        self.guard = _RequestTurn(self.turn)
        self._phase("transport_connect_register")
        self.connection = asyncio.create_task(self.session.run(self.stop))
        await self.client._ready(self.session.registered, self.connection)
        try:
            await self._open_subscription()
        except ConnectionError as exc:
            if not self.session_id or getattr(exc, "status", None) != 404:
                raise
            self._phase("create_new_chat")
            self._start_send()
            expires = (
                asyncio.get_running_loop().time() + self.client.settings.connect_timeout_seconds
            )
            while True:
                if self.submission is not None and self.submission.done():
                    response = self.submission.result()
                    if not isinstance(response, dict) or response.get("ok") is not True:
                        await self._acknowledge()
                        raise ProxyError(
                            "Muse refused the chat request.", code="muse_request_error"
                        ) from None
                if asyncio.get_running_loop().time() >= expires:
                    raise ProxyError(
                        "Timed out opening the Muse chat.", status_code=504, code="muse_timeout"
                    ) from None
                try:
                    await self._open_subscription()
                    break
                except ConnectionError as retry:
                    if getattr(retry, "status", None) != 404:
                        raise
                    await asyncio.sleep(0.05)
        self._phase("waiting_replies")
        while any(item.live for item in self.waiters):
            await self._acknowledge()
            self._start_send()
            if self._complete():
                break
            if self.connection.done():
                raise self.client._connection_error(self.connection)
            if self.subscription.done():
                self.subscription.result()
                raise ProxyError(
                    "Muse closed the chat subscription before finishing.",
                    code="muse_connection_error",
                )
            try:
                item = await asyncio.wait_for(self.inbox.get(), timeout=0.01)
            except TimeoutError:
                continue
            if item is not None:
                if self.submission is not None:
                    self._hold(item)
                else:
                    self._feed(item[1], item[0])

    async def _cleanup(self) -> None:
        tasks = [task for task in (self.subscription, self.submission) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=1.0)
        self.stop.set()
        if self.connection is not None:
            self.connection.cancel()
            await asyncio.gather(self.connection, *tasks, return_exceptions=True)
        if self.iterator is not None:
            with suppress(Exception):
                await self.iterator.aclose()
        if self.turn is not None:
            self.turn.retire()
            if self.scope is not None:
                self.scope.ids = {item for item in self.scope.ids if item[0] == "message"}
                self.scope.order = deque(item for item in self.scope.order if item[0] == "message")
