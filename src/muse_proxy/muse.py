"""Text transport using the official Muse Gadget SDK, without BLE setup."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from contextlib import aclosing, suppress
from types import SimpleNamespace
from typing import Any

from ._upstream import MuseTrace, current_trace, instrument_session, trace_context
from .credentials import CredentialStore
from .errors import ProxyError
from .events import MuseMessage
from .logging import TRACE_LEVEL


def _load_sdk() -> SimpleNamespace:
    """Import the SDK lazily so missing-dependency errors remain actionable."""
    try:
        from musegadget import __version__, muse_api
        from musegadget.link_client import DeviceDescription, LinkSession
        from musegadget.muse_turn import MuseTurn, ReplayScope
    except ImportError:
        raise ProxyError(
            "Muse SDK is unavailable; install muse-proxy with its dependencies.",
            status_code=503,
            code="muse_sdk_unavailable",
        ) from None
    return SimpleNamespace(
        api=muse_api,
        version=__version__,
        DeviceDescription=DeviceDescription,
        LinkSession=LinkSession,
        MuseTurn=MuseTurn,
        ReplayScope=ReplayScope,
    )


def _refuse_command(command: str, params: dict, timeout_ms: int | None) -> dict:
    return {
        "ok": False,
        "error": {"code": "unsupported_command", "message": "This proxy accepts text chat only."},
    }


def _chat_session_id(node_id: str, label: str) -> str:
    """Keep client labels stable while using the UUID required by Muse."""
    try:
        return str(uuid.UUID(label))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"muse-proxy/{node_id}/{label}"))


class _RequestTurn:
    """Constrain the SDK parser to the acknowledged request on a shared Muse chat."""

    def __init__(self, turn: Any):
        self.turn = turn
        self.user_ids: set[str] = set()
        self.tasks: set[str] = set()
        self.pending: list[dict] = []
        self.pending_bytes = 0
        self.acknowledged = False
        self.failed = False
        self.cancelled = False
        self.progress: deque[MuseMessage] = deque(maxlen=64)
        self._last_progress: tuple[str, str | None] | None = None
        self.emitted: set[str] = set()
        self.unbound: OrderedDict[str, list[tuple[dict, int]]] = OrderedDict()
        self.unbound_bytes = 0
        self.unbound_events = 0

    def _ingest_effects(self, effects: Any, event: dict | None = None) -> None:
        # Read only the SDK's accepted public events. Raw payloads can belong to a
        # different request or be replays, even when they carry a terminal status.
        from musegadget.muse_turn import AgentStatus, TaskFinished

        for effect in effects:
            if isinstance(effect, AgentStatus):
                code = effect.activity_code
                code = code.strip() if isinstance(code, str) else None
                text = (effect.activity_text or "").strip()
                if code in (
                    "idle",
                    "online",
                    "completed",
                    "failed",
                    "errored",
                    "cancelled",
                    "canceled",
                ):
                    continue
                text = text or code or ""
                if not text:
                    continue
                status = (text, code)
                if status != self._last_progress:
                    self.progress.append(MuseMessage("progress", text, activity_code=code))
                    self._last_progress = status
            elif isinstance(effect, TaskFinished) and event is not None:
                payload = event.get("payload", {})
                if event.get("event") == "task.status":
                    if payload.get("status") in ("cancelled", "canceled"):
                        self.cancelled = True
                    elif payload.get("status") in ("failed", "errored"):
                        self.failed = True

    def _buffer_unbound(self, message_id: str, event: dict) -> None:
        if len(message_id) > 512:
            return
        size = len(json.dumps(event).encode("utf-8"))
        self.unbound.setdefault(message_id, []).append((event, size))
        self.unbound_bytes += size
        self.unbound_events += 1
        while len(self.unbound) > 32 or self.unbound_events > 64 or self.unbound_bytes > 256 * 1024:
            _, discarded = self.unbound.popitem(last=False)
            self.unbound_bytes -= sum(size for _, size in discarded)
            self.unbound_events -= len(discarded)

    def _replay_unbound(self, message_id: str) -> None:
        held = self.unbound.pop(message_id, [])
        self.unbound_events -= len(held)
        self.unbound_bytes -= sum(size for _, size in held)
        for event, _ in held:
            self._ingest_effects(self.turn.feed(event), event)

    def acknowledge(self, response: dict) -> None:
        self._ingest_effects(self.turn.acknowledge(response))
        result = response.get("result", response)
        self.user_ids = {
            result[key]
            for key in ("message_id", "reply_to_message_id")
            if isinstance(result.get(key), str) and result[key]
        }
        self.acknowledged = True
        pending, self.pending = self.pending, []
        for event in pending:
            self.feed(event)

    def feed(self, event: dict) -> None:
        if event.get("type") != "event":
            return
        if not self.acknowledged:
            payload = event.get("payload")
            if not isinstance(payload, dict):
                return
            self.pending_bytes += sum(
                len(value.encode("utf-8")) for value in payload.values() if isinstance(value, str)
            )
            if len(self.pending) >= 64 or self.pending_bytes > 256 * 1024:
                raise ValueError("Too many Muse events before acknowledgement")
            self.pending.append(event)
            return
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        event_session = payload.get("session_id") or event.get("session_id")
        if event_session and self.turn.session_id and event_session != self.turn.session_id:
            return
        parent = payload.get("reply_to_message_id") or payload.get("parent_message_id")
        message_id = payload.get("message_id") or event.get("message_id") or payload.get("id")
        parent = parent if isinstance(parent, str) else None
        message_id = message_id if isinstance(message_id, str) else None
        if self.turn.owns_chat:
            self._ingest_effects(self.turn.feed(event), event)
            return
        reply = self.turn.replies.get(message_id)
        parent_reply = self.turn.replies.get(parent)
        linked = (
            parent in self.user_ids
            or (parent_reply is not None and parent_reply.bound)
            or (reply is not None and reply.bound)
        )
        name = event.get("event")
        if name in ("agent.status", "task.status"):
            task_id = payload.get("task_id")
            task_id = task_id if isinstance(task_id, str) else None
            if not linked and task_id not in self.tasks:
                return
            if isinstance(task_id, str) and task_id:
                self.tasks.add(task_id)
        elif not linked:
            # Keep drafts outside MuseTurn until a later event proves their parent chain.
            # Unrelated drafts cannot mark this turn busy or prevent it from completing.
            if (
                not parent
                and message_id
                and event.get("type") == "event"
                and name
                in (
                    "delta.message_start",
                    "delta.text_append",
                    "delta.message_done",
                    "message.assistant",
                )
            ):
                self._buffer_unbound(message_id, event)
            return
        if linked and message_id:
            self._replay_unbound(message_id)
        self._ingest_effects(self.turn.feed(event), event)

    def output_events(self) -> list[MuseMessage]:
        if not self.complete():
            return []
        messages = []
        for reply in self.turn.replies.values():
            if reply.bound and reply.done and reply.text and reply.message_id not in self.emitted:
                messages.append(
                    MuseMessage(
                        "reply",
                        ("\n\n" if self.emitted else "") + reply.text,
                        message_id=reply.message_id,
                    )
                )
                self.emitted.add(reply.message_id)
        return messages

    def output(self) -> list[str]:
        return [message.text for message in self.output_events()]

    def complete(self) -> bool:
        now = time.monotonic()
        return any(reply.text for reply in self.turn.replies.values()) and self.turn.complete(now)

    def empty(self) -> bool:
        return self.turn.settled(time.monotonic()) and not any(
            reply.text for reply in self.turn.replies.values()
        )


class MuseClient:
    def __init__(self, settings: Any):
        self.settings = settings
        self._store = CredentialStore(settings.state_dir, settings.sdk_token)
        self._lock = asyncio.Lock()
        self._closed = False
        self._active: set[asyncio.Task] = set()
        self._replay_scopes: OrderedDict[str, Any] = OrderedDict()
        self._routers: dict[str | None, Any] = {}

    async def close(self) -> None:
        self._closed = True
        current = asyncio.current_task()
        active = [task for task in self._active if task is not current]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        routes = [router.task for router in self._routers.values() if router.task is not None]
        for task in routes:
            task.cancel()
        if routes:
            await asyncio.gather(*routes, return_exceptions=True)
        self._replay_scopes.clear()

    async def stream_request(
        self, text: str, session_id: str | None = None, *, request_id: str
    ) -> AsyncIterator[MuseMessage]:
        """Route one HTTP request through the shared subscription for its Muse chat."""
        from ._routing import _END, _ChatRouter, _Waiter

        owner = asyncio.current_task()
        if owner is not None:
            self._active.add(owner)
        waiter = None
        router = None
        deadline = asyncio.get_running_loop().time() + self.settings.request_timeout_seconds
        try:
            if self._closed:
                raise ProxyError(
                    "Muse proxy is shutting down.", status_code=503, code="muse_unavailable"
                )
            effective = session_id or getattr(self.settings, "default_session_id", None)
            if effective:
                credentials = await self._store.load()
                effective = _chat_session_id(credentials.node_id, effective)
            router = self._routers.get(effective)
            if router is None or router.closed.is_set() or router.closing:
                router = _ChatRouter(self, effective)
                self._routers[effective] = router
            waiter = _Waiter(request_id, text)
            router.add(waiter)
            if router.task is None:
                router.task = asyncio.create_task(router.run())
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                item = await asyncio.wait_for(waiter.get(), timeout=remaining)
                if item is _END:
                    return
                if isinstance(item, ProxyError):
                    raise item
                yield item
        except TimeoutError:
            raise ProxyError(
                "Timed out waiting for Muse.", status_code=504, code="muse_timeout"
            ) from None
        except ProxyError:
            raise
        except Exception:
            raise ProxyError("Muse communication failed.", code="muse_connection_error") from None
        finally:
            if router is not None and waiter is not None:
                router.detach(waiter)
                if router.task is not None and not any(item.live for item in router.waiters):
                    while not router.task.done():
                        try:
                            await asyncio.shield(router.task)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not router.closed.is_set():
                        if self._routers.get(router.session_id) is router:
                            self._routers.pop(router.session_id, None)
                        router.closed.set()
            if owner is not None:
                self._active.discard(owner)

    async def stream_chat(self, text: str, session_id: str | None = None) -> AsyncIterator[str]:
        """Compatibility wrapper for callers that consume final reply text only."""
        async with aclosing(self.stream_events(text, session_id)) as events:
            async for message in events:
                if message.kind == "reply":
                    yield message.text
                elif message.kind == "cancel":
                    raise ProxyError(
                        "Muse did not complete the requested task.", code="muse_task_error"
                    )

    async def stream_events(
        self, text: str, session_id: str | None = None
    ) -> AsyncIterator[MuseMessage]:
        owner = asyncio.current_task()
        if owner is not None:
            self._active.add(owner)
        deadline = asyncio.get_running_loop().time() + self.settings.request_timeout_seconds
        acquired = False
        trace = current_trace()
        if trace is None and logging.getLogger("muse_proxy.upstream").isEnabledFor(TRACE_LEVEL):
            trace = MuseTrace(
                f"legacy-{uuid.uuid4().hex}",
                secrets=(getattr(self.settings, "api_key", None), self.settings.sdk_token),
            )
        try:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError
            await asyncio.wait_for(self._lock.acquire(), timeout=remaining)
            acquired = True
            if self._closed:
                raise ProxyError(
                    "Muse proxy is shutting down.", status_code=503, code="muse_unavailable"
                )
            effective_session = session_id or getattr(self.settings, "default_session_id", None)
            async with aclosing(self._request_events(text, effective_session)) as reply:
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError
                    with trace_context(trace):
                        pending = asyncio.create_task(anext(reply))
                    try:
                        chunk = await asyncio.wait_for(pending, timeout=remaining)
                    except StopAsyncIteration:
                        return
                    except asyncio.CancelledError as cancelled:
                        pending.cancel()
                        while not pending.done():
                            try:
                                await asyncio.shield(pending)
                            except asyncio.CancelledError:
                                continue
                            except Exception:
                                break
                        if not pending.cancelled():
                            with suppress(Exception):
                                pending.result()
                        raise cancelled
                    # No timeout context spans this yield: backpressure belongs to the caller.
                    yield chunk
        except TimeoutError:
            raise ProxyError(
                "Timed out waiting for Muse.", status_code=504, code="muse_timeout"
            ) from None
        except ProxyError:
            raise
        except Exception as exc:
            if trace is not None:
                trace.failure("stream_events", exc)
            # SDK/network exceptions can contain token-bearing URLs or response content.
            raise ProxyError("Muse communication failed.", code="muse_connection_error") from None
        finally:
            if acquired:
                self._lock.release()
            if owner is not None:
                self._active.discard(owner)

    async def _vm(self, sdk: Any, credentials: Any) -> tuple[dict, Any]:
        credentials = await self._store.refresh(credentials, sdk.api)
        trace = current_trace()

        async def fetch():
            if trace is not None:
                trace.credentials(credentials)
                trace.emit(
                    "muse.api.request",
                    protocol="https",
                    method="GET",
                    path="/fetch_vms",
                    api_root=credentials.api_root,
                    body=None,
                )
            try:
                vms, status = await asyncio.to_thread(
                    sdk.api.fetch_vms_with_status,
                    credentials.pairing["access_token"],
                    credentials.api_root,
                )
            except Exception as exc:
                if trace is not None:
                    trace.failure("fetch_vms", exc, protocol="https")
                raise
            if trace is not None:
                for vm in vms if isinstance(vms, (list, tuple)) else ():
                    if isinstance(vm, dict):
                        trace.add_secrets(vm.get("vm_auth_token"))
                trace.emit(
                    "muse.api.response",
                    protocol="https",
                    path="/fetch_vms",
                    http_status=status,
                    body=vms,
                    body_availability="sdk_result",
                )
            return vms, status

        vms, status = await fetch()
        if status == 401:
            credentials = await self._store.refresh(credentials, sdk.api, force=True)
            vms, status = await fetch()
        if status == 401:
            raise ProxyError(
                "Muse rejected the device credentials.", status_code=503, code="muse_auth_error"
            )
        if not isinstance(status, int) or not 200 <= status < 300:
            raise ProxyError("Could not fetch a Muse connection.", code="muse_connection_error")
        candidates = [vm for vm in vms if isinstance(vm, dict)]
        if self.settings.vm_id:
            vm = next(
                (
                    vm
                    for vm in candidates
                    if self.settings.vm_id in (vm.get("vm_id"), vm.get("vm_name"))
                ),
                None,
            )
        else:
            vm = next(
                (vm for vm in candidates if vm.get("is_default")),
                candidates[0] if candidates else None,
            )
        if not vm or not vm.get("vm_auth_token") or not (vm.get("vm_id") or vm.get("vm_name")):
            raise ProxyError(
                "No matching Muse is available.", status_code=503, code="muse_unavailable"
            )
        return vm, credentials

    @staticmethod
    def _connection_error(task: asyncio.Task) -> ProxyError:
        if not task.cancelled():
            with suppress(Exception):
                outcome = task.result()
                value = getattr(outcome, "value", outcome)
                if value in ("auth_rejected", "forbidden", "unpaired"):
                    return ProxyError(
                        "Muse rejected this connection; check the imported pairing.",
                        status_code=503,
                        code="muse_auth_error",
                    )
        return ProxyError(
            "Muse disconnected before completing the reply.", code="muse_connection_error"
        )

    async def _ready(
        self,
        event: asyncio.Event,
        connection: asyncio.Task,
        subscription: asyncio.Task | None = None,
    ) -> None:
        ready = asyncio.create_task(event.wait())
        try:
            watch = {ready, connection}
            if subscription is not None:
                watch.add(subscription)
            done, _ = await asyncio.wait(
                watch,
                timeout=self.settings.connect_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if connection in done:
                raise self._connection_error(connection)
            if subscription in done:
                subscription.result()
                raise ProxyError("Muse closed the chat subscription.", code="muse_connection_error")
            if ready not in done:
                raise ProxyError(
                    "Timed out connecting to Muse.", status_code=504, code="muse_timeout"
                )
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)

    async def _request_events(
        self, text: str, session_id: str | None
    ) -> AsyncIterator[MuseMessage]:
        credentials = await self._store.load()
        sdk = _load_sdk()
        vm, credentials = await self._vm(sdk, credentials)
        if session_id:
            session_id = _chat_session_id(credentials.node_id, session_id)
        device = sdk.DeviceDescription(
            node_id=credentials.node_id,
            display_name="Muse OpenAI Proxy",
            version=sdk.version,
            commands={},
        )
        session = sdk.LinkSession(
            noise_host=credentials.noise_host,
            vm_id=vm.get("vm_id") or vm["vm_name"],
            vm_auth_token=vm["vm_auth_token"],
            device=device,
            run_command=_refuse_command,
        )
        trace = current_trace()
        if trace is not None:
            trace.credentials(credentials)
            trace.add_secrets(vm.get("vm_auth_token"))
            instrument_session(session, trace)
        if session_id:
            scope = self._replay_scopes.get(session_id)
            if scope is None:
                scope = sdk.ReplayScope()
                self._replay_scopes[session_id] = scope
            self._replay_scopes.move_to_end(session_id)
            while len(self._replay_scopes) > 32:
                self._replay_scopes.popitem(last=False)
            parser = sdk.MuseTurn(session_id, owns_chat=True, replay_scope=scope)
        else:
            parser = sdk.MuseTurn(None)
        turn = _RequestTurn(parser)
        stop = asyncio.Event()
        connection = asyncio.create_task(session.run(stop))
        subscription = None
        submission = None
        iterator = None

        async def consume(events) -> None:
            async for event in events:
                turn.feed(event)

        async def open_subscription() -> None:
            nonlocal iterator, subscription
            iterator = session.subscribe_chat(session_id)
            subscription = asyncio.create_task(consume(iterator))
            try:
                await self._ready(session.chat_subscribed, connection, subscription)
            except BaseException:
                subscription.cancel()
                await asyncio.gather(subscription, return_exceptions=True)
                with suppress(Exception):
                    await iterator.aclose()
                subscription = None
                iterator = None
                raise

        try:
            await self._ready(session.registered, connection)
            try:
                await open_subscription()
            except ConnectionError as exc:
                if not session_id or getattr(exc, "status", None) != 404:
                    raise
                # A new thread exists only after its first message is accepted. Start that
                # single submission, then subscribe as soon as the thread becomes available.
                submission = asyncio.create_task(
                    session.send_chat(text, session_id, output_modality="text")
                )
                expires = asyncio.get_running_loop().time() + self.settings.connect_timeout_seconds
                while True:
                    if submission.done():
                        response = submission.result()
                        if not isinstance(response, dict) or response.get("ok") is not True:
                            break
                    if asyncio.get_running_loop().time() >= expires:
                        raise ProxyError(
                            "Timed out opening the Muse chat.", status_code=504, code="muse_timeout"
                        ) from None
                    try:
                        await open_subscription()
                        break
                    except ConnectionError as retry:
                        if getattr(retry, "status", None) != 404:
                            raise
                        await asyncio.sleep(0.05)
            # Never resubmit a chat request: a lost ACK may still mean it ran.
            response = (
                await submission
                if submission is not None
                else await session.send_chat(text, session_id, output_modality="text")
            )
            if not isinstance(response, dict) or response.get("ok") is not True:
                status = response.get("status") if isinstance(response, dict) else None
                if status == 401:
                    # Persist a refreshed pair for the next request, without resending this one.
                    await self._store.refresh(credentials, sdk.api, force=True)
                code = "muse_auth_error" if status in (401, 403) else "muse_request_error"
                raise ProxyError("Muse refused the chat request.", code=code)
            acknowledgment = response.get("response")
            if not isinstance(acknowledgment, dict):
                raise ProxyError(
                    "Muse returned an invalid chat acknowledgement.", code="muse_protocol_error"
                )
            try:
                turn.acknowledge(acknowledgment)
            except ValueError:
                raise ProxyError(
                    "Muse returned an invalid chat acknowledgement.", code="muse_protocol_error"
                ) from None
            while True:
                while turn.progress:
                    yield turn.progress.popleft()
                if turn.failed:
                    raise ProxyError(
                        "Muse did not complete the requested task.", code="muse_task_error"
                    )
                if turn.cancelled:
                    chunks = [MuseMessage("cancel")]
                    break
                if turn.complete():
                    chunks = turn.output_events()
                    break
                if turn.empty():
                    raise ProxyError(
                        "Muse completed the task without a text reply.", code="muse_empty_reply"
                    )
                if connection.done():
                    raise self._connection_error(connection)
                if subscription.done():
                    subscription.result()
                    raise ProxyError(
                        "Muse closed the chat subscription before finishing.",
                        code="muse_connection_error",
                    )
                await asyncio.wait(
                    {connection, subscription}, timeout=0.05, return_when=asyncio.FIRST_COMPLETED
                )
        finally:
            tasks = [task for task in (subscription, submission) if task is not None]
            for task in tasks:
                task.cancel()
            if tasks:
                # Let stream resets reach Muse before closing the transport. If a send is
                # stuck, closing the connection below releases it without an endless join.
                await asyncio.wait(tasks, timeout=1.0)
            stop.set()
            connection.cancel()
            await asyncio.gather(connection, *tasks, return_exceptions=True)
            if iterator is not None:
                with suppress(Exception):
                    await iterator.aclose()
            turn.turn.retire()
        # Close upstream resources before exposing final text to a potentially slow consumer.
        for chunk in chunks:
            yield chunk
