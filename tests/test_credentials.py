import asyncio
import json
import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from muse_proxy.credentials import TOKEN_REFRESH_AGE_SECONDS, CredentialStore
from muse_proxy.errors import ProxyError

SDK_TOKEN = "mgst_" + "A" * 43


def state(directory: Path, **changes):
    pairing = {
        "access_token": "test-access",
        "refresh_token": "test-refresh",
        "access_token_saved_at": int(time.time()),
        "sdk_token_hint": SDK_TOKEN[:12],
        "custom_future_field": {"keep": True},
        **changes,
    }
    (directory / "pairing.json").write_text(json.dumps(pairing))
    (directory / "identity.json").write_text(json.dumps({"mac": "02:01:02:ab:cd:ef"}))
    (directory / "sdk_token").write_text(SDK_TOKEN)
    return pairing


@pytest.mark.asyncio
async def test_portable_identity_and_private_repr(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSEGADGET_SDK_TOKEN", raising=False)
    pairing = state(tmp_path)
    credentials = await CredentialStore(tmp_path).load()
    assert credentials.node_id == "homelink-abcdef"
    assert credentials.pairing == pairing
    assert credentials.sdk_token == SDK_TOKEN
    assert "test-access" not in repr(credentials)
    assert SDK_TOKEN not in repr(credentials)
    assert credentials.api_root == "https://api.muse.ai"
    assert credentials.noise_host == "hatch.metaaivm.com"


@pytest.mark.asyncio
async def test_missing_identity_does_not_generate_or_modify_state(tmp_path):
    with pytest.raises(ProxyError, match="identity.json"):
        await CredentialStore(tmp_path).load()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename,content",
    [
        ("identity.json", '{"mac":"not-a-mac"}'),
        ("pairing.json", "[]"),
        ("pairing.json", '{"access_token":"secret-that-must-not-leak"}'),
        ("pairing.json", "broken-secret-that-must-not-leak"),
    ],
)
async def test_bad_state_errors_are_sanitized(tmp_path, filename, content):
    state(tmp_path)
    (tmp_path / filename).write_text(content)
    with pytest.raises(ProxyError) as caught:
        await CredentialStore(tmp_path, SDK_TOKEN).load()
    assert "secret-that-must-not-leak" not in str(caught.value)
    assert (tmp_path / filename).read_text() == content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"api_url_v2": "http://example.invalid"},
        {"api_url_v2": "https://secret@example.invalid"},
        {"noise_host": "example.invalid/path"},
        {"access_token_saved_at": "not-a-time"},
        {"access_token_saved_at": float("nan")},
    ],
)
async def test_invalid_endpoint_or_timestamp(tmp_path, change):
    state(tmp_path, **change)
    with pytest.raises(ProxyError):
        await CredentialStore(tmp_path, SDK_TOKEN).load()


@pytest.mark.asyncio
async def test_sdk_token_must_match_the_original_pairing(tmp_path):
    state(tmp_path)
    with pytest.raises(ProxyError, match="differs"):
        await CredentialStore(tmp_path, "mgst_" + "B" * 42 + "A").load()


@pytest.mark.asyncio
async def test_refresh_at_three_hours_rotates_both_and_preserves_unknown_fields(tmp_path):
    state(tmp_path, access_token_saved_at=time.time() - TOKEN_REFRESH_AGE_SECONDS)
    store = CredentialStore(tmp_path, SDK_TOKEN)
    credentials = await store.load()
    calls = []

    def refresh(*args):
        calls.append(args)
        assert threading.current_thread() is not threading.main_thread()
        return {"access_token": "new-access", "refresh_token": "new-refresh"}, 200

    updated = await store.refresh(credentials, SimpleNamespace(refresh_device_token=refresh))
    saved = json.loads((tmp_path / "pairing.json").read_text())
    assert updated.pairing == saved
    assert saved["access_token"] == "new-access"
    assert saved["refresh_token"] == "new-refresh"
    assert saved["custom_future_field"] == {"keep": True}
    assert calls == [("test-refresh", "homelink-abcdef", "https://api.muse.ai", SDK_TOKEN)]
    assert stat.S_IMODE((tmp_path / "pairing.json").stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".pairing-*.tmp"))


@pytest.mark.asyncio
async def test_fresh_token_not_refreshed_unless_forced(tmp_path):
    state(tmp_path)
    store = CredentialStore(tmp_path, SDK_TOKEN)
    credentials = await store.load()
    calls = []

    def refresh(*args):
        calls.append(args)
        return {"access_token": "new-access", "refresh_token": "new-refresh"}, 200

    api = SimpleNamespace(refresh_device_token=refresh)
    assert await store.refresh(credentials, api) is credentials
    assert not calls
    await store.refresh(credentials, api, force=True)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_concurrent_expired_snapshots_refresh_once(tmp_path):
    state(tmp_path, access_token_saved_at=0)
    store = CredentialStore(tmp_path, SDK_TOKEN)
    first_snapshot, second_snapshot = await asyncio.gather(store.load(), store.load())
    started = threading.Event()
    release = threading.Event()
    calls = []

    def refresh(*args):
        calls.append(args)
        started.set()
        release.wait(timeout=2)
        return {"access_token": "new-access", "refresh_token": "new-refresh"}, 200

    api = SimpleNamespace(refresh_device_token=refresh)
    first = asyncio.create_task(store.refresh(first_snapshot, api))
    assert await asyncio.to_thread(started.wait, 1)
    second = asyncio.create_task(store.refresh(second_snapshot, api))
    try:
        await asyncio.sleep(0)
        assert not second.done()
        assert len(calls) == 1
    finally:
        release.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert first_result == second_result
    assert second_result.pairing["refresh_token"] == "new-refresh"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_forced_refresh_of_stale_snapshot_reuses_latest_tokens(tmp_path):
    state(tmp_path)
    store = CredentialStore(tmp_path, SDK_TOKEN)
    snapshot = await store.load()
    calls = []

    def refresh(*args):
        calls.append(args)
        return {"access_token": "new-access", "refresh_token": "new-refresh"}, 200

    api = SimpleNamespace(refresh_device_token=refresh)
    updated = await store.refresh(snapshot, api, force=True)
    assert await store.refresh(snapshot, api, force=True) == updated
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 500, None])
async def test_rejected_refresh_preserves_state(tmp_path, status):
    state(tmp_path, access_token_saved_at=0)
    original = (tmp_path / "pairing.json").read_bytes()
    store = CredentialStore(tmp_path, SDK_TOKEN)
    api = SimpleNamespace(refresh_device_token=lambda *args: (None, status))
    with pytest.raises(ProxyError):
        await store.refresh(await store.load(), api)
    assert (tmp_path / "pairing.json").read_bytes() == original


@pytest.mark.asyncio
async def test_refresh_lock_released_after_error(tmp_path):
    state(tmp_path, access_token_saved_at=0)
    store = CredentialStore(tmp_path, SDK_TOKEN)
    credentials = await store.load()
    rejected_api = SimpleNamespace(refresh_device_token=lambda *args: (None, 500))
    with pytest.raises(ProxyError):
        await store.refresh(credentials, rejected_api)

    working_api = SimpleNamespace(
        refresh_device_token=lambda *args: (
            {"access_token": "new-access", "refresh_token": "new-refresh"},
            200,
        )
    )
    updated = await asyncio.wait_for(store.refresh(credentials, working_api), timeout=1)
    assert updated.pairing["refresh_token"] == "new-refresh"


@pytest.mark.asyncio
async def test_cancelled_refresh_still_persists_rotated_tokens(tmp_path):
    state(tmp_path, access_token_saved_at=0)
    store = CredentialStore(tmp_path, SDK_TOKEN)
    credentials = await store.load()
    started = threading.Event()
    release = threading.Event()

    def refresh(*args):
        started.set()
        release.wait(timeout=2)
        return {"access_token": "rotated-access", "refresh_token": "rotated-refresh"}, 200

    task = asyncio.create_task(
        store.refresh(credentials, SimpleNamespace(refresh_device_token=refresh))
    )
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert json.loads((tmp_path / "pairing.json").read_text())["refresh_token"] == "rotated-refresh"


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_rotation_before_releasing_lock(tmp_path):
    state(tmp_path, access_token_saved_at=0)
    store = CredentialStore(tmp_path, SDK_TOKEN)
    credentials = await store.load()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def refresh(*args):
        calls.append(args)
        started.set()
        release.wait(timeout=2)
        return {"access_token": "rotated-access", "refresh_token": "rotated-refresh"}, 200

    api = SimpleNamespace(refresh_device_token=refresh)
    task = asyncio.create_task(store.refresh(credentials, api))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    next_request = asyncio.create_task(store.refresh(credentials, api, force=True))
    try:
        await asyncio.sleep(0)
        assert not task.done()
        assert not next_request.done()
        assert len(calls) == 1
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    updated = await asyncio.wait_for(next_request, timeout=1)
    assert updated.pairing["refresh_token"] == "rotated-refresh"
    assert len(calls) == 1
    assert json.loads((tmp_path / "pairing.json").read_text())["refresh_token"] == "rotated-refresh"
