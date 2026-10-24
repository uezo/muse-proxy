"""Read portable Muse Gadget state and persist token rotations privately."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ._upstream import current_trace
from .errors import ProxyError

TOKEN_REFRESH_AGE_SECONDS = 3 * 3600
DEFAULT_NOISE_HOST = "hatch.metaaivm.com"
_MAC = re.compile(r"[0-9a-f]{2}(?::[0-9a-f]{2}){5}")
_SDK_TOKEN = re.compile(r"mgst_[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]")


@dataclass(frozen=True)
class Credentials:
    node_id: str
    pairing: dict[str, Any] = field(repr=False)
    sdk_token: str | None = field(default=None, repr=False)

    @property
    def api_root(self) -> str:
        return (self.pairing.get("api_url_v2") or "https://api.muse.ai").rstrip("/")

    @property
    def noise_host(self) -> str:
        return self.pairing.get("noise_host") or DEFAULT_NOISE_HOST


def _state_error(message: str) -> ProxyError:
    return ProxyError(message, status_code=503, code="muse_credentials_error")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise _state_error(
            f"Cannot read valid {path.name}; import the existing Muse pairing state."
        ) from None
    if not isinstance(data, dict):
        raise _state_error(f"Invalid {path.name}; expected a JSON object.")
    return data


def _load(directory: Path, configured_sdk_token: str | None) -> Credentials:
    identity = _read_object(directory / "identity.json")
    mac = identity.get("mac")
    if not isinstance(mac, str) or not _MAC.fullmatch(mac):
        raise _state_error("Invalid identity.json: a saved Muse Gadget MAC identifier is required.")
    pairing = _read_object(directory / "pairing.json")
    for key in ("access_token", "refresh_token"):
        if not isinstance(pairing.get(key), str) or not pairing[key].strip():
            raise _state_error(f"Invalid pairing.json: {key} is required.")
    saved_at = pairing.get("access_token_saved_at", 0)
    if (
        isinstance(saved_at, bool)
        or not isinstance(saved_at, (int, float))
        or not math.isfinite(saved_at)
        or saved_at < 0
    ):
        raise _state_error("Invalid pairing.json: access_token_saved_at must be a timestamp.")
    root = pairing.get("api_url_v2") or "https://api.muse.ai"
    if not isinstance(root, str):
        raise _state_error("Invalid pairing.json: api_url_v2 must be an HTTPS URL.")
    try:
        parsed = urlsplit(root)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        host = pairing.get("noise_host") or DEFAULT_NOISE_HOST
        if not isinstance(host, str) or any(c.isspace() for c in host):
            raise ValueError
        noise = urlsplit("wss://" + host)
        if (
            not noise.hostname
            or noise.path
            or noise.query
            or noise.fragment
            or noise.username
            or noise.password
        ):
            raise ValueError
        _ = noise.port  # validate a configured port without including it in errors
    except ValueError:
        raise _state_error("Invalid pairing.json: invalid API URL or Noise host.") from None
    token = configured_sdk_token or os.environ.get("MUSEGADGET_SDK_TOKEN")
    if not token:
        try:
            token = (directory / "sdk_token").read_text(encoding="utf-8")
        except FileNotFoundError:
            token = None
        except (OSError, ValueError):
            raise _state_error("Cannot read the saved SDK token.") from None
    if token is not None:
        token = token.strip() or None
    if token and not _SDK_TOKEN.fullmatch(token):
        raise _state_error("Invalid SDK token; use the original token from gadgets.muse.ai.")
    hint = pairing.get("sdk_token_hint")
    if hint is not None and hint != (token[:12] if token else "none"):
        raise _state_error(
            "The SDK token differs from the saved pairing; import the original SDK token."
        )
    return Credentials("homelink-" + mac.replace(":", "")[-6:], pairing, token)


def _save_pairing(directory: Path, pairing: dict[str, Any]) -> None:
    """Replace only pairing.json, without following a predictable temporary path."""
    fd, filename = tempfile.mkstemp(prefix=".pairing-", suffix=".tmp", dir=directory)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(pairing, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, directory / "pairing.json")
    finally:
        temporary.unlink(missing_ok=True)


class CredentialStore:
    def __init__(self, state_dir: Path, sdk_token: str | None = None):
        self.directory = state_dir
        self._sdk_token = sdk_token
        self._refresh_lock = asyncio.Lock()

    async def load(self) -> Credentials:
        credentials = await asyncio.to_thread(_load, self.directory, self._sdk_token)
        if trace := current_trace():
            trace.credentials(credentials)
        return credentials

    async def refresh(
        self, credentials: Credentials, api: Any, *, force: bool = False
    ) -> Credentials:
        async with self._refresh_lock:
            latest = await self.load()
            if force and any(
                latest.pairing[key] != credentials.pairing[key]
                for key in ("access_token", "refresh_token")
            ):
                # Another session already refreshed the token that this caller used.
                return latest
            if latest == credentials:
                latest = credentials
            return await self._refresh_current(latest, api, force=force)

    async def _refresh_current(
        self, credentials: Credentials, api: Any, *, force: bool
    ) -> Credentials:
        if (
            not force
            and time.time() - credentials.pairing.get("access_token_saved_at", 0)
            < TOKEN_REFRESH_AGE_SECONDS
        ):
            return credentials

        def rotate() -> Credentials:
            trace = current_trace()
            if trace is not None:
                trace.credentials(credentials)
                trace.emit(
                    "muse.api.request",
                    protocol="https",
                    method="POST",
                    path="/device_token/refresh",
                    api_root=credentials.api_root,
                    body={"device_id": credentials.node_id, "sdk_token": credentials.sdk_token},
                )
            try:
                tokens, status = api.refresh_device_token(
                    credentials.pairing["refresh_token"],
                    credentials.node_id,
                    credentials.api_root,
                    credentials.sdk_token,
                )
            except Exception as exc:
                if trace is not None:
                    trace.failure("device_token.refresh", exc, protocol="https")
                raise
            if trace is not None:
                if isinstance(tokens, dict):
                    trace.credentials(
                        Credentials(credentials.node_id, tokens, credentials.sdk_token)
                    )
                trace.emit(
                    "muse.api.response",
                    protocol="https",
                    path="/device_token/refresh",
                    http_status=status,
                    body=tokens,
                    body_availability="sdk_result",
                )
            if not isinstance(tokens, dict) or not all(
                isinstance(tokens.get(key), str) and tokens[key].strip()
                for key in ("access_token", "refresh_token")
            ):
                if status == 401:
                    raise _state_error(
                        "Muse rejected the refresh token; pair again and import the new state."
                    )
                raise ProxyError(
                    "Muse token refresh failed; the saved state has been preserved.",
                    status_code=502,
                    code="muse_auth_error",
                )
            pairing = {
                **credentials.pairing,
                "access_token": tokens["access_token"],
                "refresh_token": tokens["refresh_token"],
                "access_token_saved_at": int(time.time()),
            }
            try:
                _save_pairing(self.directory, pairing)
            except OSError:
                raise _state_error(
                    "Cannot persist refreshed Muse credentials; check state directory permissions."
                ) from None
            return Credentials(credentials.node_id, pairing, credentials.sdk_token)

        # A cancelled request must still save tokens that the server has rotated.
        rotation = asyncio.create_task(asyncio.to_thread(rotate))
        try:
            return await asyncio.shield(rotation)
        except asyncio.CancelledError as cancelled:
            while not rotation.done():
                try:
                    await asyncio.shield(rotation)
                except asyncio.CancelledError:
                    # Repeated shutdown cancellation must not release the refresh lock
                    # while this worker can still rotate and replace credentials.
                    continue
                except Exception:
                    break
            if not rotation.cancelled():
                # Retrieve any failure, preserving the original caller cancellation.
                with suppress(Exception):
                    rotation.result()
            raise cancelled
