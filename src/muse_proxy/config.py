"""Private TOML configuration and portable Muse credential paths."""

from __future__ import annotations

import json
import os
import secrets
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    """Single Muse account configuration; environment overrides apply in from_file."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, allow_inf_nan=False)

    api_key: str = Field(min_length=16, repr=False)
    state_dir: Path = Path("data/muse")
    host: str = "127.0.0.1"
    port: int = Field(default=8766, ge=1, le=65535, strict=True)
    model: str = Field(default="muse", min_length=1, max_length=128)
    default_session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]{1,64}$")
    sdk_token: str | None = Field(default=None, min_length=1, repr=False)
    vm_id: str | None = Field(default=None, min_length=1)
    request_timeout_seconds: float = Field(default=180.0, gt=0, strict=True)
    connect_timeout_seconds: float = Field(default=30.0, gt=0, strict=True)
    max_request_bytes: int = Field(default=300 * 1024, gt=0, strict=True)
    max_message_chars: int = Field(default=16000, gt=0, strict=True)

    @field_validator("api_key", "host", "model", "sdk_token", "vm_id")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("state_dir")
    @classmethod
    def expand_state_dir(cls, value: Path) -> Path:
        return value.expanduser()

    @classmethod
    def from_file(cls, path: str | Path = "proxy.toml") -> Settings:
        """Load TOML and resolve state_dir relative to that file, not the process cwd."""
        path = Path(path).expanduser().resolve()
        with path.open("rb") as source:
            values = tomllib.load(source)
        for field, variable in (
            ("api_key", "MUSE_PROXY_API_KEY"),
            ("sdk_token", "MUSEGADGET_SDK_TOKEN"),
        ):
            if variable in os.environ:
                values[field] = os.environ[variable]
        settings = cls.model_validate(values)
        if not settings.state_dir.is_absolute():
            settings.state_dir = (path.parent / settings.state_dir).resolve()
        return settings


def write_config(path: str | Path = "proxy.toml") -> Path:
    """Create a private config with a generated proxy key, refusing any overwrite."""
    path = Path(path).expanduser().absolute()
    settings = Settings(api_key=secrets.token_urlsafe(32), default_session_id="muse-proxy")
    values = settings.model_dump(mode="json", exclude_none=True)
    content = (
        "# Keep this file private. Restart muse-proxy after changing settings.\n"
        "# api_key protects the proxy API; it is separate from Muse credentials.\n"
        "# state_dir is relative to this configuration file.\n\n"
        + "\n".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in values.items()
        )
        + "\n\n# Muse SDK token: prefer the MUSEGADGET_SDK_TOKEN environment variable.\n"
        '# sdk_token = "mgst_REPLACE_ME"\n'
        '# vm_id = "OPTIONAL_VM_ID"\n'
    )
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        output.write(content)
    return path
