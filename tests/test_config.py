import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from muse_proxy.config import Settings, write_config


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.delenv("MUSE_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("MUSEGADGET_SDK_TOKEN", raising=False)


def test_generated_configuration_is_private_and_roundtrips(tmp_path):
    path = write_config(tmp_path / "proxy.toml")
    settings = Settings.from_file(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(settings.api_key) >= 32
    assert settings.api_key not in repr(settings)
    assert settings.state_dir == tmp_path / "data/muse"
    assert settings.sdk_token is None
    assert settings.vm_id is None
    assert not settings.state_dir.exists()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8766
    assert settings.model == "muse"
    assert settings.default_session_id == "muse-proxy"


def test_init_never_overwrites_existing_file_or_symlink(tmp_path):
    path = tmp_path / "proxy.toml"
    path.write_text("existing configuration")
    with pytest.raises(FileExistsError):
        write_config(path)
    assert path.read_text() == "existing configuration"
    dangling_target = tmp_path / "missing.toml"
    symlink = tmp_path / "alias.toml"
    symlink.symlink_to(dangling_target)
    with pytest.raises(FileExistsError):
        write_config(symlink)
    assert not dangling_target.exists()


def test_state_directory_uses_configuration_directory_not_cwd(tmp_path, monkeypatch):
    configuration_dir = tmp_path / "configuration"
    configuration_dir.mkdir()
    path = write_config(configuration_dir / "proxy.toml")
    monkeypatch.chdir(tmp_path)
    assert Settings.from_file(path).state_dir == configuration_dir / "data/muse"


def test_state_path_expands_home(tmp_path):
    path = tmp_path / "proxy.toml"
    path.write_text('api_key = "test-key-at-least-16"\nstate_dir = "~/credentials/muse"\n')
    assert Settings.from_file(path).state_dir == Path.home() / "credentials/muse"


def test_environment_overrides_only_named_secrets(tmp_path, monkeypatch):
    path = write_config(tmp_path / "proxy.toml")
    monkeypatch.setenv("MUSE_PROXY_API_KEY", "replacement-key-at-least-16")
    monkeypatch.setenv("MUSEGADGET_SDK_TOKEN", "mgst_test_token")
    monkeypatch.setenv("MUSE_PROXY_HOST", "0.0.0.0")
    settings = Settings.from_file(path)
    assert settings.api_key == "replacement-key-at-least-16"
    assert settings.sdk_token == "mgst_test_token"
    assert settings.host == "127.0.0.1"
    assert settings.sdk_token not in repr(settings)


def test_blank_environment_override_does_not_silently_use_file_key(tmp_path, monkeypatch):
    path = write_config(tmp_path / "proxy.toml")
    monkeypatch.setenv("MUSE_PROXY_API_KEY", "")
    with pytest.raises(ValidationError):
        Settings.from_file(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("port", 0),
        ("port", 65536),
        ("port", True),
        ("request_timeout_seconds", 0),
        ("request_timeout_seconds", -1),
        ("request_timeout_seconds", float("nan")),
        ("request_timeout_seconds", float("inf")),
        ("request_timeout_seconds", True),
        ("connect_timeout_seconds", 0),
        ("connect_timeout_seconds", float("inf")),
        ("connect_timeout_seconds", True),
        ("max_request_bytes", 0),
        ("max_request_bytes", 1.5),
        ("max_message_chars", -1),
        ("max_message_chars", True),
        ("model", " "),
        ("model", "m" * 129),
        ("host", ""),
        ("sdk_token", ""),
        ("vm_id", ""),
        ("unknown_option", "value"),
    ],
)
def test_invalid_configuration_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(api_key="test-key-at-least-16", **{field: value})


def test_secret_values_are_hidden_in_validation_errors():
    with pytest.raises(ValidationError) as caught:
        Settings(api_key="short-secret", sdk_token="mgst_private_token")
    assert "short-secret" not in str(caught.value)
    assert "mgst_private_token" not in str(caught.value)


def test_existing_configuration_without_default_session_keeps_main_chat(tmp_path):
    path = tmp_path / "proxy.toml"
    path.write_text('api_key = "test-key-at-least-16"\n')
    assert Settings.from_file(path).default_session_id is None
    assert Settings(api_key="test-key-at-least-16").default_session_id is None


@pytest.mark.parametrize("session_id", [None, "muse-proxy", "a", "AZaz09-", "-" * 64])
def test_valid_default_session_id(session_id):
    settings = Settings(api_key="test-key-at-least-16", default_session_id=session_id)
    assert settings.default_session_id == session_id


@pytest.mark.parametrize("session_id", ["", "a" * 65, "あ", "a_b", "a b", "a/b", "a\n", "a.b"])
def test_invalid_default_session_id(session_id):
    with pytest.raises(ValidationError):
        Settings(api_key="test-key-at-least-16", default_session_id=session_id)
