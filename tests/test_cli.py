import stat
import sys
from types import ModuleType

import pytest

from muse_proxy import cli
from muse_proxy.config import Settings, write_config


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.delenv("MUSE_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("MUSEGADGET_SDK_TOKEN", raising=False)


@pytest.fixture
def config(tmp_path):
    return write_config(tmp_path / "proxy.toml")


def test_init_creates_private_file_without_printing_api_key(tmp_path, capsys):
    path = tmp_path / "proxy.toml"
    assert cli.main(["init", "--config", str(path)]) == 0
    settings = Settings.from_file(path)
    output = capsys.readouterr()
    assert str(path) in output.out
    assert settings.api_key not in output.out + output.err
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not settings.state_dir.exists()


def test_init_refuses_overwrite(config, capsys):
    before = config.read_bytes()
    assert cli.main(["init", "--config", str(config)]) == 1
    assert config.read_bytes() == before
    assert "FileExistsError" in capsys.readouterr().err


@pytest.mark.parametrize("config_position", ["root", "command"])
def test_serve_uses_one_worker_and_disables_access_logs(config, monkeypatch, config_position):
    sentinel = object()
    seen = []
    app_module = ModuleType("muse_proxy.app")
    uvicorn_module = ModuleType("uvicorn")

    def create_app(settings):
        seen.append(("create", settings))
        return sentinel

    def run(app, **kwargs):
        seen.append(("run", app, kwargs))

    app_module.create_app = create_app
    uvicorn_module.run = run
    monkeypatch.setitem(sys.modules, "muse_proxy.app", app_module)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)
    config_args = ["--config", str(config)]
    argv = config_args + ["serve"] if config_position == "root" else ["serve"] + config_args
    assert cli.main(argv) == 0
    assert seen[0] == ("create", Settings.from_file(config))
    assert seen[1] == (
        "run",
        sentinel,
        {"host": "127.0.0.1", "port": 8766, "workers": 1, "access_log": False},
    )


def test_init_default_configuration_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init"]) == 0
    assert (tmp_path / "proxy.toml").exists()


def test_invalid_config_does_not_print_secrets(tmp_path, capsys):
    path = tmp_path / "proxy.toml"
    path.write_text('api_key = "short-secret"\nsdk_token = "mgst_private_token"\n')
    assert cli.main(["serve", "--config", str(path)]) == 2
    output = capsys.readouterr()
    assert "api_key" in output.err
    assert "short-secret" not in output.out + output.err
    assert "mgst_private_token" not in output.out + output.err


def test_invalid_toml_does_not_print_line_containing_secret(tmp_path, capsys):
    path = tmp_path / "proxy.toml"
    path.write_text('api_key = "private_secret" "broken syntax\n')
    assert cli.main(["serve", "--config", str(path)]) == 2
    output = capsys.readouterr()
    assert "Invalid TOML" in output.err
    assert "private_secret" not in output.out + output.err


def test_missing_config_returns_useful_error(tmp_path, capsys):
    assert cli.main(["serve", "--config", str(tmp_path / "missing.toml")]) == 1
    assert "FileNotFoundError" in capsys.readouterr().err
