import logging
import stat
import sys
from types import ModuleType

import pytest

from muse_proxy import cli
from muse_proxy.config import Settings, write_config
from muse_proxy.logging import TRACE_LEVEL, configure_logging


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.delenv("MUSE_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("MUSEGADGET_SDK_TOKEN", raising=False)
    logger = logging.getLogger("muse_proxy")
    saved = (logger.level, logger.propagate, logger.handlers[:])
    yield
    for handler in logger.handlers[:]:
        if handler not in saved[2]:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(saved[0])
    logger.propagate = saved[1]
    logger.handlers[:] = saved[2]


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
        {
            "host": "127.0.0.1",
            "port": 8766,
            "workers": 1,
            "access_log": False,
            "log_level": "info",
        },
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


@pytest.mark.parametrize("level", ["trace", "debug", "info", "warning", "error", "critical"])
def test_serve_passes_selected_log_level(config, monkeypatch, level):
    app_module = ModuleType("muse_proxy.app")
    uvicorn_module = ModuleType("uvicorn")
    seen = []
    app_module.create_app = lambda settings: object()
    uvicorn_module.run = lambda app, **kwargs: seen.append(kwargs)
    monkeypatch.setitem(sys.modules, "muse_proxy.app", app_module)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)

    assert cli.main(["serve", "--config", str(config), "--log-level", level]) == 0
    assert seen[0]["log_level"] == ("info" if level in {"trace", "debug"} else level)
    assert seen[0]["access_log"] is False
    assert logging.getLogger("muse_proxy").level == (
        TRACE_LEVEL if level == "trace" else getattr(logging, level.upper())
    )


@pytest.mark.parametrize("level", ["trace", "debug"])
def test_diagnostics_do_not_enable_uvicorn_debug(config, monkeypatch, level):
    app_module = ModuleType("muse_proxy.app")
    uvicorn_module = ModuleType("uvicorn")
    seen = []
    app_module.create_app = lambda settings: object()
    uvicorn_module.run = lambda app, **kwargs: seen.append(kwargs)
    monkeypatch.setitem(sys.modules, "muse_proxy.app", app_module)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)

    assert cli.main(["serve", "--config", str(config), "--log-level", level]) == 0
    assert logging.getLogger("muse_proxy").level == (
        TRACE_LEVEL if level == "trace" else logging.DEBUG
    )
    assert seen[0]["log_level"] == "info"
    assert seen[0]["access_log"] is False


@pytest.mark.parametrize(
    "argv", [["serve", "--log-level", "verbose"], ["init", "--log-level", "debug"]]
)
def test_log_level_rejects_unknown_levels_and_init(argv):
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


def test_logging_goes_to_stderr_without_duplicate_handlers(capsys):
    configure_logging("debug")
    configure_logging("debug")
    logging.getLogger("muse_proxy.app").info("request_start method=POST route=/v1/chat/completions")
    logging.getLogger("muse_proxy.app").debug("safe diagnostic")

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.count("request_start") == 1
    assert " INFO muse_proxy.app request_start" in output.err
    assert " DEBUG muse_proxy.app safe diagnostic" in output.err


def test_logging_level_filters_proxy_logs(capsys):
    configure_logging("warning")
    logger = logging.getLogger("muse_proxy.app")
    logger.info("hidden message")
    logger.warning("visible message")
    output = capsys.readouterr()
    assert "hidden message" not in output.err
    assert "visible message" in output.err


@pytest.mark.parametrize("level", ["trace", "debug"])
def test_logging_leaves_root_sdk_and_network_loggers_unchanged(level):
    loggers = [logging.getLogger(name) for name in ("", "musegadget", "httpx", "websockets")]
    before = [(logger.level, logger.propagate, logger.handlers[:]) for logger in loggers]
    configure_logging(level)
    after = [(logger.level, logger.propagate, logger.handlers[:]) for logger in loggers]
    assert after == before


@pytest.mark.parametrize("level,trace_visible", [("trace", True), ("debug", False)])
def test_trace_is_more_detailed_than_debug(level, trace_visible, capsys):
    configure_logging(level)
    logger = logging.getLogger("muse_proxy.http")
    logger.log(TRACE_LEVEL, "trace payload")
    logger.debug("debug diagnostic")
    output = capsys.readouterr()
    assert (" TRACE muse_proxy.http trace payload" in output.err) is trace_visible
    assert " DEBUG muse_proxy.http debug diagnostic" in output.err


def test_trace_configuration_preserves_one_cli_handler(capsys):
    configure_logging("trace")
    configure_logging("trace")
    logging.getLogger("muse_proxy.http").log(TRACE_LEVEL, "one trace payload")
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.count("one trace payload") == 1


def test_trace_level_name_is_registered_for_embedded_logging():
    assert TRACE_LEVEL < logging.DEBUG
    assert logging.getLevelName(TRACE_LEVEL) == "TRACE"
    assert logging.getLevelName("TRACE") == TRACE_LEVEL


def test_serve_help_explains_trace_logs_bodies(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--help"])
    assert error.value.code == 0
    assert "trace logs request/response bodies" in " ".join(capsys.readouterr().out.split())
