"""Create private configuration and run the Muse proxy server."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence

from pydantic import ValidationError

from .config import Settings, write_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="muse-proxy", description="Expose Muse through an OpenAI-compatible text chat API."
    )
    parser.add_argument("--config", default="proxy.toml", help="Configuration file")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("init", "Create a private configuration with a generated proxy API key"),
        ("serve", "Run the Muse proxy HTTP server"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", default=argparse.SUPPRESS, help="Configuration file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            path = write_config(args.config)
            print(f"Created {path}")
            print("Place existing Muse pairing files in state_dir before sending chat requests.")
            return 0

        settings = Settings.from_file(args.config)
        import uvicorn

        from .app import create_app

        uvicorn.run(
            create_app(settings),
            host=settings.host,
            port=settings.port,
            workers=1,
            access_log=False,
        )
        return 0
    except ValidationError as exc:
        for item in exc.errors(include_input=False, include_url=False, include_context=False):
            print(f"{'.'.join(map(str, item['loc']))}: {item['msg']}", file=sys.stderr)
        return 2
    except tomllib.TOMLDecodeError:
        print("Invalid TOML configuration; check its syntax.", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
