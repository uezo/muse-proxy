"""Configure safe application logs without changing SDK or network logging."""

from __future__ import annotations

import logging
import sys

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def configure_logging(level: str = "info") -> None:
    """Install one CLI-owned stderr handler on the proxy's logging namespace."""
    logger = logging.getLogger("muse_proxy")
    for handler in logger.handlers[:]:
        if getattr(handler, "_muse_proxy_cli_handler", False):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(sys.stderr)
    handler._muse_proxy_cli_handler = True
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(TRACE_LEVEL if level.lower() == "trace" else level.upper())
    logger.propagate = False
