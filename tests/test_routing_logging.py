import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from muse_proxy._routing import _ChatRouter
from muse_proxy.errors import ProxyError


@pytest.mark.parametrize(
    "failure",
    [
        ProxyError("private exception text", code="private-code", param="private-param"),
        RuntimeError("private exception text"),
    ],
)
async def test_router_failure_logs_only_generated_id_and_fixed_phase(caplog, failure):
    client = SimpleNamespace(
        _lock=SimpleNamespace(acquire=AsyncMock(side_effect=failure)), _routers={}
    )
    router = _ChatRouter(client, "private-session-label")
    with caplog.at_level(logging.DEBUG, logger="muse_proxy._routing"):
        await router.run()
    records = [record for record in caplog.records if record.name == "muse_proxy._routing"]
    assert [record.levelno for record in records] == [logging.DEBUG, logging.WARNING, logging.DEBUG]
    assert all(f"router_id={router.trace_id}" in record.getMessage() for record in records)
    assert records[0].getMessage().endswith("phase=waiting_account_lock")
    assert records[1].getMessage().endswith("phase=waiting_account_lock")
    assert records[2].getMessage().endswith("phase=closed")
    assert "private" not in caplog.text
    assert router.closed.is_set()
