from unittest.mock import AsyncMock, patch

import pytest

from predictiveguard.db import get_postgres_health


@pytest.mark.asyncio
async def test_get_postgres_health() -> None:
    connection = AsyncMock()
    connection.fetchval.return_value = "17.6"

    with patch("predictiveguard.db._connect", new=AsyncMock(return_value=connection)):
        result = await get_postgres_health()

    assert result.name == "postgresql"
    assert result.status == "healthy"
    assert result.version == "17.6"
    assert isinstance(result.response_time_ms, float)
    assert result.detail is None
    connection.fetchval.assert_awaited_once_with("SHOW server_version")
    connection.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_postgres_health_when_connection_fails() -> None:
    with patch(
        "predictiveguard.db._connect",
        new=AsyncMock(side_effect=TimeoutError),
    ):
        result = await get_postgres_health()

    assert result.name == "postgresql"
    assert result.status == "unavailable"
    assert result.version is None
    assert isinstance(result.response_time_ms, float)
    assert result.detail == "TimeoutError"
